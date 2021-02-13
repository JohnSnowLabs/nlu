import logging
logger = logging.getLogger('nlu')
import nlu
import nlu.pipe_components
import sparknlp
from sparknlp.base import *
from sparknlp.base import LightPipeline
from sparknlp.annotator import *
import pyspark
from pyspark.sql.types import ArrayType, FloatType, DoubleType
from pyspark.sql.functions import col as pyspark_col
from pyspark.sql.functions import  explode,   monotonically_increasing_id, greatest, expr,udf,array
import pandas as pd
import numpy as np
from  typing import List

from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from pyspark.sql import functions as F
from pyspark.sql import types as t






class BasePipe(dict):
    # we inherhit from dict so the pipe is indexable and we have a nice shortcut for accessing the spark nlp model
    def __init__(self):
        self.nlu_ref=''
        self.raw_text_column = 'text'
        self.raw_text_matrix_slice = 1  # place holder for getting text from matrix
        self.spark_nlp_pipe = None
        self.has_trainable_components = False
        self.needs_fitting = True
        self.is_fitted = False
        self.output_positions = False  # Wether to putput positions of Features in the final output. E.x. positions of tokens, entities, dependencies etc.. inside of the input document.
        self.output_level = ''  # either document, chunk, sentence, token
        self.output_different_levels = True
        self.light_pipe_configured = False
        self.spark_non_light_transformer_pipe = None
        self.pipe_components = []  # orderd list of nlu_component objects
        self.output_datatype = 'pandas'  # What data type should be returned after predict either spark, pandas, modin, numpy, string or array
        self.lang = 'en'
    def isInstanceOfNlpClassifer(self, model):
        '''
        Check for a given Spark NLP model if it is an instance of a classifier , either approach or already fitted transformer will return true
        This is used to configured the input/output columns based on the inputs
        :param model: the model to check
        :return: True if it is one of the following classes : (ClassifierDLModel,ClassifierDLModel,MultiClassifierDLModel,MultiClassifierDLApproach,SentimentDLModel,SentimentDLApproach) )
        '''
        return isinstance(model, (
            ClassifierDLModel, ClassifierDLModel, MultiClassifierDLModel, MultiClassifierDLApproach, SentimentDLModel,
            SentimentDLApproach))

    def configure_outputs(self, component, nlu_reference):
        '''
        Configure output column names of classifiers from category to something more meaningful
        Name should be Name of classifier, based on NLU reference.
        Duplicate names will be resolved by appending suffix "_i" to column name, based on how often we encounterd duplicate errors
        This updates component infos accordingly
        :param component: classifier component for which the output columns to  configured
        :param nlu_reference: nlu reference from which is component stemmed
        :return: None
        '''
        if nlu_reference == 'default_name' : return
        nlu_reference = nlu_reference.replace('train.', '')
        model_meta = nlu.extract_classifier_metadata_from_nlu_ref(nlu_reference)
        can_use_name = False
        new_output_name = model_meta[0]
        i = 0
        while can_use_name == False:
            can_use_name = True
            for c in self.pipe_components:
                if new_output_name in c.component_info.spark_input_column_names + c.component_info.spark_output_column_names and c.component_info.name != component.component_info.name:
                    can_use_name = False
        if can_use_name == False:
            new_output_name = new_output_name + '_' + str(i)
            i += 1
        # classifiers always have just 1 output col
        logger.info(f"Configured output columns name to {new_output_name} for classifier in {nlu_reference}")
        component.model.setOutputCol(new_output_name)
        component.component_info.spark_output_column_names = [new_output_name]

    def add(self, component, nlu_reference="default_name", pretrained_pipe_component=False):
        '''

        :param component:
        :param nlu_reference: NLU references, passed for components that are used specified and not automatically generate by NLU
        :return:
        '''
        self.nlu_reference = nlu_reference
        self.pipe_components.append(component)
        # ensure that input/output cols are properly set
        component.__set_missing_model_attributes__()
        # Spark NLP model reference shortcut
        name = component.component_info.name.replace(' ', '').replace('train.','')
        logger.info(f"Adding {name} to internal pipe")

        # Configure output column names of classifiers from category to something more meaningful
        if self.isInstanceOfNlpClassifer(component.model): self.configure_outputs(component, nlu_reference)

        # Add Component as self.index and in attributes
        if 'embed' in component.component_info.type and nlu_reference not in self.keys() and not pretrained_pipe_component:
            new_output_column = nlu_reference
            new_output_column = new_output_column.replace('.', '_')
            component.nlu_reference = nlu_reference
            component.model.setOutputCol(new_output_column)
            component.component_info.spark_output_column_names = [new_output_column]
            component.component_info.name = new_output_column
            self[new_output_column] = component.model
        # name parsed from component info, dont fiddle with column names of components unwrapped from pretrained pipelines
        elif name not in self.keys():
            component.nlu_reference = nlu_reference
            self[name] = component.model
        else:  # default name applied
            new_output_column = name + '@' + nlu_reference
            new_output_column = new_output_column.replace('.', '_')
            component.nlu_reference = nlu_reference
            component.model.setOutputCol(new_output_column)
            component.component_info.spark_output_column_names = [new_output_column]
            component.component_info.name = new_output_column
            self[new_output_column] = component.model
        # self.component_execution_plan.update()

class NLUPipeline(BasePipe):
    def __init__(self):
        super().__init__()
        """ Initializes a pretrained pipeline         """
        self.spark = sparknlp.start()
        self.provider = 'sparknlp'
        self.pipe_ready = False  # ready when we have created a spark df
        # The NLU pipeline uses  types of Spark NLP annotators to identify how to handle different columns
        self.levels = {
            'token': ['token', 'pos', 'ner', 'lemma', 'lem', 'stem', 'stemm', 'word_embeddings', 'named_entity',
                      'entity', 'dependency',
                      'labeled_dependency', 'dep', 'dep.untyped', 'dep.typed'],
            'sentence': ['sentence', 'sentence_embeddings', ] + ['sentiment', 'classifer', 'category'],
            'chunk': ['chunk', 'embeddings_chunk', 'chunk_embeddings'],
            'document': ['document', 'language'],
            'embedding_level': []
            # ['sentiment', 'classifer'] # WIP, wait for Spark NLP Getter/Setter fixes to implement this properly
            # embedding level  annotators output levels depend on the level of the embeddings they are fed. If we have Doc/Chunk/Word/Sentence embeddings, those annotators output at the same level.

        }


        self.annotator_levels_approach_based = {
            'document': [DocumentAssembler, Chunk2Doc,
                         YakeModel,
                         ],
            'sentence': [SentenceDetector, SentenceDetectorDLApproach, ],
            'chunk': [Chunker, ChunkEmbeddings,  ChunkTokenizer, Token2Chunk, TokenAssembler,
                      NerConverter, Doc2Chunk,NGramGenerator],
            'token': [ NerCrfApproach, NerDLApproach,
                       PerceptronApproach,
                       Stemmer,
                       ContextSpellCheckerApproach,
                       nlu.WordSegmenter,
                       Lemmatizer, TypedDependencyParserApproach, DependencyParserApproach,
                       Tokenizer, RegexTokenizer, RecursiveTokenizer
                ,StopWordsCleaner, DateMatcher, TextMatcher, BigTextMatcher, MultiDateMatcher,
                       WordSegmenterApproach
                       ],
            # sub token is when annotator is token based but some tokens may be missing since dropped/cleanes
            # are matchers chunk or sub token?
            # 'sub_token': [StopWordsCleaner, DateMatcher, TextMatcher, BigTextMatcher, MultiDateMatcher],
            # these can be document or sentence
            'input_dependent': [ViveknSentimentApproach, SentimentDLApproach, ClassifierDLApproach,
                                LanguageDetectorDL,
                                MultiClassifierDLApproach,  SentenceEmbeddings, NorvigSweetingApproach,
                                ],

            # 'unclassified': [Yake, Ngram]
        }


        self.annotator_levels_model_based = {
            'document': [],
            'sentence': [SentenceDetectorDLModel, ],
            'chunk': [ChunkTokenizerModel, ChunkTokenizerModel, ],
            'token': [ContextSpellCheckerModel, AlbertEmbeddings, BertEmbeddings, ElmoEmbeddings, WordEmbeddings,
                      XlnetEmbeddings, WordEmbeddingsModel,
                      NerDLModel, NerCrfModel, PerceptronModel, SymmetricDeleteModel, NorvigSweetingModel,
                      ContextSpellCheckerModel,
                      TypedDependencyParserModel, DependencyParserModel,
                      RecursiveTokenizerModel,
                      TextMatcherModel, BigTextMatcherModel, RegexMatcherModel,
                      WordSegmenterModel
                      ],
            # 'sub_token': [TextMatcherModel, BigTextMatcherModel, RegexMatcherModel, ],
            'input_dependent': [BertSentenceEmbeddings, UniversalSentenceEncoder, ViveknSentimentModel,
                                SentimentDLModel, MultiClassifierDLModel, MultiClassifierDLModel, ClassifierDLModel,
                                MarianTransformer,T5Transformer

                                ],
        }

        self.all_embeddings = {
            'token' : [AlbertEmbeddings, BertEmbeddings, ElmoEmbeddings, WordEmbeddings,
                       XlnetEmbeddings,WordEmbeddingsModel],
            'input_dependent' : [SentenceEmbeddings, UniversalSentenceEncoder,BertSentenceEmbeddings]

        }

    def get_sample_spark_dataframe(self):
        data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami']}
        text_df = pd.DataFrame(data)
        return sparknlp.start().createDataFrame(data=text_df)

    def verify_all_labels_exist(self,dataset):
        #todo
        return True
        # pass

    def fit(self, dataset=None, dataset_path=None, label_seperator=','):
        # if dataset is  string with '/' in it, its dataset path!
        '''
        Converts the input Pandas Dataframe into a Spark Dataframe and trains a model on it.
        :param dataset: The pandas dataset to train on, should have a y column for label and 'text' column for text features
        :param dataset_path: Path to a CONLL2013 format dataset. It will be read for NER and POS training.
        :param label_seperator: If multi_classifier is trained, this seperator is used to split the elements into an Array column for Pyspark
        :return: A nlu pipeline with models fitted.
        '''
        self.is_fitted = True
        stages = []
        for component in self.pipe_components:
            stages.append(component.model)
        self.spark_estimator_pipe = Pipeline(stages=stages)

        if dataset_path != None and 'ner' in self.nlu_ref:
            from sparknlp.training import CoNLL
            s_df = CoNLL().readDataset(self.spark,path=dataset_path, )
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(s_df.withColumnRenamed('label','y'))

        elif dataset_path != None and 'pos' in self.nlu_ref:
            from sparknlp.training import POS
            s_df = POS().readDataset(self.spark,path=dataset_path,delimiter=label_seperator,outputPosCol="y",outputDocumentCol="document",outputTextCol="text")
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(s_df)
        elif isinstance(dataset,pd.DataFrame) and 'multi' in  self.nlu_ref:
            schema = StructType([
                StructField("y", StringType(), True),
                StructField("text", StringType(), True)
                ])
            from pyspark.sql import functions as F
            df = self.spark.createDataFrame(data=dataset, schema=schema).withColumn('y',F.split('y',label_seperator))
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(df)

        elif isinstance(dataset,pd.DataFrame):
            if not self.verify_all_labels_exist(dataset) : return nlu.NluError()
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(self.convert_pd_dataframe_to_spark(dataset))

        elif isinstance(dataset,pd.DataFrame) :
            if not self.verify_all_labels_exist(dataset) : return nlu.NluError()
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(self.convert_pd_dataframe_to_spark(dataset))

        else :
            # fit on empty dataframe since no data provided
            logger.info('Fitting on empty Dataframe, could not infer correct training method. This is intended for non-trainable pipelines.')
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(self.get_sample_spark_dataframe())


        return self
    def convert_pd_dataframe_to_spark(self, data):
        #optimize
        return nlu.spark.createDataFrame(data)
    #todo rm
    def get_output_level_of_embeddings_provider(self, field_type, field_name):
        '''
        This function will go through all components to find the component which  generate @component_output_column_name.
        Then it will go gain through all components to find the component, from which @component_output_column_name is taking its inputs
        Then it will return the type of the provider component. This result isused to resolve the output level of the component that depends on the inpit for the output level
        :param field_type: The type of the field we want to resolve the input level for
        :param field_name: The name of the field we want to resolve the input level for

        :return:
        '''
        # find the component. Column output name should be unique
        component_inputs = []
        for component in self.pipe_components:
            if field_name == component.component_info.name:
                component_inputs = component.component_info.spark_input_column_names

        # get the embedding feature name
        target_output_component = ''
        for input_name in component_inputs:
            if 'embed' in input_name: target_output_component = input_name

        # get the model that outputs that feature
        for component in self.pipe_components:
            component_outputs = component.component_info.spark_output_column_names
            for input_name in component_outputs:
                if target_output_component == input_name:
                    # this is the component that feeds into the component we are trying to resolve the output  level for.
                    # That is so, because the output of this component matches the input of the component we are resolving
                    return self.resolve_type_to_output_level(component.component_info.type)


    #todo rm
    def resolve_type_to_output_level(self, field_type, field_name):
        '''
        This checks the levels dict for what the output level is for the input annotator type.
        If the annotator type depends on the embedding level, we need further checking.
        @ param field_type : type of the spark field
        @ param name : name of thhe spark field
        @ return : String, which corrosponds to the output level of this Component.
        '''
        logger.info('Resolving output level for field_type=%s and field_name=%s', field_type, field_name)
        if field_name == 'sentence':
            logger.info('Resolved output level for field_type=%s and field_name=%s to Sentence level', field_type,
                        field_name)
            return 'sentence'
        if field_type in self.levels['token']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to Token level ', field_type,
                        field_name)
            return 'token'
        if field_type in self.levels['sentence']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to sentence level', field_type,
                        field_name)
            return 'sentence'
        if field_type in self.levels['chunk']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to Chunk level ', field_type,
                        field_name)
            return 'chunk'
        if field_type in self.levels['document']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to document level', field_type,
                        field_name)
            return 'document'
        if field_type in self.levels['embedding_level']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to embeddings level', field_type,
                        field_name)
            return self.get_output_level_of_embeddings_provider(field_type, field_name)  # recursive resolution


    def get_field_types_dict(self, sdf, stranger_features, keep_stranger_features=True):
        """
        @ param sdf: Spark Dataframe which a NLU/SparkNLP pipeline has transformed.
        This function returns a dictionary that maps column names to their spark annotator types.
        @return : Dictionary, Keys are spark column column names, value is the type of annotator
        """
        logger.info('Getting field types for output SDF')
        field_types_dict = {}

        for field in sdf.schema.fieldNames():
            logger.info(f'Parsing field for {field}')

            if not keep_stranger_features and field in stranger_features: continue
            elif field in stranger_features :
                field_types_dict[field] = 'document'
                continue
            if field == 'origin_index':
                field_types_dict[field] = 'document'
                continue

            if field == self.raw_text_column: continue
            # if 'label' in field: continue  # speciel case for input lables
            # For empty DF this will crash
            a_row = sdf.select(field + '.annotatorType').take(1)[0]['annotatorType']
            if len(a_row) > 0:
                a_type = a_row[0]
            else:
                logger.exception(
                    'Error there are no rows for this Component in the final Dataframe. For field=%s. It will be dropped in the final dataset',
                    field)
                a_type = 'Error'  # (no results)
            field_types_dict[field] = a_type
            logger.info('Parsed type=%s  for field=%s', a_type, field)
        logger.info('Parsing field types done, parsed=%s', field_types_dict)
        return field_types_dict

    def reorder_column_names(self, fields_to_rename:List[str]) -> List[str]:
        '''
        Edge case swap. We must rename .metadata fields before we get the .result fields or there will be errors because of column name overwrites.. So we swap position of them
        and second analogus edge case for positional fields (.begin and .end) and .result. We will put every rseult column into the end of the list and thus avoid the erronous case always
        :param column_names:
        :return:
        '''
        # edge case swap. We must rename .metadata fields before we get the .result fields or there will be errors because of column name overwrites.. So we swap position of them
        cols_to_swap = [field for field in fields_to_rename if '.metadata' in field]
        reorderd_fields_to_rename = fields_to_rename.copy()
        for swap in cols_to_swap:
            name = swap.split('.')[0] + '.result'

            if 'ar_embed_aner_300d' in name : # AR ANER edge edge case
                reorderd_fields_to_rename.append(swap)
                continue
            reorderd_fields_to_rename[reorderd_fields_to_rename.index(swap)], reorderd_fields_to_rename[
                reorderd_fields_to_rename.index(name)] = reorderd_fields_to_rename[
                                                             reorderd_fields_to_rename.index(name)], \
                                                         reorderd_fields_to_rename[
                                                             reorderd_fields_to_rename.index(swap)]
            logger.info('Swapped selection order for  %s and %s before renaming ', swap, name)

        # second analogus edge case for positional fields (.begin and .end) and .result. We will put every rseult column into the end of the list and thus avoid the erronous case always
        for col in reorderd_fields_to_rename:
            if '.result' in col: reorderd_fields_to_rename.append(
                reorderd_fields_to_rename.pop(reorderd_fields_to_rename.index(col)))

        return reorderd_fields_to_rename
    def rename_columns_and_extract_map_values_same_level(self, ptmp, fields_to_rename, same_output_level,
                                                         stranger_features=[], meta=False):
        '''
        Extract features of Spark DF after they where exploded it was exploded
        :param ptmp: The dataframe which contains the columns wto be renamed
        :param fields_to_rename: A list of field names that will be renamed in the dataframe.
        :param same_output_level: Wether the fields that are going to be renamed are at the same output level as the pipe or at a different one.
        :param stranger_features:
        :param meta: Wether  to get meta data like prediction confidence or not
        :return: Returns tuple (list, SparkDataFrame), where the first element is a list with all the new names and the second element is a new Spark Dataframe which contains all the renamed and also old columns
        '''

        logger.info(
            'Renaming columns and extracting meta data for  outputlevel_same=%s and fields_to_rename=%s and get_meta=%s',
            same_output_level, fields_to_rename, meta)
        columns_for_select = []
        reorderd_fields_to_rename = self.reorder_column_names(fields_to_rename)


        # fields that are at the same output level have been exploded.
        # thus we ened to use the res.1 etc.. reference to get the map values and keys
        for i, field in enumerate(reorderd_fields_to_rename):
            if field in stranger_features: continue
            if self.raw_text_column in field: continue
            new_field = field.replace('.', '_').replace('_result', '').replace('_embeddings_embeddings', '_embeddings')
            logger.info('Renaming Fields for old name=%s and new name=%s', field, new_field)
            if new_field == 'embeddings_embeddings': new_field = 'embeddings'
            if 'metadata' in field:  # rename metadata to something more meaningful
                logger.info('Getting Meta Data for   : nr=%s , name=%s with new_name=%s and original', i, field,
                            new_field)
                new_fields = []
                # we iterate over the keys in the metadata and use them as new column names. The values will become the values in the columns.
                keys_in_metadata = self.extract_keys_in_metadata(ptmp,field)

                # no resulting values for this column, we wont include it in the final output
                if len( keys_in_metadata) == 0: continue
                logger.info('Extracting Meta Keys=%s for field=%s', keys_in_metadata, new_field)
                if meta == True or 'entities' in field:  # get all meta data
                    for key in keys_in_metadata:
                        logger.info('Extracting key=%s', key)
                        # drop sentences keys from Lang detector, they seem irrelevant. same for NER chunk map keys
                        if key == 'sentence' and 'language' in field: continue
                        if key == 'chunk' and 'entities' in field: continue
                        if key == 'sentence' and 'entities' in field: continue
                        if field == 'entities.metadata' : new_fields.append(new_field.replace('metadata','class'))

                        if field =='ner.metadata': new_fields.append(new_field.replace('metadata','confidence'))
                        elif 'entities' not in field: new_fields.append(new_field.replace('metadata', key + '_confidence'))
                        if new_fields[-1] == 'entities_entity': new_fields[-1] = 'ner_tag'
                        ptmp = ptmp.withColumn(new_fields[-1],pyspark_col(('res.' + str(fields_to_rename.index(field)) + '.' + key)))

                        columns_for_select.append(new_fields[-1])

                        logger.info(
                            'Created Meta Data for : nr=%s , original Meta Data key name=%s and new  new_name=%s ', i,
                            key, new_fields[-1])
                else:
                    # Get only meta data with greatest value (highest prob)

                    cols_to_max = []
                    for key in keys_in_metadata:
                        if 'sentence' == key and field == 'sentiment.metadata':continue

                        cols_to_max.append('res.' + str(fields_to_rename.index(field)) + '.' + key)

                    # For Sentiment the sentence.result contains irrelevant Metadata and is not part of the confidence we want. So we remove it here

                    # sadly because the Spark SQL method 'greatest()' does not work properly on scientific notation, we must cast our metadata to decimal with limited precision
                    # scientific notation starts after 6 decimal places, so we can have at most exactly 6
                    # since greatest() breaks the dataframe Schema, we must rename the columns first or run into issues with PySpark Struct queriying

                    for key in cols_to_max: ptmp = ptmp.withColumn(key.replace('.', '_'),pyspark_col(key).cast('decimal(7,6)'))
                    # casted = ptmp.select(*(pyspark_col(c).cast("decimal(6,6)").alias(c.replace('.','_')) for c in cols_to_max))

                    max_confidence_name = field.split('.')[0] + '_confidence'
                    renamed_cols_to_max = [col.replace('.', '_') for col in cols_to_max]

                    if len(cols_to_max) > 1:
                        ptmp = ptmp.withColumn(max_confidence_name, greatest(*renamed_cols_to_max))
                        columns_for_select.append(max_confidence_name)
                    else:
                        ptmp = ptmp.withColumnRenamed(renamed_cols_to_max[0], max_confidence_name)
                        columns_for_select.append(max_confidence_name)
                continue
            # get th e output level results row by row (could be parallelized via mapping for each annotator)
            ptmp = ptmp.withColumn(new_field, ptmp['res.' + str(fields_to_rename.index(field))])
            columns_for_select.append(new_field)
            logger.info('Renaming exploded field  : nr=%s , name=%s to new_name=%s', i, field, new_field)
        return ptmp, columns_for_select

    def extract_keys_in_metadata(self,ptmp:pyspark.sql.DataFrame,field:str) -> [str] :
        '''
        Extract keys in the metadata of the output of a annotator and returns them as a str list
        :param ptmp: Spark dataframe with outputs of an Spark NLP annotator
        :param field: Name of the field for which to find the keys in the metadata. Field should be suffixed with .metadata
        :return: Str list of keys in the metadata for the given field
        '''
        keys_in_metadata = list(ptmp.select(field).take(1))
        try_filter = False
        if len(keys_in_metadata) == 0: try_filter = True # return []
        if len(keys_in_metadata[0].asDict()['metadata']) == 0:  try_filter = True # return []
        if try_filter:
            # Filter for first with list bigger 0 to get metadata
            slen = udf(lambda s: len(s), IntegerType())
            t = ptmp.withColumn('lens', slen(ptmp[field]))
            keys_in_metadata = list(t.filter(t.lens > 0 ).select(field).take(1))
            if len(keys_in_metadata) == 0: return []
            if len(keys_in_metadata[0].asDict()['metadata']) == 0: return []

        keys_in_metadata = list(keys_in_metadata[0].asDict()['metadata'][0].keys())
        if 'sentence' in keys_in_metadata : keys_in_metadata.remove('sentence')
        logger.info(f'Field={field} has keys in metadata={keys_in_metadata}')

        return keys_in_metadata


    def rename_columns_and_extract_map_values_different_level(self, ptmp, fields_to_rename, same_output_level,
                                                              stranger_features=[], meta=True):
        '''
        This method takes in a Spark dataframe that is the result not exploded on, after applying a Spark NLP pipeline to it.
        It will peform the following transformations on the dataframe:
        1. Rename the exploded columns to something more meaningful
        2. Extract Meta data values of columns that contain maps if the data is relevant
        3. Store the new names
        :param ptmp: The dataframe which contains the columns wto be renamed
        :param fields_to_rename: A list of field names that will be renamed in the dataframe.
        :param same_output_level: Wether the fields that are going to be renamed are at the same output level as the pipe or at a different one.
        :param stranger_features:
        :param meta: Wether  to get meta data like prediction confidence or not
        :return: Returns tuple (list, SparkDataFrame), where the first element is a list with all the new names and the second element is a new Spark Dataframe which contains all the renamed and also old columns
        '''

        logger.info(
            f'Renaming columns and extracting meta data for  outputlevel_same={same_output_level} and fields_to_rename={fields_to_rename} and get_meta={meta}')
        columns_for_select = []

        reorderd_fields_to_rename = self.reorder_column_names(fields_to_rename)


        for i, field in enumerate(reorderd_fields_to_rename):
            if self.raw_text_column in field: continue
            new_field = field.replace('.', '_').replace('_result', '').replace('_embeddings_embeddings', '_embeddings')
            if new_field == 'embeddings_embeddings': new_field = 'embeddings'
            logger.info('Renaming Fields for old name=%s and new name=%s', field, new_field)
            if 'metadata' in field:
                # since the have a field with metadata, the values of the original data for which we have metadata for must exist in the dataframe as singular elements inside of a list
                # by applying the expr method, we unpack the elements from the list
                unpack_name = field.split('.')[0]
                ## ONLY for NER or Keywordswe actually expect array type output for different output levels and must do proper casting
                if field == 'entities.metadata':
                    pass  # ner result wil be fatched later
                elif field == 'keywords.metadata':
                    ptmp = ptmp.withColumn(unpack_name + '_result', ptmp[unpack_name + '.result'])
                else:
                    ptmp = ptmp.withColumn(unpack_name + '_result', expr(unpack_name + '.result[0]'))

                reorderd_fields_to_rename[reorderd_fields_to_rename.index(unpack_name + '.result')] = unpack_name + '_result'
                logger.info(f'Getting Meta Data for   : nr={i} , original_name={field} with new_name={new_field} and original')
                # we iterate over the keys in the metadata and use them as new column names. The values will become the values in the columns.
                keys_in_metadata = self.extract_keys_in_metadata(ptmp,field)
                if len(keys_in_metadata) == 0: continue

                if 'sentence' in keys_in_metadata: keys_in_metadata.remove('sentence')
                if 'chunk' in keys_in_metadata and field == 'entities.metadata': keys_in_metadata.remove('chunk')

                new_fields = []
                logger.info('Extracting Meta Keys=%s for field=%s', keys_in_metadata, new_field)

                for key in keys_in_metadata:
                    # we cant skip getting  key values for everything, even if meta=false. This is because we need to get the greatest of all confidence values , for this we must unpack them first..
                    if key =='word' and field =='ner.metadata' : continue # irrelevant metadata in the for the word key
                    if field == 'entities.metadata' or field == 'sentiment.metadata' or field =='ner.metadata': new_fields.append(new_field.replace('metadata','confidence'))
                    else : new_fields.append(new_field.replace('metadata', key + '_confidence'))
                    # entities_entity
                    if new_fields[-1] == 'entities_entity': new_fields[-1] = 'ner_tag'
                    logger.info(f'Extracting meta data for key={key} and column name={new_fields[-1]}')

                    # These Pyspark UDF extracts from a list of maps all the map values for positive and negative confidence and also spell costs
                    def extract_map_values_float(x): return [float(sentence[key]) for sentence in x]
                    def extract_map_values_str(x): return [str(sentence[key]) for sentence in x]

                    # extract map values for list of maps
                    # Since ner is only component  wit string metadata, we have this simple conditional
                    if field == 'entities.metadata':
                        array_map_values = udf(lambda z: extract_map_values_str(z), ArrayType(StringType()))
                        ptmp.withColumn(new_fields[-1], array_map_values(field)).select(expr(f'{new_fields[-1]}[0]'))
                        ptmp = ptmp.withColumn(new_fields[-1], array_map_values(field))
                    elif field == 'ner.metadata' and key =='confidence' :
                        array_map_values = udf(lambda z: extract_map_values_float(z), ArrayType(FloatType()))
                        ptmp.withColumn(new_fields[-1], array_map_values(field)).select(f'{new_fields[-1]}')
                        ptmp = ptmp.withColumn(new_fields[-1], array_map_values(field))
                        # ptmp = ptmp.withColumn(new_fields[-1], expr(new_fields[-1] + '[0]'))
                    else :
                        # EXPERIMENTAL extraction, should work for all FloatTypes?
                        # We apply Expr here because all result ing meta data is inside of a list and just a single element, which we can take out
                        # Exceptions to this rule are entities and metadata, this are scenarios wehre we want all elements from the predictions array ( since it could be multiple keywords/entities)
                        array_map_values = udf(lambda z: extract_map_values_float(z), ArrayType(FloatType()))
                        ptmp.withColumn(new_fields[-1], array_map_values(field)).select(expr(f'{new_fields[-1]}[0]'))
                        ptmp = ptmp.withColumn(new_fields[-1], array_map_values(field))
                        ptmp = ptmp.withColumn(new_fields[-1], expr(new_fields[-1] + '[0]'))
                    logger.info(f'Created Meta Data for   : nr={i} , original_name={field} with new_name={new_fields[-1]}')
                    columns_for_select.append(new_fields[-1])

                if meta == True:
                    continue  # If we dont max, we will see the confidence for all other classes. by continuing here, we will leave all the confidences for the other classes in the DF.
                else:
                    # If meta == false we need to find the meta data col umn with the HIGHEST confidence and only keep that!
                    # Assuming we have only 1 confidence value per Column. If here are Multiple then...(?)
                    # We gotta get the max confidence column, remove all other cols for selection
                    if field == 'entities.metadata': continue
                    if field == 'ner.metadata': continue
                    if field == 'keywords.metadata': continue  # We dont want to max for multiple keywords. Also it will change the name from score to confidence of the final column
                    # if field in multi_level_fields : continue # multi_classifier_dl, YAKE
                    # if field ==
                    cols_to_max = []
                    prefix = field.split('.')[0]
                    for key in keys_in_metadata: cols_to_max.append(prefix + '_' + key +'_confidence')

                    # cast all the types to decimal, remove scientific notation
                    for key in new_fields: ptmp = ptmp.withColumn(key, pyspark_col(key).cast('decimal(7,6)'))

                    max_confidence_name = field.split('.')[0] + '_confidence'
                    if len(cols_to_max) > 1:
                        ptmp = ptmp.withColumn(max_confidence_name, greatest(*cols_to_max))
                        columns_for_select.append(max_confidence_name)
                    else:
                        ptmp = ptmp.withColumnRenamed(cols_to_max[0], max_confidence_name)
                        columns_for_select.append(max_confidence_name)

                    for f in new_fields:
                        # we remove the new fields becasue they duplicate the infomration of max confidence field
                        if f in columns_for_select: columns_for_select.remove(f)

                continue  # end of special meta data case

            if field == 'entities_result':
                ptmp = ptmp.withColumn('entities_result', ptmp['entities.result'].cast(ArrayType(StringType())))  #



            ptmp = ptmp.withColumn(new_field, ptmp[field])  # get the outputlevel results row by row
            # ptmp = ptmp.withColumnRenamed(field,new_field)  # EXPERIMENTAL engine test, only works sometimes since it can break dataframe struct
            logger.info(f'Renaming non exploded field  : nr={i} , original_name={field} to new_name={new_field}')
            columns_for_select.append(new_field)
        return ptmp, columns_for_select

    def extract_multi_level_outputs(self,ptmp:pyspark.sql.DataFrame, multi_level_col_names:List[str],meta:bool) -> (pyspark.sql.DataFrame,List[str]):
        '''
        Extract the columns for toPandas conversion from a Pyspark dataframe. Applicable to outputs of MultiClassifierDL/Yake or other MultiLevel Output level NLU components
        for field.result we can just extract the raw column and rename it to smth nice
        for field.metadata  there are 2 cases
        1. if metadata==true then we want one column per key in metadata. Each column has the corrosponding confidence. In adition, we have a result column for field with conf<0.5 (except key==sentence)
        2. if metadata==false then we want just one column with the confidences which are above 0.5
        :param ptmp: spark dataframe with the output columns of a Multi-Outputlevel Annotator
        :param multi_level_col_names: The columns which are outputs of the multi_level component
        :param meta: Wether to return all additional metadata or not ( i.e. return probabilities for all classes, even if their probability is below classification threshold which is usually 0.5.
        :return: Spark Dataframe with new columns ready for toPandas conversion and also a list with all column names which should be used for Pandas conversion.
        '''
        columns_for_select = []
        logger.info(f"Extracting multi level fields={multi_level_col_names}")

        def extract_classnames_and_confidences(x,keys_in_metadata, threshold=0.5):
            ## UDF for extracting confidences and their class names as struct types if the confidence is larger than threshold
            confidences = []
            classes = []
            if not isinstance(x,dict) : return [[],[]]#[[0.0],['No Classes Detected']]
            for key in keys_in_metadata :
                if key =='sentence' : continue     # irrelevant metadata
                if float(x[key]) >= threshold :
                    confidences.append(float(x[key]))
                    classes.append(key)
            return [confidences, classes]
        schema = StructType([
            StructField("confidences", ArrayType(DoubleType()), False),
            StructField("classes", ArrayType(StringType()), False)
        ])
        def extract_map_values_float(x,key): return [float(sentence[key]) for sentence in x]

            # we dont care about .result col, we get all from metadarta
        for field in multi_level_col_names:
            base_field_name = field.split('.')[0]
            confidence_field_name = base_field_name+'_confidences'
            class_field_name = base_field_name+'_classes'
            if 'metadata' in field :

                keys_in_metadata = self.extract_keys_in_metadata(ptmp,field)
                if len(keys_in_metadata) == 0: continue
                if not meta:
                    if 'keyword' in field: # yake handling
                        array_map_values = udf(lambda z: extract_map_values_float(z,'score'), ArrayType(FloatType()))
                        ptmp = ptmp.withColumn(confidence_field_name, array_map_values(field))
                        columns_for_select += [confidence_field_name]
                    else :
                        # create a confidence and class column which both contain a list of predicted classes/confidences
                        # we apply the  UDF only to the first element because metadata is duplicated for multi classifier dl and all relevent info is in the first element of the metadata col list
                        #todo extract thresold
                        extract_classnames_and_confidences_udf = udf(lambda z: extract_classnames_and_confidences(z, keys_in_metadata, threshold=0.5), schema)
                        ptmp = ptmp.withColumn('multi_level_extract_result', extract_classnames_and_confidences_udf(expr(f'{field}[0]')))
                        ptmp = ptmp.withColumn(confidence_field_name , ptmp['multi_level_extract_result.confidences'])
                        ptmp = ptmp.withColumn(class_field_name, ptmp['multi_level_extract_result.classes'])
                        columns_for_select += [confidence_field_name, class_field_name]

                else :
                    confidence_field_names = []
                    for key in keys_in_metadata :
                        #create one col per confidence and only get those
                        if key =='sentence':continue
                        new_confidence_field_name = base_field_name + '_' +key +'_confidence'
                        if 'keyword' in field: # yake handling
                            array_map_values = udf(lambda z: extract_map_values_float(z,'score'), ArrayType(FloatType()))
                            ptmp = ptmp.withColumn(new_confidence_field_name, array_map_values(field))
                            confidence_field_names.append(new_confidence_field_name)

                        else:
                            ptmp = ptmp.withColumn(new_confidence_field_name, expr(f'{field}[0]["{key}"]'))
                            confidence_field_names.append(new_confidence_field_name)


                    columns_for_select += confidence_field_names

            else :
                base_field_name = field.split('.')[0]
                class_field_name = base_field_name+'_classes'
                ptmp = ptmp.withColumn(class_field_name,ptmp[field])
                columns_for_select += [class_field_name]

        return ptmp, columns_for_select

    def resolve_input_dependent_component_to_output_level(self, component):
        '''
        For a given NLU component  which is input dependent , resolve its output level by checking if it's input stem from document or sentence based annotators
        :param component:  to resolve
        :return: resolve component
        '''
        # (1.) A classifier, which is using sentence/document. We just check input cols

        if 'document' in component.component_info.spark_input_column_names :  return 'document'
        if 'sentence' in component.component_info.spark_input_column_names :  return 'sentence'

        # (2.) A classifier, which is using sentence/doc embeddings.
        # We iterate over the pipe and check which Embed component is feeding the classifier and what the input that embed annotator is (sent or doc)
        for c in self.pipe_components:
            # check if c is of sentence embedding class  which is always input dependent
            if any ( isinstance(c.model, e ) for e in self.all_embeddings['input_dependent']  ) :
                if 'document' in c.component_info.spark_input_column_names :  return 'document'
                if 'sentence' in c.component_info.spark_input_column_names :  return 'sentence'


    def resolve_component_to_output_level(self,component):
        '''
        For a given NLU component, resolve its output level, by checking annotator_levels dicts for approaches and models
        If output level is input dependent, resolve_input_dependent_component_to_output_level will resolve it
        :param component:  to resolve
        :return: resolve component
        '''
        for level in self.annotator_levels_model_based.keys():
            for t in self.annotator_levels_model_based[level]:
                if isinstance(component.model,t) :
                    if level == 'input_dependent' : return self.resolve_input_dependent_component_to_output_level(component)
                    else : return level

        for level in self.annotator_levels_approach_based.keys():
            for t in self.annotator_levels_approach_based[level]:
                if isinstance(component.model,t) :
                    if level == 'input_dependent' : return self.resolve_input_dependent_component_to_output_level(component)
                    else : return level



    def infer_and_set_output_level(self):
        '''
        This function checks the LAST  component of the NLU pipeline and infers
        and infers from that the output level via checking the components info.
        It sets the output level of the pipe accordingly
        param sdf : Spark dataframe after transformations
        '''
        # Loop in reverse over pipe and get first non util/sentence_detecotr/tokenizer/doc_assember. If there is non, take last
        bad_types = [ 'util','document','sentence']
        bad_names = ['token']

        for c in self.pipe_components[::-1]:
            if any (t in  c.component_info.type for t in bad_types) : continue
            if any (n in  c.component_info.name for n in bad_names) : continue
            self.output_level = self.resolve_component_to_output_level(c)
            logger.info('Inferred and set output level of pipeline to %s', self.output_level)
            break
        if self.output_level == None  or self.output_level == '': self.output_level = 'document' # Voodo Normalizer bug that does not happen in debugger bugfix
        logger.info('Inferred and set output level of pipeline to %s', self.output_level)

    def get_chunk_col_name(self):
        '''
        This methdo checks wether there is a chunk component in the pipelien.
        If there is, it will return the name of the output columns for that component
        :return: Name of the chunk type column in the dataset
        '''

        for component in self.pipe_components:
            if component.component_info.output_level == 'chunk':
                # Usually al chunk components ahve only one output and that is the cunk col so we can safely just pass the first element of the output list to the caller
                logger.info("Detected %s as chunk output column for later zipping", component.component_info.name)
                return component.component_info.spark_output_column_names[0]

    def resolve_field_to_output_level(self, field,f_type):
        '''
        For a given field from resulting datafarme, search find the component that generated that field and returns it's output level
        :param field: The field to find the output_level for
        :param f_type: The type of the field to fint the output level for
        :return: The output level of the field
        '''
        target = field.split('.')[0]
        for c in self.pipe_components:
            if target in c.component_info.spark_output_column_names:
                # MultiClassifier outputs should never be at same output level as pipe, returning special_case takes care of this
                if isinstance(c.model, (MultiClassifierDLModel, MultiClassifierDLApproach,YakeModel)): return "multi_level"
                return self.resolve_component_to_output_level(c)



    def select_features_from_result(self, field_dict, processed, stranger_features, same_output_level_fields,
                                    not_at_same_output_level_fields):
        '''
        loop over all fields and decide which ones to keep and also if there are at the current output level or at a different one .
        We sort the fields by output level. If they are at the same as the current output level and will be exploded
        Or they are at a different level, in which case they will be left as is and not zipped before explode

        :param field_dict: field dict generated by get_field_types_dict
        :param processed:  Spark dataframe generated by transformer pipeline
        :param stranger_features: Features in the dataframe, which where not generated by NLU
        :param at_same_output_level_fields:  Features which are deemed at the same output level of the pipeline
        :param not_at_same_output_level_fields: Features which are not deemed at the same output level of the pipeline
        :return: Tuple (at_same_output_level_fields, not_at_same_output_level_fields)
        '''
        multi_level_fields=[]
        for field in processed.schema.fieldNames():
            if field in stranger_features: continue
            if field == self.raw_text_column: continue
            if field == self.output_level: continue
            # if 'label' in field and 'dependency' not in field: continue  # specal case for input labels

            f_type = field_dict[field]
            logger.info('Selecting Columns for field=%s of type=%s', field, f_type)
            inferred_output_level = self.resolve_field_to_output_level( field,f_type)

            if inferred_output_level == 'multi_level' :
                if self.output_positions:
                    multi_level_fields.append(field + '.begin')
                    multi_level_fields.append(field + '.end')
                multi_level_fields.append(field + '.metadata')
                multi_level_fields.append(field + '.result')

            elif inferred_output_level == self.output_level:
                logger.info(f'Setting field for field={field} of type={f_type} to output level={inferred_output_level} which is SAME LEVEL')
                if 'embeddings' not in field and 'embeddings' not in f_type: same_output_level_fields.append(
                    field + '.result')  # result of embeddigns is just the word/sentence
                if self.output_positions:
                    same_output_level_fields.append(field + '.begin')
                    same_output_level_fields.append(field + '.end')
                if 'embeddings' in f_type:
                    same_output_level_fields.append(field + '.embeddings')
                if 'entities' in field:
                    same_output_level_fields.append(field + '.metadata')
                if 'ner' in field:
                    same_output_level_fields.append(field + '.metadata')
                if 'category' in f_type or 'spell' in f_type or 'sentiment' in f_type or 'class' in f_type or 'language' in f_type or 'keyword' in f_type:
                    same_output_level_fields.append(field + '.metadata')
            else:
                logger.info(f'Setting field for field={field} of type={f_type} to output level={inferred_output_level} which is NOT SAME LEVEL')

                if 'embeddings' not in field and 'embeddings' not in f_type: not_at_same_output_level_fields.append(
                    field + '.result')  # result of embeddigns is just the word/sentence
                if self.output_positions:
                    not_at_same_output_level_fields.append(field + '.begin')
                    not_at_same_output_level_fields.append(field + '.end')
                if 'embeddings' in f_type:
                    not_at_same_output_level_fields.append(field + '.embeddings')
                if 'category' in f_type or 'spell' in f_type or 'sentiment' in f_type or 'class' in f_type or 'keyword' in f_type:
                    not_at_same_output_level_fields.append(field + '.metadata')
                if 'entities' in field:
                    not_at_same_output_level_fields.append(field + '.metadata')
                if 'ner' in field:
                    not_at_same_output_level_fields.append(field + '.metadata')
        if self.output_level == 'document':
            # explode stranger features if output level is document
            # same_output_level_fields =list(set( same_output_level_fields + stranger_features))
            same_output_level_fields = list(set(same_output_level_fields))
            # same_output_level_fields.remove('origin_index')
        return same_output_level_fields, not_at_same_output_level_fields,multi_level_fields


    def pythonify_spark_dataframe(self, processed, get_different_level_output=True, keep_stranger_features=True,
                                  stranger_features=[], drop_irrelevant_cols=True, output_metadata=False,
                                  index_provided=False):
        '''
        This functions takes in a spark dataframe with Spark NLP annotations in it and transforms it into a Pandas Dataframe with common feature types for further NLP/NLU downstream tasks.
        It will recylce Indexes from Pandas DataFrames and Series if they exist, otherwise a custom id column will be created which is used as inex later on
            It does this by performing the following consecutive steps :
                1. Select columns to explode
                2. Select columns to keep
                3. Rename columns
                4. Create Pandas Dataframe object


        :param processed: Spark dataframe which an NLU pipeline has transformed
        :param output_level: The output level at which returned pandas Dataframe should be
        :param get_different_level_output:  Wheter to get features from different levels
        :param keep_stranger_features : Wether to keep additional features from the input DF when generating the output DF or if they should be discarded for the final output DF
        :param stranger_features: A list of features which are not known to NLU and inside of the input DF.
                                    Basically all columns, which are not named 'text' in the input.
                                    If keep_stranger_features== True, then these features will be exploded, if output_level == DOCUMENt, otherwise they will not be exploded
        :param output_metadata: Wether to keep or drop additional metadataf or predictions, like prediction confidence
        :return: Pandas dataframe which easy accessable features
        '''

        stranger_features += ['origin_index']

        if self.output_level == '': self.infer_and_set_output_level()

        field_dict = self.get_field_types_dict(processed, stranger_features,keep_stranger_features)  # map field to type of field
        not_at_same_output_level_fields = []

        if self.output_level == 'chunk':
            # if output level is chunk, we must check if we actually have a chunk column in the pipe. So we search it
            chunk_col = self.get_chunk_col_name()
            same_output_level_fields = [chunk_col + '.result']
        else:
            same_output_level_fields = [self.output_level + '.result']

        logger.info('Setting Output level as : %s', self.output_level)

        if keep_stranger_features:
            sdf = processed.select(['*'])
        else:
            features_to_keep = list(set(processed.columns) - set(stranger_features))
            sdf = processed.select(features_to_keep)

        if index_provided == False:
            logger.info("Generating origin Index via Spark. May contain irregular distributed index values.")
            sdf = sdf.withColumn(monotonically_increasing_id().alias('origin_index'))

        same_output_level_fields, not_at_same_output_level_fields,multi_level_fields = self.select_features_from_result(field_dict,
                                                                                                     processed,
                                                                                                     stranger_features,
                                                                                                     same_output_level_fields,
                                                                                                     not_at_same_output_level_fields)



        logger.info(f'exploding amd zipping at same level fields = {same_output_level_fields}')
        logger.info(f'as same level fields = {not_at_same_output_level_fields}')
        def zip_col_py(*cols): return list(zip(*cols))

        output_fields = sdf[same_output_level_fields].schema.fields
        d_types = []
        for i,o in enumerate(output_fields) : d_types.append(StructField(name=str(i),dataType= o.dataType.elementType) )
        udf_type = t.ArrayType(t.StructType(d_types))
        arrays_zip_ = F.udf(zip_col_py,udf_type)


        ptmp = sdf.withColumn('tmp', arrays_zip_(*same_output_level_fields)) \
            .withColumn("res", explode('tmp'))




        final_select_not_at_same_output_level = []



        ptmp, final_select_same_output_level = self.rename_columns_and_extract_map_values_same_level(ptmp=ptmp,
                                                                                                     fields_to_rename=same_output_level_fields,
                                                                                                     same_output_level=True,
                                                                                                     stranger_features=stranger_features,
                                                                                                     meta=output_metadata)
        if get_different_level_output:
            ptmp, final_select_not_at_same_output_level = self.rename_columns_and_extract_map_values_different_level(
                ptmp=ptmp, fields_to_rename=not_at_same_output_level_fields, same_output_level=False,
                meta=output_metadata, )

        ptmp,final_select_multi_output_level = self.extract_multi_level_outputs(ptmp, multi_level_fields, output_metadata)
        if keep_stranger_features: final_select_not_at_same_output_level += stranger_features




        logger.info('Final cleanup select of same level =%s', final_select_same_output_level)
        logger.info('Final cleanup select of different level =%s', final_select_not_at_same_output_level)
        logger.info('Final cleanup select of multi level =%s', final_select_multi_output_level)

        logger.info('Final ptmp columns = %s', ptmp.columns)

        final_cols = final_select_same_output_level + final_select_not_at_same_output_level + final_select_multi_output_level + ['origin_index']
        if drop_irrelevant_cols: final_cols = self.drop_irrelevant_cols(final_cols)
        # ner columns is NER-IOB format, mostly useless for the users. If meta false, we drop it here.
        if output_metadata == False and 'ner' in final_cols: final_cols.remove('ner')
        final_df = ptmp.select(list(set(final_cols)))

        pandas_df = self.finalize_return_datatype(final_df)
        if isinstance(pandas_df,pyspark.sql.dataframe.DataFrame):
            return pandas_df # is actually spark df
        else:
            pandas_df.set_index('origin_index', inplace=True)
            return self.convert_embeddings_to_np(pandas_df)


    def convert_embeddings_to_np(self, pdf):
        '''
        convert all the columns in a pandas df to numpy
        :param pdf: Pandas Dataframe whose embedding column will be converted to numpy array objects
        :return:
        '''

        for col in pdf.columns:
            if 'embed' in col:
                pdf[col] = pdf[col].apply(lambda x: np.array(x))
        return pdf


    def finalize_return_datatype(self, sdf):
        '''
        Take in a Spark dataframe with only relevant columns remaining.
        Depending on what value is set in self.output_datatype, this method will cast the final SDF into Pandas/Spark/Numpy/Modin/List objects
        :param sdf:
        :return: The predicted Data as datatype dependign on self.output_datatype
        '''

        if self.output_datatype == 'spark':
            return sdf
        elif self.output_datatype == 'pandas':
            return sdf.toPandas()
        elif self.output_datatype == 'modin':
            import modin.pandas as mpd
            return mpd.DataFrame(sdf.toPandas())
        elif self.output_datatype == 'pandas_series':
            return sdf.toPandas()
        elif self.output_datatype == 'modin_series':
            import modin.pandas as mpd
            return mpd.DataFrame(sdf.toPandas())
        elif self.output_datatype == 'numpy':
            return sdf.toPandas().to_numpy()
        elif self.output_datatype == 'string':
            return sdf.toPandas()
        elif self.output_datatype == 'string_list':
            return sdf.toPandas()
        elif self.output_datatype == 'array':
            return sdf.toPandas()


    def drop_irrelevant_cols(self, cols):
        '''
        Takes in a list of column names removes the elements which are irrelevant to the current output level.
        This will be run before returning the final df
        Drop column candidates are document, sentence, token, chunk.
        columns which are NOT AT THE SAME output level will be dropped
        :param cols:  list of column names in the df
        :return: list of columns with the irrelevant names removed
        '''
        if self.output_level == 'token':
            if 'document' in cols: cols.remove('document')
            if 'chunk' in cols: cols.remove('chunk')
            if 'sentence' in cols: cols.remove('sentence')
        if self.output_level == 'sentence':
            if 'token' in cols: cols.remove('token')
            if 'chunk' in cols: cols.remove('chunk')
            if 'document' in cols: cols.remove('document')
        if self.output_level == 'chunk':
            if 'document' in cols: cols.remove('document')
            if 'token' in cols: cols.remove('token')
            if 'sentence' in cols: cols.remove('sentence')
        if self.output_level == 'document':
            if 'token' in cols: cols.remove('token')
            if 'chunk' in cols: cols.remove('chunk')
            if 'sentence' in cols: cols.remove('sentence')

        return cols


    def configure_light_pipe_usage(self, data_instances, use_multi=True):
        logger.info("Configuring Light Pipeline Usage")
        if data_instances > 50000 or use_multi == False:
            logger.info("Disabling light pipeline")
            self.fit()
            return
        else:
            if self.light_pipe_configured == False:
                self.light_pipe_configured = True
                logger.info("Enabling light pipeline")
                self.spark_transformer_pipe = LightPipeline(self.spark_transformer_pipe)

    def check_if_sentence_level_requirements_met(self):
        '''
        Check if the pipeline currently has an annotator that generate sentence col as output. If not, return False
        :return:
        '''

        for c in self.pipe_components:
            if 'sentence' in c.component_info.spark_output_column_names : return True
        return False

    def add_missing_sentence_component(self):
        '''
        Add Sentence Detector to pipeline and Run it thorugh the Query Verifiyer again.
        :return: None
        '''

    def write_nlu_pipe_info(self,path):
        '''
        Writes all information required to load a NLU pipeline from disk to path
        :param path: path where to store the nlu_info.json
        :return: True if success, False if failure
        '''
        import os
        f = open(os.path.join(path,'nlu_info.txt'), "w")
        f.write(self.nlu_reference)
        f.close()
        #1. Write all primitive pipe attributes to dict
        # pipe_data = {
        #     'has_trainable_components': self.has_trainable_components,
        #     'is_fitted' : self.is_fitted,
        #     'light_pipe_configured' : self.light_pipe_configured,
        #     'needs_fitting':self.needs_fitting,
        #     'nlu_reference':self.nlu_reference,
        #     'output_datatype':self.output_datatype,
        #     'output_different_levels':self.output_different_levels,
        #     'output_level': self.output_level,
        #     'output_positions': self.output_positions,
        #     'pipe_componments': {},
        #     'pipe_ready':self.pipe_ready,
        #     'provider': self.provider,
        #     'raw_text_column': self.raw_text_column,
        #     'raw_text_matrix_slice': self.raw_text_matrix_slice,
        #     'spark_nlp_pipe': self.spark_nlp_pipe,
        #     'spark_non_light_transformer_pipe': self.spark_non_light_transformer_pipe,
        #     'component_count': len(self)
        #
        # }

        #2. Write all component/component_info to dict
        # for c in self.pipe_components:
        #     pipe_data['pipe_componments'][c.ma,e]
        #3. Any additional stuff

        return True

    def add_missing_component_if_missing_for_output_level(self):
        '''
        Check that for currently configured self.output_level one annotator for that level exists, i.e a Sentence Detetor for outpul tevel sentence, Tokenizer for level token etc..

        :return: None
        '''

        if self.output_level =='sentence':
            if self.check_if_sentence_level_requirements_met(): return
            else :
                logger.info('Adding missing sentence Dependency because it is missing for outputlevel=Sentence')
                self.add_missing_sentence_component()
    def save(self, path, component='entire_pipeline', overwrite=False):

        if nlu.is_running_in_databricks() :
            if path.startswith('/dbfs/') or path.startswith('dbfs/'):
                nlu_path = path
                if path.startswith('/dbfs/'):
                    nlp_path =  path.replace('/dbfs','')
                else :
                    nlp_path =  path.replace('dbfs','')

            else :
                nlu_path = 'dbfs/' + path
                if path.startswith('/') : nlp_path = path
                else : nlp_path = '/' + path

            if not self.is_fitted and self.has_trainable_components:
                self.fit()
                self.is_fitted = True
            if component == 'entire_pipeline':
                self.spark_transformer_pipe.save(nlp_path)
                self.write_nlu_pipe_info(nlu_path)


        if overwrite and not nlu.is_running_in_databricks():
            import shutil
            shutil.rmtree(path,ignore_errors=True)


        if not self.is_fitted :
            self.fit()
            self.is_fitted = True
        if component == 'entire_pipeline':
            self.spark_transformer_pipe.save(path)
            self.write_nlu_pipe_info(path)
        else:
            if component in self.keys():
                self[component].save(path)
            # else :
            #     print(f"Error during saving,{component} does not exist in the pipeline.\nPlease use pipe.print_info() to see the references you need to pass save()")

        print(f'Stored model in {path}')
        # else : print('Please fit untrained pipeline first or predict on a String to save it')
    def predict(self, data, output_level='', positions=False, keep_stranger_features=True, metadata=False,
                multithread=True, drop_irrelevant_cols=True, verbose=False):
        '''
        Annotates a Pandas Dataframe/Pandas Series/Numpy Array/Spark DataFrame/Python List strings /Python String

        :param data: Data to predict on
        :param output_level: output level, either document/sentence/chunk/token
        :param positions: wether to output indexes that map predictions back to position in origin string
        :param keep_stranger_features: wether to keep columns in the dataframe that are not generated by pandas. I.e. when you s a dataframe with 10 columns and only one of them is named text, the returned dataframe will only contain the text column when set to false
        :param metadata: wether to keep additonal metadata in final df or not like confidiences of every possible class for preidctions.
        :param multithread: Whether to use multithreading based lightpipeline. In some cases, this may cause errors.
        :param drop_irellevant_cols: Wether to drop cols of different output levels, i.e. when predicting token level and dro_irrelevant_cols = True then chunk, sentence and Doc will be dropped
        :return:
        '''

        if output_level != '': self.output_level = output_level

        self.output_positions = positions

        if output_level == 'chunk':
            # If no chunk output component in pipe we must add it and run the query PipelineQueryVerifier again
            chunk_provided = False
            for component in self.pipe_components:
                if component.component_info.output_level == 'chunk': chunk_provided = True
            if chunk_provided == False:
                self.pipe_components.append(nlu.get_default_component_of_type('chunk'))
                # this could break indexing..

                self = nlu.pipeline_logic.PipelineQueryVerifier.check_and_fix_nlu_pipeline(self)
        # if not self.is_fitted: self.fit()

        # currently have to always fit, otherwise parameter changes wont take effect
        if output_level == 'sentence' or output_level == 'document':
            self = nlu.pipeline_logic.PipelineQueryVerifier.configure_component_output_levels(self)
            self = nlu.pipeline_logic.PipelineQueryVerifier.check_and_fix_nlu_pipeline(self)


        if not self.is_fitted :
            if self.has_trainable_components :
                self.fit(data)
            else : self.fit()
        # self.configure_light_pipe_usage(len(data), multithread)

        sdf = None
        stranger_features = []
        index_provided = False
        infered_text_col = False

        try:
            if isinstance(data,pyspark.sql.dataframe.DataFrame):  # casting follows spark->pd
                self.output_datatype = 'spark'
                data = data.withColumn('origin_index',monotonically_increasing_id().alias('origin_index'))
                index_provided = True

                if self.raw_text_column in data.columns:
                    # store all stranger features
                    if len(data.columns) > 1:
                        stranger_features = list(set(data.columns) - set(self.raw_text_column))
                    sdf = self.spark_transformer_pipe.transform(data)
                else:
                    print(
                        'Could not find column named "text" in input Pandas Dataframe. Please ensure one column named such exists. Columns in DF are : ',
                        data.columns)
            elif isinstance(data,pd.DataFrame):  # casting follows pd->spark->pd
                self.output_datatype = 'pandas'

                # set first col as text column if there is none
                if self.raw_text_column not in data.columns:
                    data.rename(columns={data.columns[0]: 'text'}, inplace=True)
                data['origin_index'] = data.index
                index_provided = True
                if self.raw_text_column in data.columns:
                    if len(data.columns) > 1:
                        data = data.where(pd.notnull(data), None)  # make  Nans to None, or spark will crash
                        data = data.dropna(axis=1, how='all')
                        stranger_features = list(set(data.columns) - set(self.raw_text_column))
                    sdf = self.spark_transformer_pipe.transform(self.spark.createDataFrame(data))

                else:
                    logger.info(
                        'Could not find column named "text" in input Pandas Dataframe. Please ensure one column named such exists. Columns in DF are : ',
                        data.columns)
            elif isinstance(data,pd.Series):  # for df['text'] colum/series passing casting follows pseries->pdf->spark->pd
                self.output_datatype = 'pandas_series'
                data = pd.DataFrame(data).dropna(axis=1, how='all')
                index_provided = True
                # If series from a column is passed, its column name will be reused.
                if self.raw_text_column not in data.columns and len(data.columns) == 1:
                    data['text'] = data[data.columns[0]]
                else:
                    logger.info('INFO: NLU will assume', data.columns[0],
                          'as label column since default text column could not be find')
                    data['text'] = data[data.columns[0]]

                data['origin_index'] = data.index

                if self.raw_text_column in data.columns:
                    sdf = self.spark_transformer_pipe.transform(self.spark.createDataFrame(data), )

                else:
                    print(
                        'Could not find column named "text" in  Pandas Dataframe generated from input  Pandas Series. Please ensure one column named such exists. Columns in DF are : ',
                        data.columns)

            elif isinstance(data,np.ndarray):
                # This is a bit inefficient. Casting follow  np->pd->spark->pd. We could cut out the first pd step
                self.output_datatype = 'numpy_array'
                if len(data.shape) != 1:
                    print("Exception : Input numpy array must be 1 Dimensional for prediction.. Input data shape is",
                          data.shape)
                    return nlu.NluError
                sdf = self.spark_transformer_pipe.transform(self.spark.createDataFrame(
                    pd.DataFrame({self.raw_text_column: data, 'origin_index': list(range(len(data)))})))
                index_provided = True

            elif isinstance(data,np.matrix):  # assumes default axis for raw texts
                print(
                    'Predicting on np matrices currently not supported. Please input either a Pandas Dataframe with a string column named "text"  or a String or a list of strings. ')
                return nlu.NluError
            elif isinstance(data,str):  # inefficient, str->pd->spark->pd , we can could first pd
                self.output_datatype = 'string'
                sdf = self.spark_transformer_pipe.transform(self.spark.createDataFrame(
                    pd.DataFrame({self.raw_text_column: data, 'origin_index': [0]}, index=[0])))
                index_provided = True

            elif isinstance(data,list):  # inefficient, list->pd->spark->pd , we can could first pd
                self.output_datatype = 'string_list'
                if all(type(elem) == str for elem in data):
                    sdf = self.spark_transformer_pipe.transform(self.spark.createDataFrame(pd.DataFrame(
                        {self.raw_text_column: pd.Series(data), 'origin_index': list(range(len(data)))})))
                    index_provided = True

                else:
                    print("Exception: Not all elements in input list are of type string.")
            elif isinstance(data,dict):  # Assumes values should be predicted
                print(
                    'Predicting on dictionaries currently not supported. Please input either a Pandas Dataframe with a string column named "text"  or a String or a list of strings. ')
                return ''
            else:  # Modin tests, This could crash if Modin not installed
                try:
                    import modin.pandas as mpd
                    if isinstance(data, mpd.DataFrame):
                        data = pd.DataFrame(data.to_dict())  # create pandas to support type inference
                        self.output_datatype = 'modin'
                        data['origin_index'] = data.index
                        index_provided = True

                    if self.raw_text_column in data.columns:
                        if len(data.columns) > 1:
                            data = data.where(pd.notnull(data), None)  # make  Nans to None, or spark will crash
                            data = data.dropna(axis=1, how='all')
                            stranger_features = list(set(data.columns) - set(self.raw_text_column))
                        sdf = self.spark_transformer_pipe.transform(
                            # self.spark.createDataFrame(data[['text']]), ) # this takes text column as series and makes it DF
                            self.spark.createDataFrame(data))
                    else:
                        print(
                            'Could not find column named "text" in input Pandas Dataframe. Please ensure one column named such exists. Columns in DF are : ',
                            data.columns)

                    if isinstance(data, mpd.Series):
                        self.output_datatype = 'modin_series'
                        data = pd.Series(data.to_dict())  # create pandas to support type inference
                        data = pd.DataFrame(data).dropna(axis=1, how='all')
                        data['origin_index'] = data.index
                        index_provided = True
                        if self.raw_text_column in data.columns:
                            sdf = \
                                self.spark_transformer_pipe.transform(
                                    self.spark.createDataFrame(data[['text']]), )
                        else:
                            print(
                                'Could not find column named "text" in  Pandas Dataframe generated from input  Pandas Series. Please ensure one column named such exists. Columns in DF are : ',
                                data.columns)


                except:
                    print(
                        "If you use Modin, make sure you have installed 'pip install modin[ray]' or 'pip install modin[dask]' backend for Modin ")

            return self.pythonify_spark_dataframe(sdf, self.output_different_levels,
                                                  keep_stranger_features=keep_stranger_features,
                                                  stranger_features=stranger_features, output_metadata=metadata,
                                                  index_provided=index_provided,
                                                  drop_irrelevant_cols=drop_irrelevant_cols
                                                  )
        except Exception as err :
            import sys
            if multithread == True:
                logger.warning("Multithreaded mode failed. trying to predict again with non multithreaded mode ")
                return self.predict(data, output_level=output_level, positions=positions,
                                    keep_stranger_features=keep_stranger_features, metadata=metadata, multithread=False)
            logger.exception('Exception occured')
            e = sys.exc_info()
            print("No accepted Data type or usable columns found or applying the NLU models failed. ")
            print(
                "Make sure that the first column you pass to .predict() is the one that nlu should predict on OR rename the column you want to predict on to 'text'  ")
            print(
                "If you are on Google Collab, click on Run time and try factory reset Runtime run the setup script again, you might have used too much memory")
            print(
                "On Kaggle try to reset restart session and run the setup script again, you might have used too much memory")

            print('Full Stacktrace was', e)
            print('Additional info:')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            import os
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(
                'Stuck? Contact us on Slack! https://join.slack.com/t/spark-nlp/shared_invite/zt-j5ttxh0z-Fn3lQSG1Z0KpOs_SRxjdyw0196BQCDPY')
            if verbose :
                err = sys.exc_info()[1]
                print(str(err))
            return None


    def print_info(self, ):
        '''
        Print out information about every component currently loaded in the pipe and their configurable parameters
        :return: None
        '''

        print('The following parameters are configurable for this NLU pipeline (You can copy paste the examples) :')
        # list of tuples, where first element is component name and second element is list of param tuples, all ready formatted for printing
        all_outputs = []

        for i, component_key in enumerate(self.keys()):
            s = ">>> pipe['" + component_key + "'] has settable params:"
            p_map = self[component_key].extractParamMap()

            component_outputs = []
            max_len = 0
            for key in p_map.keys():
                if "outputCol" in key.name or "labelCol" in key.name or "inputCol" in key.name or "labelCol" in key.name or 'lazyAnnotator' in key.name or 'storageref' in key.name: continue
                # print("pipe['"+ component_key +"'].set"+ str( key.name[0].capitalize())+ key.name[1:]+"("+str(p_map[key])+")" + " | Info: " + str(key.doc)+ " currently Configured as : "+str(p_map[key]) )
                # print("Param Info: " + str(key.doc)+ " currently Configured as : "+str(p_map[key]) )

                if type(p_map[key]) == str:
                    s1 = "pipe['" + component_key + "'].set" + str(key.name[0].capitalize()) + key.name[
                                                                                               1:] + "('" + str(
                        p_map[key]) + "') "
                else:
                    s1 = "pipe['" + component_key + "'].set" + str(key.name[0].capitalize()) + key.name[1:] + "(" + str(
                        p_map[key]) + ") "

                s2 = " | Info: " + str(key.doc) + " | Currently set to : " + str(p_map[key])
                if len(s1) > max_len: max_len = len(s1)
                component_outputs.append((s1, s2))

            all_outputs.append((s, component_outputs))

        # make strings aligned
        form = "{:<" + str(max_len) + "}"
        for o in all_outputs:
            print(o[0])  # component name
            for o_parm in o[1]:
                if len(o_parm[0]) < max_len:
                    print(form.format(o_parm[0]) + o_parm[1])
                else:
                    print(o_parm[0] + o_parm[1])

