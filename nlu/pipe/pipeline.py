from pyspark.sql.types import StructType, StructField, StringType
from nlu.pipe.utils.resolution.storage_ref_utils import StorageRefUtils
from nlu.pipe.utils.component_utils import ComponentUtils
from nlu.pipe.utils.output_level_resolution_utils import OutputLevelUtils
from nlu.pipe.utils.data_conversion_utils import DataConversionUtils
from nlu.utils.environment.env_utils import is_running_in_databricks
from nlu.pipe.col_substitution.col_name_substitution_utils import ColSubstitutionUtils
from nlu.universe.universes import Licenses
from nlu.pipe.nlu_component import NluComponent
from nlu.pipe.extractors.extractor_methods.base_extractor_methods import *
from typing import List, Union
from sparknlp.base import *
from sparknlp.base import LightPipeline
from nlu.pipe.extractors.extractor_configs_HC import default_full_config
from nlu.pipe.pipe_logic import PipeUtils
from nlu.universe.component_universes import NLP_NODE_IDS
import nlu.pipe.pipe_component
import sparknlp
import pyspark
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('nlu')


class NLUPipeline(dict):
    # we inherhit from dict so the component_list is indexable and we have a nice shortcut for accessing the spark nlp model
    def __init__(self):
        """ Initializes a pretrained pipeline, should only be created after a
        Spark Context has been created
          """
        self.spark = sparknlp.start()
        self.provider = 'sparknlp'
        self.pipe_ready = False  # ready when we have created a spark df
        self.failed_pyarrow_conversion = False
        self.anno2final_cols = []  # Maps Anno to output pandas col
        self.light_spark_transformer_pipe = None
        self.contains_ocr_components = False
        self.has_nlp_components = False
        self.nlu_ref = ''
        self.raw_text_column = 'text'
        self.raw_text_matrix_slice = 1  # place holder for getting text from matrix
        self.spark_nlp_pipe = None
        self.has_trainable_components = False
        self.needs_fitting = True
        self.is_fitted = False
        self.output_positions = False  # Wether to putput positions of Features in the final output. E.x. positions of tokens, entities, dependencies etc.. inside of the input document.
        self.prediction_output_level = ''  # either document, chunk, sentence, token
        self.component_output_level = ''  # document or sentence, depending on how input dependent Sentence/Doc classifier are fed
        self.output_different_levels = True
        self.light_pipe_configured = False
        self.spark_non_light_transformer_pipe = None
        self.components = []  # orderd list of nlu_component objects
        self.output_datatype = 'pandas'  # What data type should be returned after predict either spark, pandas, modin, numpy, string or array
        self.lang = 'en'
        self.spark_transformer_pipe = None
        self.spark_estimator_pipe_pipe = None
        self.has_licensed_components = False

    def add(self, component: NluComponent, nlu_reference="default_name", pretrained_pipe_component=False,
            name_to_add='', idx = None):
        '''

        :param component:
        :param nlu_reference: NLU references, passed for components that are used specified and not automatically generate by NLU
        :return:
        '''
        nlu_reference = component.nlu_ref
        if idx :
            self.components.insert(idx,component)
        else :
            self.components.append(component)
        # ensure that input/output cols are properly set
        # Spark NLP model reference shortcut
        name = component.name  # .replace(' ', '').replace('train.', '')

        if StorageRefUtils.has_storage_ref(component) and component.is_trained:
            # Converters have empty storage ref intially
            storage_ref = StorageRefUtils.extract_storage_ref(component)
            if storage_ref != '':
                name = name + '@' + StorageRefUtils.extract_storage_ref(component)
        logger.info(f"Adding {name} to internal component_list")

        # Configure output column names of classifiers from category to something more meaningful
        # if self.isInstanceOfNlpClassifer(component_to_resolve.model): self.configure_outputs(component_to_resolve, nlu_ref)

        if name_to_add == '':
            # Add Component as self.index and in attributes
            if component.is_storage_ref_producer and nlu_reference not in self.keys() and not pretrained_pipe_component:
                self[name] = component.model
            elif name not in self.keys():
                self[name] = component.model
            else:
                nlu_identifier = ComponentUtils.get_nlu_ref_identifier(component)
                self[name + "@" + nlu_identifier] = component.model
        else:
            self[name_to_add] = component.model

    def get_sample_spark_dataframe(self):
        data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami']}
        text_df = pd.DataFrame(data)
        return sparknlp.start().createDataFrame(data=text_df)

    def verify_all_labels_exist(self, dataset):
        return 'y' in dataset.columns  # or 'label' in dataset.columns or 'labels' in dataset.columns

    def fit(self, dataset=None, dataset_path=None, label_seperator=','):
        '''
        if dataset is  string with '/' in it, its dataset path!
        Converts the input Pandas Dataframe into a Spark Dataframe and trains a model on it.
        :param dataset: The pandas dataset to train on, should have a y column for label and 'text' column for text features
        :param dataset_path: Path to a CONLL2013 format dataset. It will be read for NER and POS training.
        :param label_seperator: If multi_classifier is trained, this seperator is used to split the elements into an Array column for Pyspark
        :return: A nlu pipeline with models fitted.
        '''
        stages = []
        for component in self.components: stages.append(component.model)
        self.spark_estimator_pipe = Pipeline(stages=stages)

        if dataset_path != None and 'ner' in self.nlu_ref:
            from sparknlp.training import CoNLL
            s_df = CoNLL().readDataset(self.spark, path=dataset_path, )
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(s_df.withColumnRenamed('label', 'y'))
            self.light_spark_transformer_pipe = LightPipeline(self.spark_transformer_pipe)
        elif dataset_path != None and 'pos' in self.nlu_ref:
            from sparknlp.training import POS
            s_df = POS().readDataset(self.spark, path=dataset_path, delimiter=label_seperator, outputPosCol="y",
                                     outputDocumentCol="document", outputTextCol="text")
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(s_df)
            self.light_spark_transformer_pipe = LightPipeline(self.spark_transformer_pipe)

        elif isinstance(dataset, pd.DataFrame) and 'multi' in self.nlu_ref:
            schema = StructType([
                StructField("y", StringType(), True),
                StructField("text", StringType(), True)
            ])
            from pyspark.sql import functions as F
            df = self.spark.createDataFrame(data=dataset).withColumn('y', F.split('y', label_seperator))
            # df = self.spark.createDataFrame(data=dataset, schema=schema).withColumn('y',F.split('y',label_seperator))
            # df = self.spark.createDataFrame(dataset)
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(df)
            self.light_spark_transformer_pipe = LightPipeline(self.spark_transformer_pipe)

        elif isinstance(dataset, pd.DataFrame):
            if not self.verify_all_labels_exist(dataset): raise ValueError(
                f"Could not detect label in provided columns={dataset.columns}\nMake sure a column named label, labels or y exists in your dataset.")
            dataset.y = dataset.y.apply(str)
            if self.has_licensed_components:
                # Configure Feature Assembler
                for c in self.components:
                    if c.name == 'feature_assembler':
                        vector_assembler_input_cols = [c for c in dataset.columns if
                                                       c != 'text' and c != 'y' and c != 'label' and c != 'labels']
                        c.model.setInputCols(vector_assembler_input_cols)
                        # os_components.model.spark_input_column_names = vector_assembler_input_cols
                # Configure Chunk resolver Sentence to Document substitution in all cols. when training Chunk resolver, we must substitute all SENTENCE cols with DOC. We MAY NOT FEED SENTENCE to CHUNK RESOLVE or we get errors
                self.components = PipeUtils.configure_component_output_levels_to_document(self)

            # Substitute @ notation to ___ because it breaks Pyspark SQL Parser...
            for c in self.components:
                for inp in c.spark_input_column_names:
                    if 'chunk_embedding' in inp:
                        c.spark_input_column_names.remove(inp)
                        c.spark_input_column_names.append(inp.replace('@', "___"))
                        c.model.setInputCols(c.spark_input_column_names)
                    if 'sentence_embedding' in inp:
                        c.spark_input_column_names.remove(inp)
                        c.spark_input_column_names.append(inp.replace('@', "___"))
                        c.model.setInputCols(c.spark_input_column_names)
                for out in c.spark_output_column_names:
                    if 'chunk_embedding' in out:
                        c.spark_output_column_names.remove(out)
                        c.spark_output_column_names.append(out.replace('@', "___"))
                        c.model.setOutputCol(c.spark_output_column_names[0])
                    if 'sentence_embedding' in out:
                        c.spark_output_column_names.remove(out)
                        c.spark_output_column_names.append(out.replace('@', "___"))
                        c.model.setOutputCol(c.spark_output_column_names[0])

            stages = []
            for component in self.components: stages.append(component.model)
            ## TODO set storage ref on fitted model
            self.spark_estimator_pipe = Pipeline(stages=stages)
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(
                DataConversionUtils.pdf_to_sdf(dataset, self.spark)[0])
            self.light_spark_transformer_pipe = LightPipeline(self.spark_transformer_pipe)

        elif isinstance(dataset, pyspark.sql.DataFrame):
            if not self.verify_all_labels_exist(dataset): raise ValueError(
                f"Could not detect label in provided columns={dataset.columns}\nMake sure a column named label, labels or y exists in your dataset.")
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(
                DataConversionUtils.sdf_to_sdf(dataset, self.spark))
            self.light_spark_transformer_pipe = LightPipeline(self.spark_transformer_pipe)

        else:
            # fit on empty dataframe since no data provided
            if not self.is_fitted:
                logger.info(
                    'Fitting on empty Dataframe, could not infer correct training method. This is intended for non-trainable pipelines.')
                self.spark_transformer_pipe = self.spark_estimator_pipe.fit(self.get_sample_spark_dataframe())
                self.light_spark_transformer_pipe = LightPipeline(self.spark_transformer_pipe)

        self.has_trainable_components = False
        self.is_fitted = True
        self.light_pipe_configured = True
        self.components = PipeUtils.replace_untrained_component_with_trained(self, self.spark_transformer_pipe)

        return self


    def get_extraction_configs(self, full_meta, positions, get_embeddings):
        """Search first OC namespace and if not found the HC Namespace for each Annotator Class in pipeline and get
        corresponding config Returns a dictionary of methods, where keys are column names values are methods  that
        are applied to extract and represent the data in these columns in a more pythonic and panda-esque way
        """
        c_level_mapping = OutputLevelUtils.get_output_level_mapping_by_component(self)
        # todo doc level annos and same level annos can be popped always.
        anno_2_ex_config = {}
        for c in self.components:
            if c.license == Licenses.ocr:
                anno_2_ex_config[c.spark_output_column_names[0]] = c.pdf_extractor_methods['default'](
                    output_col_prefix=c.spark_output_column_names[0])
                continue
            if 'embedding' in c.type and not get_embeddings:
                continue

            if 'default' in c.pdf_extractor_methods.keys() and not full_meta:
                anno_2_ex_config[c.spark_output_column_names[0]] = c.pdf_extractor_methods['default'](
                    output_col_prefix=c.spark_output_column_names[0])
            elif 'default_full' in c.pdf_extractor_methods.keys() and full_meta:
                anno_2_ex_config[c.spark_output_column_names[0]] = c.pdf_extractor_methods['default_full'](
                    output_col_prefix=c.spark_output_column_names[0])
            else:
                # Fallback if no output defined
                anno_2_ex_config[c.spark_output_column_names[0]] = default_full_config(
                    output_col_prefix=c.spark_output_column_names[0])

            # Tune the Extractor configs based on prediction parameters
            if c_level_mapping[c] == 'document' and not anno_2_ex_config[c.spark_output_column_names[0]].pop_never:
                # Disable popping for doc level outputs, output will not be [element] but instead element in each row.
                anno_2_ex_config[c.spark_output_column_names[0]].pop_meta_list = True
                anno_2_ex_config[c.spark_output_column_names[0]].pop_result_list = True
            if positions:
                anno_2_ex_config[c.spark_output_column_names[0]].get_positions = True
            else:
                anno_2_ex_config[c.spark_output_column_names[0]].get_begin = False
                anno_2_ex_config[c.spark_output_column_names[0]].get_end = False
                anno_2_ex_config[c.spark_output_column_names[0]].get_positions = False

            if c.loaded_from_pretrained_pipe:
                # Use original col name of pretrained pipes as prefix
                anno_2_ex_config[c.spark_output_column_names[0]].output_col_prefix = \
                    c.spark_output_column_names[0]

        return anno_2_ex_config

    def unpack_and_apply_extractors(self, pdf: Union[pyspark.sql.DataFrame, pd.DataFrame], keep_stranger_features=True,
                                    stranger_features=[],
                                    anno_2_ex_config={},
                                    light_pipe_enabled=True,
                                    get_embeddings=False
                                    ) -> pd.DataFrame:
        """1. Unpack SDF to PDF with Spark NLP Annotator Dictionaries
           2. Get the extractor configs for the corresponding Annotator classes
           3. Apply The extractor configs with the extractor methods to each column and merge back with zip/explode
           Uses optimized PyArrow conversion to avoid representing data multiple times between the JVM and PVM

           Can process Spark DF output from Vanilla pipes and Pandas Converts outputs of Lightpipeline
           """

        if light_pipe_enabled and not get_embeddings and not isinstance(pdf, pyspark.sql.dataframe.DataFrame):
            return apply_extractors_and_merge(extract_light_pipe_rows(pdf),
                                              anno_2_ex_config, keep_stranger_features, stranger_features)

        if not self.failed_pyarrow_conversion and self.check_pyspark_pyarrow_optimization_compatibility():
            from nlu.pipe.utils.pyarrow_conversion.pa_conversion import PaConversionUtils
            try:
                # Custom Pyarrow Conversion
                return apply_extractors_and_merge(
                    PaConversionUtils.convert_via_pyarrow(pdf).applymap(extract_pyarrow_rows),
                    anno_2_ex_config, keep_stranger_features, stranger_features)
            except:
                #     Default Conversion, No PyArrow (auto-Schema-Inferrence from PyArrow failed)
                self.failed_pyarrow_conversion = True
                return apply_extractors_and_merge(pdf.toPandas().applymap(extract_pyspark_rows), anno_2_ex_config,
                                                  keep_stranger_features, stranger_features)
        else:
            # Vanilla Spark Pipe
            return apply_extractors_and_merge(pdf.toPandas().applymap(extract_pyspark_rows), anno_2_ex_config,
                                              keep_stranger_features, stranger_features)

    def pythonify_spark_dataframe(self, processed,
                                  keep_stranger_features=True,
                                  stranger_features=[],
                                  drop_irrelevant_cols=True,
                                  output_metadata=False,
                                  positions=False,
                                  output_level='',
                                  get_embeddings=True):
        '''
        This functions takes in a spark dataframe with Spark NLP annotations in it and transforms it into a Pandas
        Dataframe with common feature types for further NLP/NLU downstream tasks. It will recycle Indexes from Pandas
        DataFrames and Series if they exist, otherwise a custom id column will be created which is used as index later
        on It does this by performing the following consecutive steps : 1. Select columns to explode 2. Select
        columns to keep 3. Rename columns 4. Create Pandas Dataframe object
        :param processed: Spark dataframe which an NLU pipeline has transformed
        :param output_level: The output level at which returned pandas Dataframe should be
        :param keep_stranger_features : Whether to keep additional features from the input DF when generating the output
                                        DF or if they should be discarded for the final output DF
        :param stranger_features: A list of features which are not known to NLU and inside the input DF.
                                    Basically all columns, which are not named 'text' in the input.
                                    If keep_stranger_features== True, then these features will be exploded,
                                    if pipe_prediction_output_level == document, otherwise they will not be exploded
        :param output_metadata: Whether to keep or drop additional metadata or predictions, like prediction confidence
        :return: Pandas dataframe which easy accessible features
        '''
        stranger_features += ['origin_index']
        if output_level == '':
            # Infer output level if none defined
            self.prediction_output_level = OutputLevelUtils.infer_prediction_output_level(self)
            logger.info(f'Inferred and set output level of pipeline to {self.prediction_output_level}')
        else:
            self.prediction_output_level = output_level

        # Get mapping from component to feature extractor method configs
        anno_2_ex_config = self.get_extraction_configs(output_metadata, positions, get_embeddings)

        # Processed becomes pandas after applying extractors
        processed = self.unpack_and_apply_extractors(processed, keep_stranger_features, stranger_features,
                                                     anno_2_ex_config, self.light_pipe_configured, get_embeddings)

        # Get mapping between column_name and pipe_prediction_output_level
        same_level = OutputLevelUtils.get_columns_at_same_level_of_pipe(self, processed, anno_2_ex_config,
                                                                        get_embeddings)
        logger.info(f"Extracting for same_level_cols = {same_level}\n")
        processed = zip_and_explode(processed, same_level)
        processed = self.convert_embeddings_to_np(processed)
        processed = ColSubstitutionUtils.substitute_col_names(processed, anno_2_ex_config, self, stranger_features,
                                                              get_embeddings)
        processed = processed.loc[:, ~processed.columns.duplicated()]

        if drop_irrelevant_cols:
            processed = processed[self.drop_irrelevant_cols(list(processed.columns))]
        # Sort cols alphabetically
        processed = processed.reindex(sorted(processed.columns), axis=1)
        return processed

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

    def finalize_return_datatype(self, df):
        '''
        Take in a Spark dataframe with only relevant columns remaining.
        Depending on what value is set in self.output_datatype, this method will cast the final SDF into Pandas/Spark/Numpy/Modin/List objects
        :param df:
        :return: The predicted Data as datatype dependign on self.output_datatype
        '''
        if self.output_datatype == 'spark':
            return df
        elif self.output_datatype == 'pandas':
            return df
        elif self.output_datatype == 'modin':
            import modin.pandas as mpd
            return mpd.DataFrame(df)
        elif self.output_datatype == 'pandas_series':
            return df
        elif self.output_datatype == 'modin_series':
            import modin.pandas as mpd
            return mpd.DataFrame(df)
        elif self.output_datatype == 'numpy':
            return df.to_numpy()
        return df

    def drop_irrelevant_cols(self, cols, keep_origin_index=False):
        '''
        Takes in a list of column names removes the elements which are irrelevant to the current output level.
        This will be run before returning the final df
        Drop column candidates are document, sentence, token, chunk.
        columns which are NOT AT THE SAME output level will be dropped
        :param cols:  list of column names in the df
        :return: list of columns with the irrelevant names removed
        '''
        if 'doc2chunk' in cols: cols.remove('doc2chunk')

        if self.prediction_output_level == 'token':
            if 'document' in cols: cols.remove('document')
            if 'chunk' in cols: cols.remove('chunk')
            if 'sentence' in cols: cols.remove('sentence')
        if self.prediction_output_level == 'sentence':
            if 'token' in cols: cols.remove('token')
            if 'chunk' in cols: cols.remove('chunk')
            if 'document' in cols: cols.remove('document')
        if self.prediction_output_level == 'chunk':
            # if 'document' in cols: cols.remove('document')
            if 'token' in cols: cols.remove('token')
            if 'sentence' in cols: cols.remove('sentence')
        if self.prediction_output_level == 'document':
            if 'token' in cols: cols.remove('token')
            if 'chunk' in cols: cols.remove('chunk')
            if 'sentence' in cols: cols.remove('sentence')
        if self.prediction_output_level == 'relation':
            if 'token' in cols: cols.remove('token')
            if 'chunk' in cols: cols.remove('chunk')
            if 'sentence' in cols: cols.remove('sentence')
        if keep_origin_index == False and 'origin_index' in cols: cols.remove('origin_index')
        return cols

    def configure_light_pipe_usage(self, data_instances, use_multi=True, force=False):
        logger.info("Configuring Light Pipeline Usage")
        if data_instances > 50 or use_multi is False:
            logger.info("Disabling light pipeline")
            if not self.is_fitted:
                self.fit()
        else:
            if not self.light_spark_transformer_pipe or force:
                if not self.is_fitted:
                    self.fit()
                self.light_pipe_configured = True
                logger.info("Enabling light pipeline")
                self.light_spark_transformer_pipe = LightPipeline(self.spark_transformer_pipe, parse_embeddings=True)

    def save(self, path, component='entire_pipeline', overwrite=False):
        if nlu.is_running_in_databricks():
            if path.startswith('/dbfs/') or path.startswith('dbfs/'):
                nlu_path = path
                if path.startswith('/dbfs/'):
                    nlp_path = path.replace('/dbfs', '')
                else:
                    nlp_path = path.replace('dbfs', '')
            else:
                nlu_path = 'dbfs/' + path
                if path.startswith('/'):
                    nlp_path = path
                else:
                    nlp_path = '/' + path

            if not self.is_fitted and self.has_trainable_components:
                self.fit()
                self.is_fitted = True
            if component == 'entire_pipeline':
                self.spark_transformer_pipe.save(nlp_path)
        if overwrite and not nlu.is_running_in_databricks():
            import shutil
            shutil.rmtree(path, ignore_errors=True)
        if not self.is_fitted:
            self.fit()
            self.is_fitted = True
        if component == 'entire_pipeline':
            if isinstance(self.spark_transformer_pipe, LightPipeline):
                self.spark_transformer_pipe.pipeline_model.save(path)
            else:
                self.spark_transformer_pipe.save(path)
        else:
            if component in self.keys():
                self[component].save(path)
        print(f'Stored model in {path}')

    def predict(self,
                data,
                output_level='',
                positions=False,
                keep_stranger_features=True,
                metadata=False,
                multithread=True,
                drop_irrelevant_cols=True,
                return_spark_df=False,
                get_embeddings=True
                ):
        '''
        Annotates a Pandas Dataframe/Pandas Series/Numpy Array/Spark DataFrame/Python List strings /Python String

        :param data: Data to predict on
        :param output_level: output level, either document/sentence/chunk/token
        :param positions: whether to output indexes that map predictions back to position in origin string
        :param keep_stranger_features: whether to keep columns in the dataframe that are not generated by pandas. I.e.
                when you s a dataframe with 10 columns and only one of them is named text, the returned dataframe will only contain the text column when set to false
        :param metadata: whether to keep additional metadata in final df or not like
                confidiences of every possible class for preidctions.
        :param multithread: Whether to use multithreading based lightpipeline. In some cases, this may cause errors.
        :param drop_irellevant_cols: Whether to drop cols of different output levels, i.e. when predicting token level and
                dro_irrelevant_cols = True then chunk, sentence and Doc will be dropped
        :param return_spark_df: Prediction results will be returned right after transforming with the Spark NLP pipeline
        :param get_embeddings: Whether to return embeddings or not
        :return:
        '''
        from nlu.pipe.utils.predict_helper import __predict__
        return __predict__(self, data, output_level, positions, keep_stranger_features, metadata, multithread,
                           drop_irrelevant_cols, return_spark_df, get_embeddings)

    def print_info(self, minimal=True):
        '''
        Print out information about every component_to_resolve currently loaded in the component_list and their configurable parameters.
        If minimal is false, all Spark NLP Model parameters will be printed, including output/label/input cols and other attributes a NLU user should not touch. Useful for debugging.
        :return: None
        '''
        print('The following parameters are configurable for this NLU pipeline (You can copy paste the examples) :')
        # list of tuples, where first element is component_to_resolve name and second element is list of param tuples, all ready formatted for printing
        all_outputs = []
        iterable = None
        for i, component_key in enumerate(self.keys()):
            s = ">>> component_list['" + component_key + "'] has settable params:"
            p_map = self[component_key].extractParamMap()

            component_outputs = []
            max_len = 0
            for key in p_map.keys():
                if minimal:
                    if "outputCol" in key.name or "labelCol" in key.name or "inputCol" in key.name or "labelCol" in key.name or 'lazyAnnotator' in key.name or 'storageref' in key.name: continue

                # print("component_list['"+ component_key +"'].set"+ str( key.name[0].capitalize())+ key.name[1:]+"("+str(p_map[key])+")" + " | Info: " + str(key.doc)+ " currently Configured as : "+str(p_map[key]) )
                # print("Param Info: " + str(key.doc)+ " currently Configured as : "+str(p_map[key]) )

                if type(p_map[key]) == str:
                    s1 = "component_list['" + component_key + "'].set" + str(key.name[0].capitalize()) + key.name[
                                                                                                         1:] + "('" + str(
                        p_map[key]) + "') "
                else:
                    s1 = "component_list['" + component_key + "'].set" + str(key.name[0].capitalize()) + key.name[
                                                                                                         1:] + "(" + str(
                        p_map[key]) + ") "

                s2 = " | Info: " + str(key.doc) + " | Currently set to : " + str(p_map[key])
                if len(s1) > max_len: max_len = len(s1)
                component_outputs.append((s1, s2))

            all_outputs.append((s, component_outputs))

        # make strings aligned
        form = "{:<" + str(max_len) + "}"
        for o in all_outputs:
            print(o[0])  # component_to_resolve name
            for o_parm in o[1]:
                if len(o_parm[0]) < max_len:
                    print(form.format(o_parm[0]) + o_parm[1])
                else:
                    print(o_parm[0] + o_parm[1])

    def print_exception_err(self, err):
        '''Print information about exception during converting or transforming dataframe'''
        import sys
        logger.exception('Exception occured')
        e = sys.exc_info()
        print("No accepted Data type or usable columns found or applying the NLU models failed. ")
        print(
            "Make sure that the first column you pass to .predict() is the one that nlu should predict on OR rename the column you want to predict on to 'text'  ")
        print(
            "On try to reset restart Jupyter session and run the setup script again, you might have used too much memory")
        print('Full Stacktrace was', e)
        print('Additional info:')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        import os
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        err = sys.exc_info()[1]
        print(str(err))
        print(
            'Stuck? Contact us on Slack! https://join.slack.com/t/spark-nlp/shared_invite/zt-lutct9gm-kuUazcyFKhuGY3_0AMkxqA')

    def viz(self, text_to_viz: str, viz_type='', labels_to_viz=None, viz_colors={}, return_html=False,
            write_to_streamlit=False, streamlit_key='NLU_streamlit'):
        """Visualize predictions of a Pipeline, using Spark-NLP-Display
        text_to_viz : String to viz
        viz_type    : Viz type, one of [ner,dep,resolution,relation,assert]. If none defined, nlu will infer and apply all applicable viz
        labels_to_viz : Defines a subset of NER labels to viz i.e. ['PER'] , by default=[] which will display all labels. Applicable only for NER viz
        viz_colors  : Applicable for [ner, resolution, assert ] key = label, value=hex color, i.e. viz_colors={'TREATMENT':'#008080', 'problem':'#800080'}
        """
        from nlu.utils.environment.env_utils import install_and_import_package
        install_and_import_package('spark-nlp-display', import_name='sparknlp_display')
        if self.spark_transformer_pipe is None: self.fit()
        is_databricks_env = is_running_in_databricks()
        if return_html: is_databricks_env = True
        # self.configure_light_pipe_usage(1, force=True)
        from nlu.pipe.viz.vis_utils import VizUtils

        if viz_type == '': viz_type = VizUtils.infer_viz_type(self)
        # anno_res = self.spark_transformer_pipe.fullAnnotate(text_to_viz)[0]
        # anno_res = self.spark.createDataFrame(pd.DataFrame({'text':text_to_viz}))
        data, stranger_features, output_datatype = DataConversionUtils.to_spark_df(text_to_viz, self.spark,
                                                                                   self.raw_text_column)
        anno_res = self.spark_transformer_pipe.transform(data)
        anno_res = anno_res.collect()[0]
        if self.has_licensed_components == False:
            HTML = VizUtils.viz_OS(anno_res, self, viz_type, viz_colors, labels_to_viz, is_databricks_env,
                                   write_to_streamlit, streamlit_key)
        else:
            HTML = VizUtils.viz_HC(anno_res, self, viz_type, viz_colors, labels_to_viz, is_databricks_env,
                                   write_to_streamlit)
        if return_html or is_databricks_env: return HTML

    def viz_streamlit(self,
                      # Base Params
                      text: Union[str, List[
                          str], pd.DataFrame, pd.Series] = "Angela Merkel from Germany and Donald Trump from America dont share many opinions",
                      model_selection: List[str] = [],
                      # SIMILARITY PARAMS
                      similarity_texts: Tuple[str, str] = ('I love NLU <3', 'I love Streamlit <3'),
                      # UI PARAMS
                      title: str = 'NLU ❤️ Streamlit - Prototype your NLP startup in 0 lines of code🚀',
                      sub_title: str = 'Play with over 1000+ scalable enterprise NLP models',
                      side_info: str = None,
                      visualizers: List[str] = (
                              "dependency_tree", "ner", "similarity", "token_features", 'classification', 'manifold'),
                      show_models_info: bool = True,
                      show_model_select: bool = True,
                      show_viz_selection: bool = False,
                      show_logo: bool = True,
                      set_wide_layout_CSS: bool = True,
                      show_code_snippets: bool = False,
                      model_select_position: str = 'side',  # main or side
                      display_infos: bool = True,
                      key: str = "NLU_streamlit",
                      display_footer: bool = True,
                      num_similarity_cols: int = 2,

                      ) -> None:
        """Display Viz in streamlit"""
        # try: from nlu.component_list.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        try:
            from nlu.pipe.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        except  ImportError:
            print("You need to install Streamlit to run this functionality.")
        StreamlitVizBlockHandler.viz_streamlit_dashboard(self,
                                                         text,
                                                         model_selection,
                                                         similarity_texts,
                                                         title,
                                                         sub_title,
                                                         side_info,
                                                         visualizers,
                                                         show_models_info,
                                                         show_model_select,
                                                         show_viz_selection,
                                                         show_logo,
                                                         set_wide_layout_CSS,
                                                         show_code_snippets,
                                                         model_select_position,
                                                         display_infos,
                                                         key,
                                                         display_footer,
                                                         num_similarity_cols
                                                         )

    def viz_streamlit_token(
            self,
            text: str = 'NLU and Streamlit go together like peanutbutter and jelly',
            title: Optional[str] = "Token features",
            sub_title: Optional[str] = 'Pick from `over 1000+ models` on the left and `view the generated features`',
            show_feature_select: bool = True,
            features: Optional[List[str]] = None,
            metadata: bool = True,
            output_level: str = 'token',
            positions: bool = False,
            set_wide_layout_CSS: bool = True,
            generate_code_sample: bool = False,
            key="NLU_streamlit",
            show_model_select=True,
            model_select_position: str = 'side',  # main or side
            show_infos: bool = True,
            show_logo: bool = True,
            show_text_input: bool = True,

    ):
        try:
            from nlu.pipe.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        except  ImportError:
            print("You need to install Streamlit to run this functionality.")
        StreamlitVizBlockHandler.visualize_tokens_information(self, text, title, sub_title, show_feature_select,
                                                              features, metadata, output_level, positions,
                                                              set_wide_layout_CSS, generate_code_sample, key,
                                                              show_model_select, model_select_position, show_infos,
                                                              show_logo, show_text_input)

    def viz_streamlit_classes(
            self,  # nlu component_list
            text: Union[str, list, pd.DataFrame, pd.Series, List[str]] = (
                    'I love NLU and Streamlit and sunny days!', 'I hate rainy daiys', 'CALL NOW AND WIN 1000$M'),
            output_level: Optional[str] = 'document',
            title: Optional[str] = "Text Classification",
            sub_title: Optional[
                str] = 'View predicted `classes` and `confidences` for `hundreds of text classifiers` in `over 200 languages`',
            metadata: bool = False,
            positions: bool = False,
            set_wide_layout_CSS: bool = True,
            generate_code_sample: bool = False,
            key: str = "NLU_streamlit",
            show_model_selector: bool = True,
            model_select_position: str = 'side',
            show_infos: bool = True,
            show_logo: bool = True,
    ) -> None:
        try:
            from nlu.pipe.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        except  ImportError:
            print("You need to install Streamlit to run this functionality.")
        StreamlitVizBlockHandler.visualize_classes(self, text, output_level, title, sub_title, metadata, positions,
                                                   set_wide_layout_CSS, generate_code_sample, key, show_model_selector,
                                                   model_select_position, show_infos, show_logo)

    def viz_streamlit_dep_tree(
            self,  # nlu component_list
            text: str = 'Billy likes to swim',
            title: Optional[str] = "Dependency Parse & Part-of-speech tags",
            sub_title: Optional[
                str] = 'POS tags define a `grammatical label` for `each token` and the `Dependency Tree` classifies `Relations between the tokens` ',
            set_wide_layout_CSS: bool = True,
            generate_code_sample: bool = False,
            key="NLU_streamlit",
            show_infos: bool = True,
            show_logo: bool = True,
            show_text_input: bool = True,
    ) -> None:
        try:
            from nlu.pipe.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        except  ImportError:
            print("You need to install Streamlit to run this functionality.")
        StreamlitVizBlockHandler.visualize_dep_tree(self, text, title, sub_title, set_wide_layout_CSS,
                                                    generate_code_sample, key, show_infos, show_logo, show_text_input, )

    def viz_streamlit_ner(
            self,  # Nlu component_list
            text: str = 'Donald Trump from America and Angela Merkel from Germany do not share many views.',
            ner_tags: Optional[List[str]] = None,
            show_label_select: bool = True,
            show_table: bool = False,
            title: Optional[str] = "Named Entities",
            sub_title: Optional[
                str] = "Recognize various `Named Entities (NER)` in text entered and filter them. You can select from over `100 languages` in the dropdown.",
            colors: Dict[str, str] = {},
            show_color_selector: bool = False,
            set_wide_layout_CSS: bool = True,
            generate_code_sample: bool = False,
            key="NLU_streamlit",
            model_select_position: str = 'side',  # main or side
            show_model_select=True,
            show_infos: bool = True,
            show_logo: bool = True,
            show_text_input: bool = True,

    ):
        try:
            from nlu.pipe.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        except ImportError:
            print("You need to install Streamlit to run this functionality.")
        StreamlitVizBlockHandler.visualize_ner(self, text, ner_tags, show_label_select, show_table, title, sub_title,
                                               colors, show_color_selector, set_wide_layout_CSS, generate_code_sample,
                                               key, model_select_position, show_model_select, show_infos, show_logo,
                                               show_text_input)

    def viz_streamlit_word_similarity(
            self,  # nlu component_list
            texts: Union[Tuple[str, str], List[str]] = (
                    "Donald Trump likes to party!", "Angela Merkel likes to party!"),
            threshold: float = 0.5,
            title: Optional[str] = "Vectors & Scalar Similarity & Vector Similarity & Embedding Visualizations  ",
            sub_tile: Optional[
                str] = "Visualize a `word-wise similarity matrix` and calculate `similarity scores` for `2 texts` and every `word embedding` loaded",
            write_raw_pandas: bool = False,
            display_embed_information: bool = True,
            similarity_matrix=True,
            show_algo_select: bool = True,
            dist_metrics: List[str] = ('cosine'),
            set_wide_layout_CSS: bool = True,
            generate_code_sample: bool = False,
            key: str = "NLU_streamlit",
            num_cols: int = 2,
            display_scalar_similarities: bool = False,
            display_similarity_summary: bool = False,
            model_select_position: str = 'side',
            show_infos: bool = True,
            show_logo: bool = True,

    ):
        try:
            from nlu.pipe.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        except ImportError:
            print("You need to install Streamlit to run this functionality.")
        StreamlitVizBlockHandler.display_word_similarity(self, texts, threshold, title, sub_tile, write_raw_pandas,
                                                         display_embed_information, similarity_matrix, show_algo_select,
                                                         dist_metrics, set_wide_layout_CSS, generate_code_sample, key,
                                                         num_cols, display_scalar_similarities,
                                                         display_similarity_summary, model_select_position, show_infos,
                                                         show_logo, )

    def viz_streamlit_word_embed_manifold(self,
                                          default_texts: List[str] = (
                                                  "Donald Trump likes to party!", "Angela Merkel likes to party!",
                                                  'Peter HATES TO PARTTY!!!! :('),
                                          title: Optional[
                                              str] = "Lower dimensional Manifold visualization for word embeddings",
                                          sub_title: Optional[
                                              str] = "Apply any of the 11 `Manifold` or `Matrix Decomposition` algorithms to reduce the dimensionality of `Word Embeddings` to `1-D`, `2-D` and `3-D` ",
                                          write_raw_pandas: bool = False,
                                          default_algos_to_apply: List[str] = ('TSNE', 'PCA',),
                                          target_dimensions: List[int] = (1, 2, 3),
                                          show_algo_select: bool = True,
                                          show_embed_select: bool = True,
                                          show_color_select: bool = True,
                                          MAX_DISPLAY_NUM: int = 100,
                                          display_embed_information: bool = True,
                                          set_wide_layout_CSS: bool = True,
                                          num_cols: int = 3,
                                          model_select_position: str = 'side',  # side or main
                                          key: str = "NLU_streamlit",
                                          additional_classifiers_for_coloring: List[str] = ['pos', 'sentiment.imdb'],
                                          generate_code_sample: bool = False,
                                          show_infos: bool = True,
                                          show_logo: bool = True,
                                          n_jobs: Optional[int] = 3,  # False
                                          ):
        try:
            from nlu.pipe.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        except ImportError:
            print("You need to install Streamlit to run this functionality.")
        StreamlitVizBlockHandler.viz_streamlit_word_embed_manifold(self,
                                                                   default_texts,
                                                                   title,
                                                                   sub_title,
                                                                   write_raw_pandas,
                                                                   default_algos_to_apply,
                                                                   target_dimensions,
                                                                   show_algo_select,
                                                                   show_embed_select,
                                                                   show_color_select,
                                                                   MAX_DISPLAY_NUM,
                                                                   display_embed_information,
                                                                   set_wide_layout_CSS,
                                                                   num_cols,
                                                                   model_select_position,
                                                                   key,
                                                                   additional_classifiers_for_coloring,
                                                                   generate_code_sample,
                                                                   show_infos,
                                                                   show_logo,
                                                                   n_jobs, )

    def viz_streamlit_sentence_embed_manifold(self,
                                              default_texts: List[str] = (
                                                      "Donald Trump likes to party!", "Angela Merkel likes to party!",
                                                      'Peter HATES TO PARTTY!!!! :('),
                                              title: Optional[
                                                  str] = "Lower dimensional Manifold visualization for sentence embeddings",
                                              sub_title: Optional[
                                                  str] = "Apply any of the 11 `Manifold` or `Matrix Decomposition` algorithms to reduce the dimensionality of `Word Embeddings` to `1-D`, `2-D` and `3-D` ",
                                              write_raw_pandas: bool = False,
                                              default_algos_to_apply: List[str] = ('TSNE', 'PCA',),
                                              target_dimensions: List[int] = (1, 2, 3),
                                              show_algo_select: bool = True,
                                              show_embed_select: bool = True,
                                              show_color_select: bool = True,
                                              MAX_DISPLAY_NUM: int = 100,
                                              display_embed_information: bool = True,
                                              set_wide_layout_CSS: bool = True,
                                              num_cols: int = 3,
                                              model_select_position: str = 'side',  # side or main
                                              key: str = "NLU_streamlit",
                                              additional_classifiers_for_coloring: List[str] = ['sentiment.imdb'],
                                              generate_code_sample: bool = False,
                                              show_infos: bool = True,
                                              show_logo: bool = True,
                                              n_jobs: Optional[int] = 3,  # False
                                              ):
        try:
            from nlu.pipe.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        except ImportError:
            print("You need to install Streamlit to run this functionality.")
        StreamlitVizBlockHandler.viz_streamlit_sentence_embed_manifold(self,
                                                                       default_texts,
                                                                       title,
                                                                       sub_title,
                                                                       write_raw_pandas,
                                                                       default_algos_to_apply,
                                                                       target_dimensions,
                                                                       show_algo_select,
                                                                       show_embed_select,
                                                                       show_color_select,
                                                                       MAX_DISPLAY_NUM,
                                                                       display_embed_information,
                                                                       set_wide_layout_CSS,
                                                                       num_cols,
                                                                       model_select_position,
                                                                       key,
                                                                       additional_classifiers_for_coloring,
                                                                       generate_code_sample,
                                                                       show_infos,
                                                                       show_logo,
                                                                       n_jobs, )

    def viz_streamlit_entity_embed_manifold(self,
                                            default_texts: List[str] = ("Donald Trump likes to visit New York",
                                                                        "Angela Merkel likes to visit Berlin!",
                                                                        'Peter hates visiting Paris'),
                                            title: Optional[
                                                str] = "Lower dimensional Manifold visualization for Entity embeddings",
                                            sub_title: Optional[
                                                str] = "Apply any of the 11 `Manifold` or `Matrix Decomposition` algorithms to reduce the dimensionality of `Entity Embeddings` to `1-D`, `2-D` and `3-D` ",
                                            default_algos_to_apply: List[str] = ("TSNE", "PCA"),
                                            target_dimensions: List[int] = (1, 2, 3),
                                            show_algo_select: bool = True,
                                            set_wide_layout_CSS: bool = True,
                                            num_cols: int = 3,
                                            model_select_position: str = 'side',  # side or main
                                            key: str = "NLU_streamlit",
                                            show_infos: bool = True,
                                            show_logo: bool = True,
                                            n_jobs: Optional[int] = 3,  # False
                                            ):
        try:
            from nlu.pipe.viz.streamlit_viz.streamlit_dashboard_OS import StreamlitVizBlockHandler
        except ImportError:
            print("You need to install Streamlit to run this functionality.")
        StreamlitVizBlockHandler.viz_streamlit_entity_embed_manifold(self,
                                                                     default_texts,
                                                                     title,
                                                                     sub_title,
                                                                     default_algos_to_apply,
                                                                     target_dimensions,
                                                                     show_algo_select,
                                                                     set_wide_layout_CSS,
                                                                     num_cols,
                                                                     model_select_position,
                                                                     key,
                                                                     show_infos,
                                                                     show_logo,
                                                                     n_jobs)

    def check_pyspark_pyarrow_optimization_compatibility(self):
        # Only works for pyspark        "3.1.2"
        v = pyspark.version.__version__.split('.')
        if int(v[0]) == 3 and int(v[1]) >= 1: return True
