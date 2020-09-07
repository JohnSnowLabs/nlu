# BASE PIPELINE CLASS
from sparknlp import pretrained
import sparknlp
import pandas as pd
import numpy as np
from sparknlp.base import *
import logging
import nlu
logger = logging.getLogger('nlu')


from pyspark.sql.functions import flatten, explode, arrays_zip, map_keys, map_values, monotonically_increasing_id, greatest,expr
from pyspark.sql.functions import col as pyspark_col
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType,FloatType, StringType, DoubleType


class BasePipe(dict):
    # we inherhit from dict so the pipe is indexable and we have a nice shortcut for accessing the spark nlp model
    def __init__(self):
        self.raw_text_column = 'text'
        self.raw_text_matrix_slice = 1  # place holder for getting text from matrix
        self.spark_nlp_pipe = None
        self.needs_fitting = True
        self.is_fitted = False
        self.output_positions = False  # Wether to putput positions of Features in the final output. E.x. positions of tokens, entities, dependencies etc.. inside of the input document.
        self.output_level = ''  # either document, chunk, sentence, token
        self.output_different_levels = True
        self.pipe_components = []                                         # orderd list of nlu_component objects
        self.output_datatype = 'pandas' # What data type should be returned after predict either spark, pandas, modin, numpy, string or array 
    def add(self, component, component_name="auto_generate"):
        
        self.pipe_components.append(component)
        
        
        # Spark NLP model reference shortcut
        name = component.component_info.name.replace(' ','')
        if name not in self.keys() : self[name]=component.model 
        else : self[name]=component.model
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
            'token': ['token', 'pos', 'ner', 'lemma','lem','stem', 'stemm', 'word_embeddings', 'named_entity', 'entity', 'dependency',
                      'labeled_dependency', 'dep', 'dep.untyped', 'dep.typed'],
            'sentence': ['sentence', 'sentence_embeddings', ] + ['sentiment', 'classifer', 'category'],
            'chunk': ['chunk', 'embeddings_chunk', 'chunk_embeddings'],
            'document': ['document','language'],
            'embedding_level': [] #['sentiment', 'classifer'] # todo, wait for Spark NLP Getter/Setter fixes to implement this properly
            # embedding level  annotators output levels depend on the level of the embeddings they are fed. If we have Doc/Chunk/Word/Sentence embeddings, those annotators output at the same level.

        }

    def get_sample_spark_dataframe(self):
        data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami' ]}
        text_df = pd.DataFrame(data)
        return sparknlp.start().createDataFrame(data=text_df)


    def fit(self, dataset=None):
        if dataset == None:  # todo implement fitting on input datset
            #todo somehow regex matcher does not see document column?????>>?>!@>!?>??!@
            stages = []
            for component in self.pipe_components:
                stages.append(component.model)
            self.is_fitted = True
            self.spark_estimator_pipe = Pipeline(stages=stages)
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(self.get_sample_spark_dataframe())

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
                    # this is the component that feeds into the component we are trying to resolve the output  level for.  That is so, because the output of this component matches the input of the component we are resolving
                    return self.resolve_type_to_output_level(component.component_info.type)

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
            logger.info('Resolved output level for field_type=%s and field_name=%s to Sentence level', field_type, field_name)
            return 'sentence'
        if field_type in self.levels['token']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to Token level ', field_type, field_name)
            return 'token'
        if field_type in self.levels['sentence']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to sentence level', field_type, field_name)
            return 'sentence'
        if field_type in self.levels['chunk']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to Chunk level ', field_type, field_name)
            return 'chunk'
        if field_type in self.levels['document']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to document level', field_type, field_name)
            return 'document'
        if field_type in self.levels['embedding_level']:
            logger.info('Resolved output level for field_type=%s and field_name=%s to embeddings level', field_type, field_name)
            return self.get_output_level_of_embeddings_provider(field_type, field_name) #recursive resolution

    def resolve_outputlevel_to_int(self, output_level):
        '''
        @ param field_type : type of the spark field
        @ param name : name of thhe spark field
        This checks the levels dict for what the output level is for the input annotator type.
        If the annotator type depends on the embedding level, we need further checking.
        @ return : String, which corrosponds to the output level of this Component.
        '''
        if output_level == 'token' : return 0
        if output_level == 'sentence': return 1
        if output_level == 'chunk': return 2
        if output_level == 'document': return 3
        if output_level == 'language': return 3 # special case

    def resolve_int_outputlevel_to_str(self, output_level):
        '''
        This function maps output int levels back to string
        @ param output_level : Int level output
        @ return : String, which corrosponds to the output level of this Component.
        '''
        logger.info("resolving int output level to str")
        if output_level ==  0  : return 'token'
        if output_level == 1  : return 'sentence'
        if output_level == 2  : return 'chunk'
        if output_level == 3  : return 'document'
        if output_level == 3  : return 'language' # special case


    def get_field_types_dict(self,sdf, stranger_features):
        """
        @ param sdf: Spark Dataframe which a NLU/SparkNLP pipeline has transformed.
        This function returns a dictionary that maps column names to their spark annotator types.
        @return : Dictionary, Keys are spark column column names, value is the type of annotator
        """
        logger.info('Getting field types for output SDF')
        field_types_dict = {}
        
        
        
        for field in sdf.schema.fieldNames():
            if field in stranger_features : continue
            if field == self.raw_text_column: continue
            if 'label' in field: continue  # speciel case for input lables
            
            # For empty DF this will crash
            a_row = sdf.select(field + '.annotatorType').take(1)[0]['annotatorType']
            if len(a_row) > 0:
                a_type = a_row[0]
            else:
                logger.exception('Error there are no rows for this Component in the final Dataframe. For field=%s. It will be dropped in the final dataset', field)
                a_type = 'Error'  # (no results)
            field_types_dict[field] = a_type
            logger.info('Parsed type=%s  for field=%s', a_type, field)
        logger.info('Parsing field types done, parsed=%s', field_types_dict)
        return field_types_dict


    def rename_columns_and_extract_map_values(self,ptmp, fields_to_rename, same_output_level, stranger_features=[], meta=True):
        # This method takes in a Spark dataframe that is the result of an explosion or not after the spark Pipeline transformation .
        # It will peform the following transformations on the dataframe:
        # 1. Rename the exploded columns to something more meaningful
        # 2. Extract Meta data values of columns that contain maps if the data is relevant
        # 3. Store the new names
        # @ param ptmp The dataframe which contains the columns wto be renamed
        # @ param fields_to_nreame : A list of field names that will be renamed in the dataframe.
        # @ param same_output_level : Wether the fields that are going to be renamed are at the same output level as the pipe or at a different one.
        # @ param meta: wether  to get meta data like prediction confidence or not
        # @ return : Returns tuple (list, SparkDataFrame), where the first element is a list with all the new names and the second element is a new Spark Dataframe which contains all the renamed and also old columns

        logger.info('Renaming columns and extracting meta data for  outputlevel_same=%s and fields_to_rename=%s and get_meta=%s', same_output_level,fields_to_rename,meta)
        columns_for_select = []
        
        # edge case swap. We must rename .metadata fields before we get the .result fields or there will be errors because of column name overwrites.. So we swap position of them
        cols_to_swap = [field for field in fields_to_rename if '.metadata' in field]
        reorderd_fields_to_rename = fields_to_rename.copy()
        for swap in cols_to_swap : 
            name = swap.split('.')[0] +'.result'
            reorderd_fields_to_rename[reorderd_fields_to_rename.index(swap)], reorderd_fields_to_rename[reorderd_fields_to_rename.index(name)] = reorderd_fields_to_rename[reorderd_fields_to_rename.index(name)] , reorderd_fields_to_rename   [reorderd_fields_to_rename.index(swap)] 
            logger.info('Swapped selection order for  %s and %s before renaming ', swap, name)

        # second analogus edge case for positional fields (.begin and .end) and .result. We will put every rseult column into the end of the list and thus avoid the erronous case always
        for col in reorderd_fields_to_rename :
            if '.result' in col  : reorderd_fields_to_rename.append(reorderd_fields_to_rename.pop(reorderd_fields_to_rename.index(col)))
            
        

        if same_output_level == True :
            # fields that are at the same output level have been exploded.
            # thus we ened to use the res.1 etc.. reference to get the map values and keys
            for i, field in enumerate(reorderd_fields_to_rename):
                if field in stranger_features : continue
                if self.raw_text_column in field: continue
                new_field = field.replace('.', '_').replace('_result','').replace('_embeddings_embeddings','_embeddings')

                if 'metadata' in field :  # rename metadata to something more meaningful
                    logger.info('Getting Meta Data for   : nr=%s , name=%s with new_name=%s and original', i, field, new_field)
                    new_fields = []
                    # we iterate over the keys in the metadata and use them as new column names. The values will become the values in the columns.
                    keys_in_metadata = list(ptmp.select(field).take(1))
                    if len(keys_in_metadata) == 0 : continue # no resulting values for this column, we wont include it in the final output
                    keys_in_metadata = list(keys_in_metadata[0].asDict()['metadata'][0].keys()) #
                    
                    if meta == True :  # get all meta data 
                        for key in keys_in_metadata:
                            #drop sentences keys from Lang detector, they seem irrelevant.
                            if key == 'sentence' and 'language' in field  : continue 
                            new_fields.append(new_field.replace('metadata',key))
    
                            
                            ptmp = ptmp.withColumn(new_fields[-1],pyspark_col(('res.' + str(fields_to_rename.index(field)) + '.'+key) ))
    
                            columns_for_select.append(new_fields[-1])   
    
    
                            logger.info('Created Meta Data for : nr=%s , original Meta Data key name=%s and new  new_name=%s ', i, key,new_fields[-1])
                    else :  # Get only meta data with greatest value (highest prob)

                        cols_to_max = []
                        for key in keys_in_metadata: cols_to_max.append('res.' + str(fields_to_rename.index(field)) + '.'+key)

                        # sadly because the Spark SQL method 'greatest()' does not work properly on scientific notation, we must cast our metadata to decimal with limited precision
                        # scientific notation starts after 6 decimal places, so we can have at most exactly 6
                        # since greatest() breaks the dataframe Schema, we must rename the columns first or run into issues with Pysark Struct queriying
                        for key in cols_to_max : ptmp = ptmp.withColumn(key.replace('.','_'), pyspark_col(key).cast('decimal(7,6)'))
                        # casted = ptmp.select(*(pyspark_col(c).cast("decimal(6,6)").alias(c.replace('.','_')) for c in cols_to_max))
                        
                        max_confidence_name  = field.split('.')[0] +'_confidence'
                        renamed_cols_to_max = [col.replace('.','_') for col in cols_to_max]

                        if len(cols_to_max) > 1 :
                            ptmp = ptmp.withColumn(max_confidence_name , greatest(*renamed_cols_to_max))
                            columns_for_select.append(max_confidence_name)
                        else :
                            ptmp = ptmp.withColumnRenamed(renamed_cols_to_max[0], max_confidence_name  )
                            columns_for_select.append(max_confidence_name)

                    continue


                ptmp = ptmp.withColumn(new_field, ptmp['res.' + str(fields_to_rename.index(field))])  # get the outputlevel results row by row
                columns_for_select.append(new_field)
                logger.info('Renaming exploded field  : nr=%s , name=%s to new_name=%s', i, field,new_field)
            return ptmp, columns_for_select

        elif same_output_level==False :
            # This case works on the original Spark Columns which have beenn untouched sofar.
            for i, field in enumerate(reorderd_fields_to_rename):
                if self.raw_text_column in field: continue
                new_field = field.replace('.', '_').replace('_result','').replace('_embeddings_embeddings','_embeddings')
                if 'metadata' in field :
                    # since the have a field with metadata, the values of the original data for which we have metadata for must exist in the dataframe as singular elements inside of a list
                    # by applying the expr method, we unpack the elements from the list 
                    unpack_name = field.split('.')[0]
                    
                    ptmp = ptmp.withColumn(unpack_name+'_result', expr(unpack_name+'.result[0]'))
                    reorderd_fields_to_rename[reorderd_fields_to_rename.index(unpack_name+'.result')] = unpack_name+'_result' 
                    logger.info('Getting Meta Data for   : nr=%s , name=%s with new_name=%s and original', i, field,new_field)
                    # we iterate over the keys in the metadata and use them as new column names. The values will become the values in the columns.
                    keys_in_metadata = list(ptmp.select(field).take(1))
                    if len(keys_in_metadata) == 0 : continue
                    keys_in_metadata = list(keys_in_metadata[0].asDict()['metadata'][0].keys()) #
                    if 'sentence' in keys_in_metadata : keys_in_metadata.remove('sentence') # todo realy remove here?
                    
                    new_fields=[]
                    for key in keys_in_metadata: 
                        # we cant skip getting  key values for everything, even if meta=false. This is because we need to get the greatest of all confidence values , for this we must unpack them first..
                        new_fields.append(new_field.replace('metadata',key))

                        # These Pyspark UDF extracts from a list of maps all the map values for positive and negative confidence and also spell costs
                        def extract_map_values(x): return [float(sentence[key]) for sentence in x]

                        #extract map values for list of maps 
                        array_map_values = udf(lambda z: extract_map_values(z), ArrayType(FloatType()))

                        ptmp = ptmp.withColumn(new_fields[-1],array_map_values(field)) 
                        # We apply Expr here because all resulting meta data is inside of a list and just a single element, which we can take out 
                        ptmp = ptmp.withColumn(new_fields[-1],expr(new_fields[-1]+'[0]'))
                        logger.info('Created Meta Data for   : nr=%s , name=%s with new_name=%s and original', i, field,new_fields[-1])
                        if meta == True : continue
                        columns_for_select.append(new_fields[-1]) #?

                    if meta == True : continue 
                    else : # We gotta get the max confidence column, remove all other cols for selection
                        cols_to_max = []
                        prefix = field.split('.')[0]
                        for key in keys_in_metadata: cols_to_max.append( prefix+'_'+key)

                        #cast all the types to decimal, remove scientific notation
                        for key in cols_to_max : ptmp = ptmp.withColumn(key, pyspark_col(key).cast('decimal(7,6)'))

                        max_confidence_name  = field.split('.')[0] +'_confidence'
                        if len(cols_to_max) > 1 : 
                            ptmp = ptmp.withColumn(max_confidence_name , greatest(*cols_to_max))
                            columns_for_select.append(max_confidence_name)
                        else :
                            ptmp = ptmp.withColumnRenamed( cols_to_max[0], max_confidence_name )
                            columns_for_select.append(max_confidence_name)
                            
                        for f in new_fields:
                            # we remove the new fields becasue they duplicate the infomration of max confidence field
                            if f in columns_for_select: columns_for_select.remove(f)
                        

                    continue # end of special meta data case 
                

                ptmp = ptmp.withColumn(new_field, ptmp[field])  # get the outputlevel results row by row
                logger.info('Renaming non exploded field  : nr=%s , name=%s to new_name=%s', i, field,new_field)
                columns_for_select.append(new_field)
            return ptmp, columns_for_select



    def infer_and_set_output_level(self,sdf):
        '''
        This function checks the last component of the NLU pipeline and infers
        and infers from that the output level via checking the components info.
        It sets the output level of the pipe accordingly
        param sdf : Spark dataframe after transformations 
        '''
        new_output_level = self.pipe_components[-1].component_info.output_level
        if new_output_level == 'input_dependent' :
            reversed_pipe = self.pipe_components.copy()
            reversed_pipe.reverse()
            for i, component in enumerate(reversed_pipe):
                if i == 0 : continue # first element is always input dependent in this case
                if component.component_info.output_level == 'input_dependent' : continue
                self.output_level = component.component_info.component_info.output_level
        else :
            self.output_level=new_output_level
            #soemtimes in piplines cleaners will remove the sentence columns, thus we must check if it is early here if the level is there
        if not self.output_level  in sdf.columns :
            if 'document'  in sdf.columns: self.output_level='document'
            elif 'sentence' in sdf.columns: self.output_level='sentence' 
            elif 'chunk'  in sdf.columns: self.output_level='chunk'
            elif 'token'  in sdf.columns: self.output_level='token'
        logger.info('Inferred and set output level of pipeline to %s', self.output_level)



    def pythonify_spark_dataframe(self, processed, get_different_level_output=True, keep_stranger_features = True, stranger_features = [] , drop_irrelevant_cols=True, output_metadata=False):
        '''
        This functions takes in a spark dataframe with Spark NLP annotations in it and transforms it into a Pandas Dataframe with common feature types for further NLP/NLU downstream tasks.
            It does this by performing the following consecutive steps :
                1. Select columns to explode
                2. Select columns to keep
                3. Rename columns
                4. Create Pandas Dataframe object
        :param processed: Spark dataframe which an NLU pipeline has transformed
        :param output_level: The output level at which returned pandas Dataframe should be
        :param get_different_level_output:  Wheter to get features from different levels
        :param keep_stranger_features : Wheter to keep additional features from the input DF when generating the output DF or if they should be discarded for the final output DF 
        :param stranger_features: A list of features which are not known to NLU and inside of the input DF. 
                                    Basically all columns, which are not named 'text' in the input. 
                                    If keep_stranger_features== True, then these features will be exploded, if output_level == DOCUMENt, otherwise they will not be exploded
        :param output_metadata: Wether to keep or drop additional metadataf or predictions, like prediction confidence  
        :return: Pandas dataframe which easy accessable features
        '''
        
        
        if self.output_level==''  : self.infer_and_set_output_level(processed)
        
        
        field_dict = self.get_field_types_dict(processed, stranger_features) #map field to type of field
        not_at_same_output_level_fields = []
        same_output_level_fields = [self.output_level + '.result']
        logger.info('Setting Output level as : %s', self.output_level=='')

                
        if keep_stranger_features :
            sdf = processed.select(['*', monotonically_increasing_id().alias('id')])  # todo implement optional ID provided by user or use pandas index if there is one
        else :
            features_to_keep = list(set(processed.columns) - set(stranger_features))
            sdf = processed.select(features_to_keep +[ monotonically_increasing_id().alias('id')])  # todo implement optional ID provided by user or use pandas index if there is one

        # loop over all fields and decide which ones to keep and also if there are at the current output level or at a different one .
        # We sort the fields by output level. Thez are at the same as the current output level and will be exploded
        # Or they are at a different level, in which case they will be left as is and not zipped before explode
        for field in processed.schema.fieldNames():
            if field in stranger_features : continue
            if field == self.raw_text_column: continue
            if field == self.output_level: continue
            if 'label' in field and 'dependency' not in field: continue  # specal case for input labels

            f_type = field_dict[field]
            logger.info('Selecting Columns for field=%s of type=%s', field, f_type)
            if self.resolve_type_to_output_level(f_type, field) == self.output_level:
                logger.info('Setting field for field=%s of type=%s to output level SAME LEVEL', field, f_type)
                
                if 'embeddings' not in field and 'embeddings' not in f_type : same_output_level_fields.append(field + '.result')  # result of embeddigns is just the word/sentence
                if self.output_positions:
                    same_output_level_fields.append(field + '.begin')
                    same_output_level_fields.append(field + '.end')
                if 'embeddings' in f_type:
                    same_output_level_fields.append(field + '.embeddings')
                if 'category' in f_type  or 'spell' in f_type or 'sentiment' in f_type or 'class' in f_type or 'language' in f_type :
                    same_output_level_fields.append(field + '.metadata')

            else:
                logger.info('Setting field for field=%s of type=%s to output level NOT SAME LEVEL', field, f_type)

                if 'embeddings' not in field and 'embeddings' not in f_type: not_at_same_output_level_fields.append(field + '.result')  # result of embeddigns is just the word/sentence
                if self.output_positions:
                    not_at_same_output_level_fields.append(field + '.begin')
                    not_at_same_output_level_fields.append(field + '.end')
                if 'embeddings' in f_type:
                    not_at_same_output_level_fields.append(field + '.embeddings')
                if 'category' in f_type  or 'spell' in f_type or 'sentiment' in f_type or 'class' in f_type :
                    not_at_same_output_level_fields.append(field + '.metadata')

                
        if self.output_level == 'document':
            #explode stranger features if output level is document
            same_output_level_fields =list(set( same_output_level_fields + stranger_features))
                

        ptmp = sdf.withColumn("tmp", arrays_zip(*same_output_level_fields)).withColumn("res", explode('tmp'))
        final_select_not_at_same_output_level = []
        logger.info(' zipping and exploding columnns %s-', same_output_level_fields)

        # explode the columns which are at the same output level..if there are maps at the different output level we will get array maps.  then we use some UDF functions to extract the resulting array maps 
      
        ptmp,final_select_same_output_level =  self.rename_columns_and_extract_map_values(ptmp=ptmp, fields_to_rename=same_output_level_fields, same_output_level=True, stranger_features=stranger_features, meta=output_metadata)
        if get_different_level_output :
            ptmp,final_select_not_at_same_output_level = self.rename_columns_and_extract_map_values(ptmp=ptmp, fields_to_rename=not_at_same_output_level_fields, same_output_level=False, meta=output_metadata)
        

        if self.output_level != 'document':final_select_not_at_same_output_level+= stranger_features
            
        final_select_same_output_level.append('id')
        ptmp.toPandas()
        logger.info('Final cleanup select of same level =%s', final_select_same_output_level)
        logger.info('Final cleanup select of different level =%s', final_select_not_at_same_output_level)
        logger.info('Final ptmp columns = %s', ptmp.columns)
         
        final_cols = final_select_same_output_level + final_select_not_at_same_output_level
        if drop_irrelevant_cols : final_cols = self.drop_irrelevant_cols(final_cols)
        
        final_df = ptmp.select(list(set(final_cols)))
                
        return  self.finalize_return_datatype(final_df)

    def finalize_return_datatype(self, sdf):
        '''
        Take in a Spark dataframe with only relevant columns remaining.
        Depending on what value is set in self.output_datatype, this method will cast the final SDF into Pandas/Spark/Numpy/Modin/List objects
        :param sdf: 
        :return: The predicted Data as datatype dependign on self.output_datatype
        '''

        if self.output_datatype == 'spark' : 
            return sdf
        elif self.output_datatype == 'pandas' : 
            return sdf.toPandas()
        elif self.output_datatype == 'modin' :
            import modin.pandas as mpd
            return mpd.DataFrame(sdf.toPandas())
        # todo actual series and return String/array objects
        elif self.output_datatype == 'pandas_series' :
            return sdf.toPandas()
        elif self.output_datatype == 'modin_series' :
            import modin.pandas as mpd
            return mpd.DataFrame(sdf.toPandas())
        elif self.output_datatype == 'numpy' :
            return sdf.toPandas().to_numpy()
        elif self.output_datatype == 'string' :
            return sdf.toPandas()
        elif self.output_datatype == 'string_list' :
            return sdf.toPandas()
        elif self.output_datatype == 'array' :
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
        if self.output_level =='token':
            if 'document' in cols: cols.remove('document')
            if 'chunk' in cols: cols.remove('chunk')
            if 'sentence' in cols: cols.remove('sentence')
        if self.output_level =='sentence':
            if 'token' in cols: cols.remove('token')
            if 'chunk' in cols: cols.remove('chunk')
            if 'document' in cols: cols.remove('document')
        if self.output_level =='chunk':
            if 'document' in cols: cols.remove('document')
            if 'token' in cols: cols.remove('token')
            if 'sentence' in cols: cols.remove('sentence')
        if self.output_level =='document':
            if 'token' in cols: cols.remove('token')
            if 'chunk' in cols: cols.remove('chunk')
            if 'sentence' in cols: cols.remove('sentence')

        return cols

    def predict(self, data, output_level='', positions=False, keep_stranger_features=True, metadata=False):
        '''
        Annotates a Pandas Dataframe/Pandas Series/Numpy Array/Spark DataFrame/Python List strings /Python String
        
        :param data: 
        :param output_level: 
        :param positions: 
        :param keep_stranger_features: 
        :param metadata:weather to keep additonal metadata in final df or not 
        :return: 
        '''
        if output_level !='': self.output_level = output_level
        self.output_positions= positions
        if output_level=='chunk': 
            # If chunk not in pipe we must add it and run the query verifyier again 
            chunk_provided = False
            for component in self.pipe_components:
                if component.component_info.type =='chunker' : chunk_provided = True 
            if chunk_provided == False : 
                self.pipe_components.append(nlu.get_default_component_of_type('chunk'))
                self =  PipelineQueryVerifier.check_and_fix_nlu_pipeline(self)
        if not self.is_fitted: self.fit()
        sdf = None
        import pyspark  
        stranger_features = []
        try:
            if type(data) is  pyspark.sql.dataframe.DataFrame : # casting follows spark->pd
                self.output_datatype = 'spark'
                if self.raw_text_column in data.columns:
                    # store all stranger features 
                    if len(data.columns) > 1 :
                        stranger_features = list( set(data.columns) - set(self.raw_text_column))
                    sdf = self.spark_transformer_pipe.transform(data )
                else :
                    print('Could not find column named "text" in input Pandas Dataframe. Please ensure one column named such exists. Columns in DF are : ', data.columns)
            if type(data) is pd.DataFrame:  # casting follows pd->spark->pd
                self.output_datatype = 'pandas'
                
                if self.raw_text_column in data.columns:
                    if len(data.columns) > 1 :
                        data = data.where(pd.notnull(data), None) # make  Nans to None, or spark will crash
                        data = data.dropna(axis=1, how='all')
                        stranger_features = list( set(data.columns) - set(self.raw_text_column))
                    sdf = self.spark_transformer_pipe.transform(
                    # self.spark.createDataFrame(data[['text']]), ) # this takes text column as series and makes it DF
                    self.spark.createDataFrame(data ))
                else : 
                    print('Could not find column named "text" in input Pandas Dataframe. Please ensure one column named such exists. Columns in DF are : ', data.columns)
            elif type(data) is pd.Series:  # for df['text'] colum/series passing casting follows pseries->pdf->spark->pd
                self.output_datatype='pandas_series'
                data = pd.DataFrame(data).dropna(axis=1, how='all')
                
                if self.raw_text_column in data.columns: sdf = \
                    self.spark_transformer_pipe.transform(
                    self.spark.createDataFrame(data[['text']]), )
                else: print('Could not find column named "text" in  Pandas Dataframe generated from input  Pandas Series. Please ensure one column named such exists. Columns in DF are : ', data.columns)
    
            elif type(data) is np.ndarray:  # This is a bit inefficient. Casting follow  np->pd->spark->pd. We could cut out the first pd step
                self.output_datatype='numpy_array'
                if len(data.shape) != 1:
                    print("Exception : Input numpy array must be 1 Dimensional for prediction.. Input data shape is",data.shape)
                    return ''
                sdf = self.spark_transformer_pipe.transform(self.spark.createDataFrame(pd.DataFrame({self.raw_text_column:data})))
            elif type(data) is np.matrix: # assumes default axis for raw texts
                print('Predicting on np matrices currently not supported. Please input either a Pandas Dataframe with a string column named "text"  or a String or a list of strings. ' )
                return ''
            elif type(data) is str:  # inefficient, str->pd->spark->pd , we can could first pd
                self.output_datatype='string'
                sdf = self.spark_transformer_pipe.transform(self.spark.createDataFrame(
                    pd.DataFrame({self.raw_text_column: data}, index=[0])))  
            elif type(data) is list:  # inefficient, list->pd->spark->pd , we can could first pd
                self.output_datatype='string_list'
                if all(type(elem) == str for elem in data):
                    sdf = self.spark_transformer_pipe.transform(self.spark.createDataFrame(pd.DataFrame(
                        {self.raw_text_column: pd.Series(data)})))  
                else:
                    print("Exception: Not all elements in input list are of type string.")
            elif type(data) is dict():  # Assumes values should be predicted
                print('Predicting on dictionaries currently not supported. Please input either a Pandas Dataframe with a string column named "text"  or a String or a list of strings. ' )
                return ''
            else : # Modin tests, This could crash if Modin not installed 
                try : 
                    import modin.pandas as mpd
                    if type(data) is mpd.DataFrame  :
                        data = pd.DataFrame(data.to_dict()) # create pandas to support type inference
                        self.output_datatype = 'modin'
    
                    if self.raw_text_column in data.columns:
                        if len(data.columns) > 1 :
                            data = data.where(pd.notnull(data), None) # make  Nans to None, or spark will crash
                            data = data.dropna(axis=1, how='all')
                            stranger_features = list( set(data.columns) - set(self.raw_text_column))
                        sdf = self.spark_transformer_pipe.transform(
                            # self.spark.createDataFrame(data[['text']]), ) # this takes text column as series and makes it DF
                            self.spark.createDataFrame(data ))
                    else :
                        print('Could not find column named "text" in input Pandas Dataframe. Please ensure one column named such exists. Columns in DF are : ', data.columns)

                    if type(data) is mpd.Series  :
                        self.output_datatype='modin_series'
                        data = pd.Series(data.to_dict()) # create pandas to support type inference
                        data = pd.DataFrame(data).dropna(axis=1, how='all')
        
                        if self.raw_text_column in data.columns: sdf = \
                            self.spark_transformer_pipe.transform(
                                self.spark.createDataFrame(data[['text']]), )
                        else: print('Could not find column named "text" in  Pandas Dataframe generated from input  Pandas Series. Please ensure one column named such exists. Columns in DF are : ', data.columns)
                        
                    
                except : 
                    print("If you use Modin, make sure you have installed 'pip install modin[ray]' or 'pip install modin[dask]' backend for Modin ")


            return self.pythonify_spark_dataframe(sdf, self.output_different_levels, keep_stranger_features=keep_stranger_features, stranger_features=stranger_features, output_metadata=metadata)
        except : 
            import sys
            e = sys.exc_info()
            print(e[0])
            print(e[1])

            print("No accepted Data type or usable columns found. Does your Dataframe contain a column named text?")
            print('Stacktrace was',e)
            return None


    def print_info(self,):
        '''
        Print out information about every component currently loaded in the pipe and their configurable parameters
        :return: None
        '''
        
        print('The following parameters are configurable for this NLU pipeline (You can copy paste the examples) :')
        # list of tuples, where first element is component name and second element is list of param tuples, all ready formatted for printing
        all_outputs = []
    
        for i, component_key in enumerate(self.keys()) :
            s=">>> pipe['"+ component_key +"'] has settable params:"
            p_map = self[component_key].extractParamMap()
            
            component_outputs = []
            max_len = 0
            for key in p_map.keys():
                if "outputCol" in key.name or "labelCol" in key.name or "inputCol" in key.name or "labelCol" in key.name : continue
                # print("pipe['"+ component_key +"'].set"+ str( key.name[0].capitalize())+ key.name[1:]+"("+str(p_map[key])+")" + " | Info: " + str(key.doc)+ " currently Configured as : "+str(p_map[key]) )
                # print("Param Info: " + str(key.doc)+ " currently Configured as : "+str(p_map[key]) )
                
                if type(p_map[key]) == str :
                    s1 = "pipe['"+ component_key +"'].set"+ str( key.name[0].capitalize())+ key.name[1:]+"('"+str(p_map[key])+"') "
                else :
                    s1 = "pipe['"+ component_key +"'].set"+ str( key.name[0].capitalize())+ key.name[1:]+"("+str(p_map[key])+") "

                s2 =  " | Info: " + str(key.doc)+ " | Currently set to : "+str(p_map[key])  
                if len(s1) > max_len : max_len = len(s1)
                component_outputs.append((s1,s2))

            all_outputs.append((s,component_outputs))
                
        # make strings aligned
        form = "{:<"+str(max_len) + "}"
        for o in all_outputs : 
            print(o[0]) # component name
            for o_parm in o[1] : 
                if len(o_parm[0]) < max_len : 
                    print(form.format(o_parm[0]) + o_parm[1])
                else :
                    print(o_parm[0] + o_parm[1])





class PipelineQueryVerifier():
    '''
        Pass a list of NLU components to the pipeline (or a NLU pipeline)
        For every component, it checks if all requirements are met.
        It checks and fixes the following issues  for a list of components:
        1. Missing Features / component requirements
        2. Bad order of components (which will cause missing features.
        3. Check Feature naems in the output
        4. Check wether pipeline needs to be fitted
    '''
    @staticmethod
    def has_embeddings_requirement(component):
        '''
        Check for the input component, wether it depends on some embedding. Returns True if yes, otherwise False.

        :param component:  The component to check
        :return: True if the component needs some specifc embedding (i.e.glove, bert, elmo etc..). Otherwise returns False
        '''

        if type(component) == list or type(component) == set :
            for feature in component:
                if 'embed' in feature : return  True
            return False
        else :
            for feature in component.component_info.inputs:
                if 'embed' in feature and feature : return  True
        return False

    @staticmethod
    def has_embeddings_provisions(component):
        '''
        Check for the input component, wether it depends on some embedding. Returns True if yes, otherwise False.
        :param component:  The component to check
        :return: True if the component needs some specifc embedding (i.e.glove, bert, elmo etc..). Otherwise returns False
        '''
        if type(component) == type(list) or type(component) == type(set):
            for feature in component:
                if 'embed' in feature : return  True
            return False
        else :
            for feature in component.component_info.outputs:
                if 'embed' in feature : return  True
        return False

    @staticmethod
    def clean_irrelevant_features(component_list):
        '''
        Remove irrelevant features from a list of component features
        :param component_list: list of features
        :return: list with only relevant feature names
        '''
        #remove irrelevant missing features for pretrained models
        if 'text' in component_list: component_list.remove('text')
        if 'raw_text' in component_list: component_list.remove('raw_text')
        if 'raw_texts' in component_list: component_list.remove('raw_texts')
        if 'label' in component_list: component_list.remove('label')
        if 'sentiment_label' in component_list: component_list.remove('sentiment_label')
        return component_list

    @staticmethod
    def get_missing_required_features (pipe:NLUPipeline)  :
        '''
        Takes in a NLUPipeline and returns a list of missing  feature types types which would cause the pipeline to crash if not added
        If it is some kind of model that uses embeddings, it will check the metadata for that model and return a string with moelName@spark_nlp_embedding_reference format
        '''
        logger.info('Resolving missing components')
        pipe_requirements = [['sentence','token']] #default requirements so we can support all output levels. minimal extra comoputation effort. If we add CHUNK here, we will aalwayshave POS default
        pipe_provided_features = []
        # pipe_types = [] # list of string identifiers
        for component in pipe.pipe_components:

            # 1. Get all feature provisions from the pipeline
            logger.info("Getting Missing Feature for component =%s",component.component_info.name)
            if not component.component_info.inputs == component.component_info.outputs:
                pipe_provided_features.append(component.component_info.outputs)  # edge case for components that provide token and require token and similar cases.

            # 2. get all feature requirements for pipeline
            if PipelineQueryVerifier.has_embeddings_requirement(component) :
                # special case for models with embedding requirements. we will modify the output string which then will be resolved by the default component resolver (which will get the correct embedding )
                if component.component_info.type == 'chunk_embeddings':
                    #there is no ref for Chunk embeddings, so we have a special case here and need to define a default value that will always be used for chunkers
                    sparknlp_embeddings_requirement_reference ='glove'
                else :sparknlp_embeddings_requirement_reference = component.model.extractParamMap()[component.model.getParam('storageRef')]
                inputs_with_sparknlp_reference = []
                for feature in component.component_info.inputs:
                    if 'embed' in feature : inputs_with_sparknlp_reference.append(feature + '@' + sparknlp_embeddings_requirement_reference)
                    else : inputs_with_sparknlp_reference.append(feature)
                pipe_requirements.append(inputs_with_sparknlp_reference)
            else : pipe_requirements.append(component.component_info.inputs)

        # 3. Some components have "word_embeddings" als input configured, but no actual wordembedding has "word_embedding" as output configured. 
        # Thus we must check in a different way here first if embeddings are provided and if they are there we have to remove them form the requirements list

        

        # 4. get missing requirements, by substracting provided from requirements
        # Flatten lists, make them to sets and get missing components by substracting them.
        flat_requirements = set(item for sublist in pipe_requirements for item in sublist)
        flat_provisions = set(item for sublist in pipe_provided_features for item in sublist)
        #rmv spark identifier from provision
        flat_requirements_no_ref = set(item.split('@')[0] if '@' in item else item for item in flat_requirements)

        #see what is missing, with identifier removed
        missing_components = PipelineQueryVerifier.clean_irrelevant_features(flat_requirements_no_ref - flat_provisions)
        logger.info("Required columns no ref flat =%s",flat_requirements_no_ref)
        logger.info("Provided columns flat =%s",flat_provisions)
        logger.info("Missing columns no ref flat =%s",missing_components)
        # since embeds are missing, we add embed with reference back
        if PipelineQueryVerifier.has_embeddings_requirement(missing_components):
            missing_components = PipelineQueryVerifier.clean_irrelevant_features(flat_requirements - flat_provisions)

        if len(missing_components) == 0 :
            logger.info('No more components missing!')
            return []
        else:
            # we must recaclulate the difference, because we reoved the spark nlp reference previously for our set operation. Since it was not 0, we ad the Spark NLP rererence back
            logger.info('Components missing=%s', str(missing_components))
            return missing_components

    @staticmethod
    def check_and_fix_nlu_pipeline(pipe:NLUPipeline):
        # main entry point for Model stacking withouth pretrained pipelines
        # requirements and provided features will be lists of lists
        all_features_provided = False
        while all_features_provided == False :
            # After new components have been added, we must loop agan and check for the new components if requriements are met
            # OR we implement a function caled "Add components with requirements". That one needs to know though, which requirements are already met ...

            # Find missing components
            missing_components = PipelineQueryVerifier.get_missing_required_features(pipe)
            if len(missing_components) == 0: break  # Now all features are provided

            components_to_add = []
            # Create missing components
            for missing_component in missing_components:
                components_to_add.append(nlu.get_default_component_of_type(missing_component))
            logger.info('Resolved for missing components the following NLU components : %s', str(components_to_add))


            # Add missing components and validate order of components is correct
            for new_component in components_to_add:
                pipe.add(new_component)
                logger.info('adding %s=', new_component.component_info.name)


        logger.info('Fixing column names')
        #  Validate naming of output columns is correct and no error will be thrown in spark
        pipe = PipelineQueryVerifier.check_and_fix_component_output_column_names(pipe)

        # 3.  fix order
        logger.info('Optimizing pipe component order')

        pipe = PipelineQueryVerifier.check_and_fix_component_order(pipe)
        # 5. Todo Download all file depenencies like train files or  dictionaries
        logger.info('Done with pipe optimizing')

        return pipe

    @staticmethod
    def check_and_fix_component_output_column_names(pipe: NLUPipeline):
        '''
        This function verifies that every input and output column name of a component is satisfied.
        If some output names are missing, it will be added by this methods.
        Usually classifiers need to change their input column name, so that it matches one of the previous embeddings because they have dynamic output names
        This function peforms the following steps :
        1. For each component we veryify that all input column names are satisfied  by checking all other components output names
        2. When a input column is missing we do the following :
        2.1 Figure out the type of the missing input column. The name of the missing column should be equal to the type
        2.2 Check if there is already a component in the pipe, which provides this input (It should)
        2.3. When the providing component is found, update its output name, or update the original coponents input name
        :return: NLU pipeline where the output and input column names of the models have been adjusted to each other
        '''

        all_names_provided = False

        for component_to_check in pipe.pipe_components:
            all_names_provided_for_component = False
            input_columns = set(component_to_check.component_info.spark_input_column_names)
            logger.info('Checking for component %s wether input %s is satisfied by another component in the pipe ',component_to_check.component_info.name, input_columns)
            for other_component in pipe.pipe_components:
                if component_to_check.component_info.name == other_component.component_info.name: continue
                output_columns = set(other_component.component_info.spark_output_column_names)
                input_columns -= output_columns  # set substraction

            input_columns = PipelineQueryVerifier.clean_irrelevant_features(input_columns)

            if len(input_columns) != 0:  # fix missing column name
                for missing_column in input_columns:
                    for other_component in pipe.pipe_components:
                        if component_to_check.component_info.name == other_component.component_info.name: continue
                        if other_component.component_info.type == missing_column:
                            # resolve which setter to use ...
                            # We update the output name for the component which provides our feature
                            other_component.component_info.spark_output_column_names = [missing_column]
                            logger.info('Setting output columns for component %s to %s ', other_component.component_info.name, missing_column)
                            other_component.model.setOutputCol(missing_column)

        return pipe


    @staticmethod
    def check_and_fix_component_order_old(pipe: NLUPipeline):
        '''
        This method takes care that the order of components is the correct in such a way,
        that the pipeline can be iteratively processed by spark NLP.
        '''

        # Get all components by order. Inefficent Double Loop. List size will be very small, so it will not be that dramatic.,
        components_by_priority = {}  # Keys from level 0 to  max_level. Each value is a list of components of that priority level in the current pipe
        for priority_level in range(nlu.PipelineQueryVerifier.amount_of_component_priority_levels + 1):
            if not priority_level in components_by_priority.keys(): components_by_priority[priority_level] = []
            for component in pipe.pipe_components:
                component_priority_level = component.component_info.pipe_priorioty_slot  # nlu.PipelineQueryVerifier.component_priority_level_resolver(component.component_info.type, component.component_info.type)
                if int(component_priority_level) == int(
                        priority_level):  # semi bug under the hood, check typing in json files and info class
                    components_by_priority[priority_level].append(component)

        correct_order_component_pipeline = []
        for priority_level in range(nlu.PipelineQueryVerifier.amount_of_component_priority_levels + 1):
            correct_order_component_pipeline.append(components_by_priority[priority_level])

        # flatten nested list
        correct_order_component_pipeline = [e for esub in correct_order_component_pipeline for e in esub]

        pipe.pipe_components = correct_order_component_pipeline
        return pipe

    @staticmethod
    def check_and_fix_component_order(pipe: NLUPipeline):
        '''
        This method takes care that the order of components is the correct in such a way,
        that the pipeline can be iteratively processed by spark NLP.
        '''
        logger.info("Starting to optimize component order ")
        correct_order_component_pipeline = []
        all_components_orderd = False
        all_components = pipe.pipe_components

        provided_features = []
        while all_components_orderd == False:
            for component in all_components:
                logger.info("Optimizing order for component %s", component.component_info.name)

                if component.component_info.name == 'document_assembler':
                    provided_features.append('document')
                    correct_order_component_pipeline.append(component)
                    all_components.remove(component)

                input_columns = PipelineQueryVerifier.clean_irrelevant_features(component.component_info.inputs)
                if set(input_columns).issubset(provided_features):  #  component not in correct_order_component_pipeline:
                    correct_order_component_pipeline.append(component)
                    if component in all_components : all_components.remove(component) # ??? dirty bugged fixed
                    for feature in component.component_info.outputs: provided_features.append(feature)
            if len(all_components) == 0: all_components_orderd = True

        pipe.pipe_components = correct_order_component_pipeline

        return pipe

