import logging
logger = logging.getLogger('nlu')

from nlu.pipe.extractors.extraction_resolver_OS import OS_anno2config
from nlu.pipe.extractors.extractor_methods.base_extractor_methods import *
from nlu.pipe.pipe_logic import PipeUtils

import nlu
import nlu.pipe.pipe_components
import sparknlp
from  typing import List, Dict

from sparknlp.base import *
from sparknlp.base import LightPipeline
from sparknlp.annotator import *
import pyspark
import pandas as pd
import numpy as np
from pyspark.sql.types import StructType,StructField, StringType, IntegerType
from nlu.pipe.component_resolution import extract_classifier_metadata_from_nlu_ref
from nlu.pipe.utils.storage_ref_utils import StorageRefUtils
from nlu.pipe.utils.component_utils import ComponentUtils
from nlu.pipe.utils.output_level_resolution_utils import OutputLevelUtils
from nlu.pipe.utils.data_conversion_utils import DataConversionUtils
from nlu.environment.env_utils import is_running_in_databricks
from nlu.pipe.col_substitution.col_name_substitution_utils import ColSubstitutionUtils
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
        self.components = []  # orderd list of nlu_component objects
        self.output_datatype = 'pandas'  # What data type should be returned after predict either spark, pandas, modin, numpy, string or array
        self.lang = 'en'
        self.spark_transformer_pipe = None
        self.spark_estimator_pipe_pipe = None
        self.has_licensed_components = False
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
    def configure_outputs(self, component, nlu_reference, name_to_add=''):
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
        model_meta = extract_classifier_metadata_from_nlu_ref(nlu_reference)
        can_use_name = False
        new_output_name = model_meta[0]
        i = 0
        while can_use_name == False:
            can_use_name = True
            for c in self.components:
                if new_output_name in c.info.spark_input_column_names + c.info.spark_output_column_names and c.info.name != component.info.name:
                    can_use_name = False
        if can_use_name == False:
            new_output_name = new_output_name + '_' + str(i)
            i += 1
        # classifiers always have just 1 output col
        logger.info(f"Configured output columns name to {new_output_name} for classifier in {nlu_reference}")
        component.model.setOutputCol(new_output_name)
        component.info.spark_output_column_names = [new_output_name]
    def add(self, component, nlu_reference="default_name", pretrained_pipe_component=False, name_to_add=''):
        '''

        :param component:
        :param nlu_reference: NLU references, passed for components that are used specified and not automatically generate by NLU
        :return:
        '''
        if hasattr(component.info,'nlu_ref'): nlu_reference = component.info.nlu_ref
        # self.nlu_reference = component.info.nlu_ref if component.info
        self.components.append(component)
        # ensure that input/output cols are properly set
        # Spark NLP model reference shortcut

        name = component.info.name.replace(' ', '').replace('train.', '')

        if StorageRefUtils.has_storage_ref(component):
            name = name +'@' + StorageRefUtils.extract_storage_ref(component)
        logger.info(f"Adding {name} to internal pipe")

        # Configure output column names of classifiers from category to something more meaningful
        if self.isInstanceOfNlpClassifer(component.model): self.configure_outputs(component, nlu_reference)

        if name_to_add == '':
            # Add Component as self.index and in attributes
            if 'embed' in component.info.type and nlu_reference not in self.keys() and not pretrained_pipe_component:
                self[name] = component.model
            elif name not in self.keys():
                if not hasattr(component.info,'nlu_ref'):  component.info.nlu_ref  = nlu_reference
                self[name] = component.model
            else:
                nlu_identifier = ComponentUtils.get_nlu_ref_identifier(component)
                component.info.nlu_ref = nlu_reference
                self[component.info.name +"@"+ nlu_identifier] = component.model
        else :self[name_to_add] = component.model
class NLUPipeline(BasePipe):
    def __init__(self):
        super().__init__()
        """ Initializes a pretrained pipeline         """
        self.spark = sparknlp.start()
        self.provider = 'sparknlp'
        self.pipe_ready = False  # ready when we have created a spark df
        # The NLU pipeline uses  types of Spark NLP annotators to identify how to handle different columns
    def get_sample_spark_dataframe(self):
        data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami']}
        text_df = pd.DataFrame(data)
        return sparknlp.start().createDataFrame(data=text_df)
    def verify_all_labels_exist(self,dataset):
        return 'y'  in dataset.columns or 'label' in dataset.columns or 'labels' in dataset.columns
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
        for component in self.components:stages.append(component.model)
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
            df = self.spark.createDataFrame(data=dataset).withColumn('y',F.split('y',label_seperator))
            # df = self.spark.createDataFrame(data=dataset, schema=schema).withColumn('y',F.split('y',label_seperator))
            # df = self.spark.createDataFrame(dataset)
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(df)

        elif isinstance(dataset,pd.DataFrame):
            if not self.verify_all_labels_exist(dataset) : raise ValueError(f"Could not detect label in provided columns={dataset.columns}\nMake sure a column named label, labels or y exists in your dataset.")
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(DataConversionUtils.pdf_to_sdf(dataset,self.spark)[0])

        elif isinstance(dataset,pyspark.sql.DataFrame) :
            if not self.verify_all_labels_exist(dataset) : raise ValueError(f"Could not detect label in provided columns={dataset.columns}\nMake sure a column named label, labels or y exists in your dataset.")
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(DataConversionUtils.sdf_to_sdf(dataset,self.spark))

        else :
            # fit on empty dataframe since no data provided
            logger.info('Fitting on empty Dataframe, could not infer correct training method. This is intended for non-trainable pipelines.')
            self.spark_transformer_pipe = self.spark_estimator_pipe.fit(self.get_sample_spark_dataframe())


        return self
    def get_annotator_extraction_configs(self,full_meta,c_level_mapping,positions):

        """Search first OC namespace and if not found the HC Namespace for each Annotator Class in pipeline and get corrosponding config
        Returns a dictionary of methods, where keys are column names values are methods  that are applied to extract and represent the data in these
        these columns in a more pythonic and panda-esque way
        """
        # tood doc level annos and same level annos can be popped always.
        anno_2_ex_config = {}
        for c in self.components:
            if type(c.model) in OS_anno2config.keys():
                if OS_anno2config[type(c.model)]['default'] == '' or full_meta:
                    if not full_meta : logger.info(f'could not find default configs, using full default for model ={c.model}')
                    anno_2_ex_config[c.info.spark_output_column_names[0]] = OS_anno2config[type(c.model)]['default_full'](output_col_prefix=c.info.outputs[0])
                else :
                    anno_2_ex_config[c.info.spark_output_column_names[0]] = OS_anno2config[type(c.model)]['default'](output_col_prefix=c.info.outputs[0])
            else:
                from nlu.pipe.extractors.extraction_resolver_HC import HC_anno2config
                if HC_anno2config[type(c.model)]['default'] == '' or full_meta :
                    if not full_meta : logger.info(f'could not find default configs in hc resolver space, using full default for model ={c.model}')
                    anno_2_ex_config[c.info.spark_output_column_names[0]] = HC_anno2config[type(c.model)]['default_full'](output_col_prefix=c.info.outputs[0])
                else :
                    anno_2_ex_config[c.info.spark_output_column_names[0]] = HC_anno2config[type(c.model)]['default'](output_col_prefix=c.info.outputs[0])

            if c_level_mapping[c] == 'document' and  not anno_2_ex_config[c.info.spark_output_column_names[0]].pop_never :
                anno_2_ex_config[c.info.spark_output_column_names[0]].pop_meta_list = True
                anno_2_ex_config[c.info.spark_output_column_names[0]].pop_result_list = True
            # if c_level_mapping[c] == 'relation' and  not anno_2_ex_config[c.info.spark_output_column_names[0]].pop_never :
            if positions :
                anno_2_ex_config[c.info.spark_output_column_names[0]].get_positions = True

        return anno_2_ex_config
    def unpack_and_apply_extractors(self,sdf:pyspark.sql.DataFrame, keep_stranger_features=True, stranger_features=[],anno_2_ex_config={})-> pd.DataFrame:
        """1. Unpack SDF to PDF with Spark NLP Annotator Dictionaries
           2. Get the extractor configs for the corrosponding Annotator classes
           3. Apply The extractor configs with the extractor methods to each column and merge back with zip/explode"""
        unpack_df = sdf.toPandas().applymap(extract_pyspark_rows)
        return apply_extractors_and_merge(unpack_df,anno_2_ex_config, keep_stranger_features,stranger_features)
    def pythonify_spark_dataframe(self, processed,
                                  keep_stranger_features=True,
                                  stranger_features=[],
                                  drop_irrelevant_cols=True,
                                  output_metadata=False,
                                  positions = False
                                  ):
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

        if self.output_level == '' : self.output_level = OutputLevelUtils.infer_output_level(self)
        c_level_mapping =OutputLevelUtils.get_output_level_mapping_by_component(self)

        anno_2_ex_config                                                          = self.get_annotator_extraction_configs(output_metadata,c_level_mapping,positions)
        pretty_df = self.unpack_and_apply_extractors(processed, keep_stranger_features, stranger_features,anno_2_ex_config)
        col2output_level,same_output_level,not_same_output_level                  = OutputLevelUtils.get_output_level_mappings(self,pretty_df,anno_2_ex_config)
        logger.info(f"Extracting for same_level_cols = {same_output_level}\nand different_output_level_cols = {not_same_output_level}")
        pretty_df = zip_and_explode(pretty_df, same_output_level, not_same_output_level, self.output_level)
        pretty_df = self.convert_embeddings_to_np(pretty_df)
        pretty_df = ColSubstitutionUtils.substitute_col_names(pretty_df,anno_2_ex_config,self, stranger_features)
        if  drop_irrelevant_cols :  pretty_df = pretty_df[self.drop_irrelevant_cols(list(pretty_df.columns))]
        return pretty_df.loc[:,~pretty_df.columns.duplicated()]

    def viz(self, text_to_viz:str, viz_type='', labels_to_viz=None,viz_colors={},):
        """Visualize predictions of a Pipeline, using Spark-NLP-Display
        text_to_viz : String to viz
        labels_to_viz : Defines a subset of NER labels to viz i.e. ['PER'] , by default=[] which will display all labels. Applicable only for NER viz
        viz_type    :  Viz type, one of [ner,dep,resolution,relation,assert]
        viz_colors  : Applicable for [ner, resolution, assert ] key = label, value=hex color, i.e. viz_colors={'TREATMENT':'#008080', 'problem':'#800080'}

        """

        """        # TODO DATABRICKS HANDLING!
                configure return_html=True on visualizers and then Pipe HTML into displayHTML(vis_html)
        """
        from nlu.environment.env_utils import install_and_import_package
        install_and_import_package('spark-nlp-display',import_name='sparknlp_display')
        if self.spark_transformer_pipe is None : self.fit()
        is_databricks_env = is_running_in_databricks()
        self.configure_light_pipe_usage(1)
        from nlu.pipe.viz.vis_utils import VizUtils

        if viz_type == '' : viz_type  = VizUtils.infer_viz_type(self)
        anno_res = self.spark_transformer_pipe.fullAnnotate(text_to_viz)[0]
        if self.has_licensed_components==False :
            VizUtils.viz_OS(anno_res, self, viz_type,viz_colors,labels_to_viz,is_databricks_env)
        else :
            VizUtils.viz_HC(anno_res, self, viz_type,viz_colors,labels_to_viz,is_databricks_env)

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
            return df # todo
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
            if 'document_results' in cols: cols.remove('document_results')
            if 'chunk_results' in cols: cols.remove('chunk_results')
            if 'sentence_results' in cols: cols.remove('sentence_results')
        if self.output_level == 'sentence':
            if 'token_results' in cols: cols.remove('token_results')
            if 'chunk_results' in cols: cols.remove('chunk_results')
            if 'document_results' in cols: cols.remove('document_results')
        if self.output_level == 'chunk':
            # if 'document_results' in cols: cols.remove('document_results')
            if 'token_results' in cols: cols.remove('token_results')
            if 'sentence_results' in cols: cols.remove('sentence_results')
        if self.output_level == 'document':
            if 'token_results' in cols: cols.remove('token_results')
            if 'chunk_results' in cols: cols.remove('chunk_results')
            if 'sentence_results' in cols: cols.remove('sentence_results')
        if self.output_level == 'relation':
            if 'token_results' in cols: cols.remove('token_results')
            if 'chunk_results' in cols: cols.remove('chunk_results')
            if 'sentence_results' in cols: cols.remove('sentence_results')
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

        for c in self.components:
            if 'sentence' in c.info.spark_output_column_names : return True
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
        f.write(self.nlu_ref)
        f.close()
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
            if isinstance(self.spark_transformer_pipe, LightPipeline):
                self.spark_transformer_pipe.pipeline_model.save(path)
            else: self.spark_transformer_pipe.save(path)
        else:
            if component in self.keys():
                self[component].save(path)
        print(f'Stored model in {path}')

    def predict(self, data, output_level='', positions=False, keep_stranger_features=True, metadata=False,
                multithread=True, drop_irrelevant_cols=True, return_spark_df = False):
        '''
        Annotates a Pandas Dataframe/Pandas Series/Numpy Array/Spark DataFrame/Python List strings /Python String

        :param data: Data to predict on
        :param output_level: output level, either document/sentence/chunk/token
        :param positions: wether to output indexes that map predictions back to position in origin string
        :param keep_stranger_features: wether to keep columns in the dataframe that are not generated by pandas. I.e. when you s a dataframe with 10 columns and only one of them is named text, the returned dataframe will only contain the text column when set to false
        :param metadata: wether to keep additonal metadata in final df or not like confidiences of every possible class for preidctions.
        :param multithread: Whether to use multithreading based lightpipeline. In some cases, this may cause errors.
        :param drop_irellevant_cols: Wether to drop cols of different output levels, i.e. when predicting token level and dro_irrelevant_cols = True then chunk, sentence and Doc will be dropped
        :param return_spark_df: Prediction results will be returned right after transforming with the Spark NLP pipeline
        :return:
        '''
        if output_level != '': self.output_level = output_level
        if output_level == 'sentence' or output_level == 'document':self.components = PipeUtils.configure_component_output_levels(self)
        if output_level in ['token', 'chunk','relation']:self.components = PipeUtils.configure_component_output_levels(self,'document')

        try:
            #1. Convert data to Spark DF
            data, stranger_features, output_datatype = DataConversionUtils.to_spark_df(data, self.spark, self.raw_text_column)

            #2. configure Lightpipline usage
            self.configure_light_pipe_usage(data.count(), multithread)
            # 3. Fit
            if not self.is_fitted :
                if self.has_trainable_components:self.fit(data)
                else : self.fit()

            #4. Apply Spark Pipeline
            data = self.spark_transformer_pipe.transform(data)
            #5. Convert resulting spark DF into nicer format and by default into pandas.
            if return_spark_df : return data  # Returns RAW  Spark Dataframe result of pipe prediction
            return self.pythonify_spark_dataframe(data,
                                                  keep_stranger_features=keep_stranger_features,
                                                  stranger_features=stranger_features,
                                                  output_metadata=metadata,
                                                  drop_irrelevant_cols=drop_irrelevant_cols,
                                                  positions=positions
                                                  )


        except Exception as err :

            import sys
            if multithread == True:
                logger.warning("Multithreaded mode failed. trying to predict again with non multithreaded mode ")
                return self.predict(data, output_level=output_level, positions=positions,keep_stranger_features=keep_stranger_features, metadata=metadata, multithread=False)
            else: self.print_exception_err(err)


    def print_info(self,minimal=True):
        '''
        Print out information about every component currently loaded in the pipe and their configurable parameters.
        If minimal is false, all Spark NLP Model parameters will be printed, including output/label/input cols and other attributes a NLU user should not touch. Useful for debugging.
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
                if minimal:
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





    def print_exception_err(self,err):
        '''Print information about exception during converting or transforming dataframe'''
        import sys
        logger.exception('Exception occured')
        e = sys.exc_info()
        print("No accepted Data type or usable columns found or applying the NLU models failed. ")
        print("Make sure that the first column you pass to .predict() is the one that nlu should predict on OR rename the column you want to predict on to 'text'  ")
        print("On try to reset restart Jupyter session and run the setup script again, you might have used too much memory")
        print('Full Stacktrace was', e)
        print('Additional info:')
        exc_type, exc_obj, exc_tb = sys.exc_info()
        import os
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        err = sys.exc_info()[1]
        print(str(err))
        print('Stuck? Contact us on Slack! https://join.slack.com/t/spark-nlp/shared_invite/zt-lutct9gm-kuUazcyFKhuGY3_0AMkxqA')

