
from nlu import *

from dataclasses import dataclass
import glob
import os
import json
import sys
import logging
COMPONENT_INFO_FILE_NAME = 'component_infos.json'
logger = logging.getLogger('nlu')

class AllComponentsInfo:
    def __init__(self):
        ''' Initialize every NLU component info object and provide access to them'''

        self.all_components = {}
        self.classifiers = {}
        self.embeddings = {}
        self.normalizers  = {}
        self.pretrained_pipelines = {}
        self.selectors = {}
        self.spell_checkers = {}
        self.stemmers = {}
        self.tokenizers = {}
        self.utils = {}
        self.all_pretrained_pipe_languages = ['en', 'nl','fr','de','it','no','pl','pt','ru','es','xx',]
        self.all_pretrained_model_languages = ['da','fr','de','it','nb','no','nn','pl','pt','ru','es','af','ar','hy','eu','bn','br','bg','ca','cs','eo','fi','gl','el','ha','he','hi','hu','id','ga','ja','la','lv','mr','fa','ro','sk','sl','so','st','sw','sv','th','tr','uk','yo','zu','xx',]
        self.all_languages = set(self.all_pretrained_pipe_languages).union(set(self.all_pretrained_model_languages))

        # this maps a requsted token to a class
        self.all_component_types = ['tokenize','pos','ner','embed','classify','sentiment','emotion','spell','dependency',
                                    'labled_dependency','lemma', 'norm', 'select', 'pretrained_pipe','util']


        all_component_paths_regex = nlu.nlu_package_location + 'components/*/*/'
        all_component_paths = glob.glob(all_component_paths_regex)

        for path in all_component_paths :
            if '__py' in path : continue
            logger.info('Loading info dict @ path'+ path)
            component = ComponentInfo.from_directory(path)
            self.all_components[component.name] = component
            if component.type == 'classifier' : self.classifiers[component.name] = component
            if component.type == 'embedding' : self.embeddings[component.name] = component
            if component.type == 'normalizer' : self.normalizers[component.name] = component
            if component.type == 'pretrained_pipeline' : self.pretrained_pipelines[component.name] = component
            if component.type == 'selector' : self.selectors[component.name] = component
            if component.type == 'spell_checker' : self.spell_checkers[component.name] = component
            if component.type == 'stemmer' : self.stemmers[component.name] = component
            if component.type == 'tokenizer' : self.tokenizers[component.name] = component
            if component.type == 'util' : self.utils[component.name] = component
            # todo labled dependecy, unlabled dep, CHunk, Date, categoryu, sentence detector



    def list_all_components(self):
        print("--------------Avaiable Components in NLU :--------------")
        for name in self.all_components.keys(): print(name)

    def DEBUG_list_all_components(self):
        print("--------------Avaiable Components in NLU :--------------")
        for name in self.all_components.keys():
            print(name, " INPUT_F : ", self.all_components[name].inputs, " OUTPUT_F ", self.all_components[name].inputs," INPUT_N ", self.all_components[name].spark_output_column_names, "OUTPUT_N ", self.all_components[name].spark_output_column_names)

    def get_component_info_by_name(self,name): return self.all_components[name]
    def list_all_components_of_type(self,component_type='embeddings'): pass
    @staticmethod
    def list_all_components_of_language(component_lang='ger'): pass
    @staticmethod
    def list_all_components_of_languageand_type(component_lang='ger', component_type='embeddings'): pass
    @staticmethod
    def get_default_component_of_type():pass
    @staticmethod
    def list_avaiable_output_types():pass
    @staticmethod
    def get_all_component_info_obj():pass


@dataclass
class ComponentInfo:
    name: str
    description: str  # general annotator/model/component/pipeline info
    outputs: list  # this is which columns/output types this component is providing
    inputs: list  # this tells us which columns/input types the component is depending on
    type: str  # this tells us which kind of component this is
    file_dependencies: dict  # Dict, where keys are file name identifiers and value is a dict of attributes (Where to download file, whats the size, etc..) (( MAYBE EMBELISH IN A CLASS?)
    pipe_priorioty_slot: int  # What is the position at which this component should occur? 0 for first like Document Assembler, 1 For Tokenizer, etc.. See pipe group ranked priorties
    output_level : str # document, sentence, token, chunk, input_dependent or model_dependent
    spark_input_column_names: list  # default expected name for input columns when forking with spark nlp annotators on spark DFs
    spark_output_column_names: list  # default expected name for output columns when forking with spark nlp annotators on spark DFs
    critical_parameters: list  # List of parameters that MUST be specified by the user to run this component, like [bla,bla,bla]
    tags: list  # list of string tags
    categories: list  # list of categories this component belongs to
    metrics: list  # list of string identifiers for metrics, whiech this component can solve
    made_from_components: list  # list of component identifiers which are used to build a pr etrained pipeline. Just the name of the component in a list or can be left empty for non-pre-built pipelines

    component_options : list # list of str for different versions of component. Only applicable to single language components (or multi lang) or when all languages have exactly the same parameters

    languages : list # Languages in which this component is available
    component_parameters_info : dict # Hyperparameters of the component. Key parameter name, value is param info
    component_language_options : dict # Dict in which every key is a language and every value is a dictionary of valid options for selected the model. (This is requried, since each language for a pretraiend pipeline does not have the same parameter options)

    meta_data : dict # additonal data about the component as a dict

    provider : str  # Who provides the implementation of this annotator, Spark-NLP for base. Would be
    version : str # Version of the model
    support_version : str # First version NLU in which this model is supported
    documentation_urls : list # List of URLs relevant to this component
    source_urls : list # List of URLs relevant to this component (Source code etc..)
    reference_urls : list # List of URLs relevant to this component (Additional Docs or other library links(
    citations : list # List of paper citations
    license : str # open source or private
    docs_author : str # Who is responsible for this NLU component docs
    component_author : str # Who built the model?
    nlu_author : str # Who is responsible for this NLU component and built a wrapper around this so its available in NLU?

    computation_context : str # Will this component do its computation in Spark land (like all of Spark NLP annotators do) or does it require some other computation engine or library like Tensorflow, Numpy, HuggingFace, etc..
    output_context : str # Will this components final result


    @classmethod
    def from_directory(cls, component_info_dir):
        """Create ComponentInfo  class from the component_infos.json which is provided for every component
        @param component_info_dir:
            dataset_info_dir: `str` The directory containing the metadata file. This
            should be the root directory of a specific dataset version.
        """
        if not component_info_dir:
            raise ValueError("Calling DatasetInfo.from_directory() with undefined dataset_info_dir.")

        with open(os.path.join(component_info_dir, COMPONENT_INFO_FILE_NAME), "r") as f:
            dataset_info_dict = json.load(f)

        try:
            return cls(**dataset_info_dict)# dataset_info_dict
        except :
            print (" Exception Occured! For Path",component_info_dir , " Json file most likely has missing  features. Todo nicer output error info", sys.exc_info()[0])
            raise

    @classmethod
    def preprocess_path_string(cls): pass
    # removes all double / and makes class name lowercase
