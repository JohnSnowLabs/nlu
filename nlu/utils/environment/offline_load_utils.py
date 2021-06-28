import os,sys, json

import nlu
from nlu.pipe.pipe_components import SparkNLUComponent
# from nlu.pipe.pipe_logic import PipelineQueryVerifier
from nlu.pipe.pipeline import  *
from nlu.pipe.pipe_components import SparkNLUComponent
from pyspark.ml import PipelineModel

from sparknlp.annotator import *

def is_pipe(model_path):
    """Check wether there is a pyspark pipe stored in path"""
    return 'stages' in os.listdir(model_path)
def is_model(model_path):
    """Check wether there is a model stored in path"""
    return'metadata' in os.listdir(model_path)

def get_model_class(model_path):
    """Extract class from a model saved in model_path"""
    with open(model_path+'/stages/part-00000') as json_file:
        java_class = json.load(json_file)['class']
        pyth_class = java_class.split('.')[-1]
    return java_class,pyth_class

def verify_and_create_model(model_path:str, nlu_referenced_requirements = [], nlp_referenced_requirements = [], test_data='Hello world from John Snow Labs '):
    """
     Build model with requirements
     Figures out class name by checking metadata json file
     assumes metadatra is always called part-00000
    """
    with open(model_path+'/metadata/'+'part-00000') as json_f:
        class_name = json.load(json_f)['class'].split('.')[-1]
        # The last element in the Class name can be used to just load the model from disk!
        # Just call eval on it, which will give you the actual Python class reference which should have a .load() method
        try :
            m = eval(class_name).load(model_path)
        except:
            from nlu.utils.environment.offline_load_utils_licensed import verify_model_licensed
            m = verify_model_licensed(class_name, model_path)



    component_type,nlu_anno_class, = resolve_annotator_class_to_nlu_component_info(class_name)
    # Wrap model with NLU Custom Model class so the NLU pipeline Logic knows what to do with it
    c = CustomModel(annotator_class=nlu_anno_class, component_type=component_type, model=m)
    if nlu.hard_offline_checks:
        storage_ref =  m.getStorageRef() if hasattr(m,'getStorageRef')  else -1
        feeds_from_embeddings = False
        for inp in c.info.inputs() :
            if 'embed' in inp : feeds_from_embeddings = True
        if type(m) not in [SentenceEmbeddings,ChunkEmbeddings]  and storage_ref != -1 and feeds_from_embeddings:
            # Check storage ref is matching either some NLU or NLP ref.
            if not check_if_storage_ref_exists(storage_ref) :
                pass
            raise ValueError("When loading a model from disk with embeddings in it's inputs or you must set the Storage Reference attribute on the model."
                             "Additionaly, the configured Storage Reference must either match to a NLU-Model-Reference or a Spark-NLP-Model-Reference."
                             "To view all references visit the modelshub or view https://github.com/JohnSnowLabs/nlu/blob/master/nlu/spellbook.py "
                             "You can set a storage reference via calling model.setStorageRef(storage_reference)"
                             "To disable this check, run nlu.disable_hard_offline_checks()")

    return c
def resolve_annotator_class_to_nlu_component_info(anno_class ='LemmatizerModel'):
    """
    SparkNLUComponent.__init__(self, annotator_class, component_type)
    RECURISIVELY SEARCH through NLU COMPONENTS source code for each class for a given <CLASSS>
    Find the file, which called <CLASS>.pretrained or just <CLASS> !!!
    In the folder containing that found file there will be the component_json info we need!]\
    """
    import nlu
    p = nlu.nlu_package_location+'components/'
    # Check if class has pretrained. If not, dont add .

    import os
    for dirpath, dirs, files in os.walk(p):
        # search for folder that has the component info for that anno_class
        for filename in files:
            if '.py' not in filename or '.pyc' in filename : continue
            fname = os.path.join(dirpath,filename)
            # check annotator name and also getter is in file. This should only be true for the correct annotator creator files
            if (test_check_if_string_in_file(fname, anno_class) and test_check_if_string_in_file(fname, 'def get_.*:', True)) and\
                    (test_check_if_string_in_file(fname, 'return *'+ anno_class, True)):

                parts = dirpath.split('/')
                component_type = parts[-2]
                nlu_anno_class = parts[-1]
                component_type = component_type[:-1]
                return component_type,nlu_anno_class,
    print("COULD NOT FIND COMPONENT INFO FOR ANNO_CLASS", anno_class)
    return False

def test_check_if_string_in_file(file_name, string_to_search, regex=False):
    """ Check if any line in the file contains given string """
    # Open the file in read only mode
    # print('reading ', file_name)
    import re

    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            if regex :
                    if len(re.findall(string_to_search,line)) > 0 : return True
            else :
                if string_to_search in line: return True
    return False


def NLP_ref_to_NLU_ref(nlp_ref,lang) :
    """Resolve a Spark NLP reference to a NLU reference"""
    nlu_namespaces_to_check = [nlu.Spellbook.pretrained_pipe_references, nlu.Spellbook.pretrained_models_references]
    for dict_ in nlu_namespaces_to_check:
        if lang in dict_.keys():
            for reference in dict_[lang]:
                if dict_[lang][reference] == nlp_ref:
                    return reference


class CustomModel(SparkNLUComponent):
    """ Builds a NLU Components with component info"""
    def __init__(self, annotator_class='sentiment_dl',  component_type='classifier', model = None):
        self.model = model
        SparkNLUComponent.__init__(self, annotator_class, component_type)
        # Make sure input/output cols match up with NLU defaults
        if len(self.info.spark_input_column_names) == 1 :
            model.setInputCols(self.info.spark_input_column_names[0])
        else :
            model.setInputCols(self.info.spark_input_column_names)

        if len(self.info.spark_output_column_names) == 1 :
            model.setOutputCol(self.info.spark_output_column_names[0])
        else :
            model.setOutputCol(self.info.spark_output_column_names)



    # pipe = NLUPipeline()
    # pipe.add(c)
    #
    # # get requirements
    # if len(nlu_referenced_requirements)==0 and len(nlp_referenced_requirements) == 0 :
    #     pipe = PipelineQueryVerifier.check_and_fix_nlu_pipeline(pipe)
    #     res = pipe.predict(test_data)
    #     print(res)
    #     return res
    # elif len(nlu_referenced_requirements) >0:
    #     for r in nlu_referenced_requirements:
    #         pipe.add(nlu.pipe.component_resolution.nlu_ref_to_component(r))
    # elif len(nlp_referenced_requirements) > 0:
    #     for r in nlu_referenced_requirements:
    #         # map back to NLU ref
    #         pipe.add(nlu.pipe.component_resolution.nlu_ref_to_component(NLP_ref_to_NLU_ref(r)))
    # # run pipe with dependencies
    # pipe = PipelineQueryVerifier.check_and_fix_nlu_pipeline(pipe)
    # res = pipe.predict(test_data)
    # print(res)
    # return res
def check_if_storage_ref_exists(storage_ref):
    """Check if storage ref exists as a NLP ref or NLU ref in the spellbook"""
    spaces = [nlu.spellbook.Spellbook.pretrained_models_references, nlu.Spellbook.pretrained_healthcare_model_references]
    for space in spaces :
        if check_namespace_for_storage_ref(space,storage_ref) : return True
    return False
def check_namespace_for_storage_ref(space, storage_ref):
    """Check if storage ref exists as a NLP ref or NLU ref in a given namespace"""
    for lang, mappings in space.items():
        for nlu_ref, nlp_ref in mappings.items():
            if nlu_ref == storage_ref or nlp_ref == storage_ref : return True
    return False