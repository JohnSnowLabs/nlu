import json
import os

from nlu.universe.annotator_class_universe import AnnoClassRef
from nlu.universe.component_universes import  jsl_id_to_empty_component
from nlu.universe.universes import Licenses

# These Spark NLP imports are not unused and will be dynamically loaded, do not remove!
from sparknlp.annotator import *
from sparknlp.base import *
# we just use some classes here, so that intelij cleanup will not remove the imports
DocumentAssembler.name,BertEmbeddings.__name__


def is_pipe(model_path):
    """Check whether there is a pyspark component_list stored in path"""
    return 'stages' in os.listdir(model_path)


def is_model(model_path):
    """Check whether there is a model_anno_obj stored in path"""
    return 'metadata' in os.listdir(model_path)


def get_model_class(model_path):
    """Extract class from a model_anno_obj saved in model_path"""
    with open(model_path + '/stages/part-00000') as json_file:
        java_class = json.load(json_file)['class']
        pyth_class = java_class.split('.')[-1]
    return java_class, pyth_class


def verify_and_create_model(model_path: str):
    """
     Build model_anno_obj with requirements
     Figures out class name by checking metadata json file
     assumes metadata is always called part-00000
    """
    with open(model_path + '/metadata/' + 'part-00000') as json_f:
        class_name = json.load(json_f)['class'].split('.')[-1]
        # The last element in the Class name can be used to just load the model_anno_obj from disk!
        # Just call eval on it, which will give you the actual Python class reference which should have a .load() method
        try:

            m = eval(class_name).load(model_path)
        except:
            from nlu.utils.environment.offline_load_utils_licensed import verify_model_licensed
            m = verify_model_licensed(class_name, model_path)
    os_annos = AnnoClassRef.get_os_pyclass_2_anno_id_dict()
    hc_annos = AnnoClassRef.get_hc_pyclass_2_anno_id_dict()
    ocr_annos = AnnoClassRef.get_ocr_pyclass_2_anno_id_dict()

    # component_type, nlu_anno_class, = resolve_annotator_class_to_nlu_component_info(class_name)

    # construct_component_from_identifier('xx', nlu_ref = class_name, nlp_ref = class_name, anno_class_name=class_name)
    if class_name in os_annos.keys():
        jsl_anno_id = os_annos[class_name]
        nlu_component = jsl_id_to_empty_component(jsl_anno_id)
        return nlu_component.set_metadata(m,
                                          jsl_anno_id, jsl_anno_id,
                                          'xx',
                                          False, Licenses.open_source)

    elif class_name in hc_annos.keys():
        jsl_anno_id = hc_annos[class_name]
        nlu_component = jsl_id_to_empty_component(jsl_anno_id)
        return nlu_component.set_metadata(m,
                                          jsl_anno_id, jsl_anno_id,
                                          'xx',
                                          False, Licenses.hc)
    elif class_name in ocr_annos.keys():
        pass
    raise ValueError(f'Could not detect correct Class for nlp class ={class_name}')



def test_check_if_string_in_file(file_name, string_to_search, regex=False):
    """ Check if any line in the file contains given string """
    # Open the file in read only mode
    # print('reading ', file_name)
    import re

    with open(file_name, 'r') as read_obj:
        # Read all lines in the file one by one
        for line in read_obj:
            # For each line, check if line contains the string
            if regex:
                if len(re.findall(string_to_search, line)) > 0: return True
            else:
                if string_to_search in line: return True
    return False


def NLP_ref_to_NLU_ref(nlp_ref, lang):
    """Resolve a Spark NLP reference to a NLU reference"""
    from nlu import Spellbook
    nlu_namespaces_to_check = [Spellbook.pretrained_pipe_references, Spellbook.pretrained_models_references,
                               Spellbook.pretrained_healthcare_model_references, Spellbook.pretrained_pipe_references,
                               ]
    for dict_ in nlu_namespaces_to_check:
        if lang in dict_.keys():
            for reference in dict_[lang]:
                if dict_[lang][reference] == nlp_ref:
                    return reference
