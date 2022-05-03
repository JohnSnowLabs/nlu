# These Spark NLP imports are not unused and will be dynamically loaded, do not remove!
from sparknlp_jsl.annotator import *
# we just use some classes here, so that intelij cleanup will not remove the imports
MedicalNerModel.name
def verify_model_licensed(class_name: str, model_path: str):
    """
    Load a licensed model_anno_obj from HDD
    """
    try:
        m = eval(class_name).load(model_path)
        return m
    except:
        print(f"Could not load Annotator class={class_name} located in {model_path}. Try updaing spark-nlp-jsl")
