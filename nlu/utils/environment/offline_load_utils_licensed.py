from sparknlp_jsl.annotator import *
def verify_model_licensed(class_name : str, model_path:str):
    """
    Load a licensed model from HDD
    """
    try :
        m = eval(class_name).load(model_path)
        return m
    except:
        print(f"Could not load Annotator class={class_name} located in {model_path}. Try updaing spark-nlp-jsl")
