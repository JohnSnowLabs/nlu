import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class T5:
    @staticmethod
    def get_default_model():
        return   T5Transformer.pretrained() \
        .setInputCols("document") \
        .setOutputCol("T5")

    @staticmethod
    def get_pretrained_model(name, language):
        return   T5Transformer.pretrained(name, language) \
            .setInputCols("document") \
            .setOutputCol("T5")



