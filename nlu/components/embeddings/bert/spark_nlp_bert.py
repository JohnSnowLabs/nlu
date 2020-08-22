import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class SparkNLPBert:
    @staticmethod
    def get_default_model():
        return   BertEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("bert")

    @staticmethod
    def get_pretrained_model(name, language):
        return   BertEmbeddings.pretrained(name,language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("bert")



