import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class SparkNLPXlnet:
    @staticmethod
    def get_default_model():
        return   XlnetEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("xlnet_embeddings")



    @staticmethod
    def get_pretrained_model(name, language):
        return   XlnetEmbeddings.pretrained(name,language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("xlnet_embeddings")
