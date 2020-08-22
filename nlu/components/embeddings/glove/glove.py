import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class Glove:
    @staticmethod
    def get_default_model():
        return   WordEmbeddingsModel.pretrained() \
        .setInputCols("document", "token") \
        .setOutputCol("glove_embeddings")


    @staticmethod
    def get_pretrained_model(name, language):
        return   WordEmbeddingsModel.pretrained(name,language) \
            .setInputCols("document", "token") \
            .setOutputCol("glove_embeddings")



