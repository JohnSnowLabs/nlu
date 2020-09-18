import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class SparkNLPLemmatizer:
    @staticmethod
    def get_default_model():
        return LemmatizerModel.pretrained() \
            .setInputCols(["token"]) \
            .setOutputCol("lemma")

    @staticmethod
    def get_pretrained_model(name, language):
        return LemmatizerModel.pretrained(name, language) \
            .setInputCols(["token"]) \
            .setOutputCol("lemma")

    @staticmethod
    def get_default_trainable_model():
        return Lemmatizer \
            .setInputCols(["token"]) \
            .setOutputCol("lemma")
