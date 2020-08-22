import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class SparkNLPStemmer:
    @staticmethod
    def get_default_model():
        return Stemmer() \
            .setInputCols(["token"]) \
            .setOutputCol("stem")


