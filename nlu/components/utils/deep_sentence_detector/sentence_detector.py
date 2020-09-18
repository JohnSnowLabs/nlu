import nlu.pipe_components
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

class SparkNLPSentenceDetector:
    @staticmethod
    def get_default_model():
        return SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")



