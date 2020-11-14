import nlu.pipe_components
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

class SentenDetectorDeep:
    @staticmethod
    def get_default_model():
        return SentenceDetectorDLModel()\
            .setInputCols(["document"]) \
            .setOutputCol("sentence")
