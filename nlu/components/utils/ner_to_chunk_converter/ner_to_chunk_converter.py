import nlu.pipe_components
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

class NerToChunkConverter:
    @staticmethod
    def get_default_model():
        return NerConverter() \
            .setInputCols(["sentence", "token", "ner"]) \
            .setOutputCol("entities")