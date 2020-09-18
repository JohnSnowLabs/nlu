import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class NLUStopWordcleaner:
    @staticmethod
    def get_default_model():
        return StopWordsCleaner.pretrained() \
            .setInputCols(["token"]) \
            .setOutputCol("cleanTokens")

    @staticmethod
    def get_pretrained_model(name, language):
        return StopWordsCleaner.pretrained(name, language) \
            .setInputCols(["token"]) \
            .setOutputCol("cleanTokens")

    @staticmethod
    def get_default_trainable_model():
        return StopWordsCleaner \
            .setInputCols(["token"]) \
            .setOutputCol("cleanTokens")
