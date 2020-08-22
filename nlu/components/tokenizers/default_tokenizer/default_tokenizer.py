import nlu.pipe_components
from sparknlp.annotator import *

class DefaultTokenizer:
    @staticmethod
    def get_default_model():
        return Tokenizer()\
            .setInputCols(["document"]) \
            .setOutputCol("token")

    @staticmethod
    def get_pretrained_model(name, language):
        return Tokenizer.pre

