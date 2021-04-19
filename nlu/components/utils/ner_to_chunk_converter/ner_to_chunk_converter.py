from sparknlp.annotator import *

class NerToChunkConverter:
    @staticmethod
    def get_default_model():
        return NerConverter() \
            .setInputCols(["sentence", "token", "ner"]) \
            .setOutputCol("entities")
