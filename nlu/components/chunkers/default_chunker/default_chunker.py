import nlu.pipe_components
from sparknlp.annotator import *

class DefaultChunker:
    @staticmethod
    def get_default_model():
        return Chunker() \
            .setInputCols(["document", "pos"]) \
            .setOutputCol("chunk") \
            .setRegexParsers(["‹NNP›+", "‹DT|PP\\$›?‹JJ›*‹NN›"])


