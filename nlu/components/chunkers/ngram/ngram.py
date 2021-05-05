from sparknlp.annotator import *

class NGram:
    @staticmethod
    def get_default_model():
        return NGramGenerator() \
            .setInputCols(["token"]) \
            .setOutputCol("ngrams") \
            .setN(2) \
            .setEnableCumulative(False)

