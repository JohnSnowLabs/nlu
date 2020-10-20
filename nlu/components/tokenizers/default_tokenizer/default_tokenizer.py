from sparknlp.annotator import *

class DefaultTokenizer:
    @staticmethod
    def get_default_model():
        return Tokenizer()\
            .setInputCols(["document"]) \
            .setOutputCol("token")


