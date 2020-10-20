from sparknlp.annotator import *

class RegexTokenizer:
    @staticmethod
    def get_default_model():
        return Tokenizer()\
            .setInputCols(["sentence"]) \
            .setOutputCol("token")


