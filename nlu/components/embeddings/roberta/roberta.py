from sparknlp.annotator import *

class Roberta:
    @staticmethod
    def get_default_model():
        return RoBertaEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("roberta")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return RoBertaEmbeddings.pretrained(name, language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("roberta")



