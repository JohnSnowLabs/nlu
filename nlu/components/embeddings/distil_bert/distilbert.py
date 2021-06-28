from sparknlp.annotator import *

class DistilBert:
    @staticmethod
    def get_default_model():
        return DistilBertEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("distilbert")

    @staticmethod
    def get_pretrained_model(name, language):
        return DistilBertEmbeddings.pretrained(name, language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("distilbert")



