from sparknlp.annotator import *

class Marian:

    @staticmethod
    def get_default_model():
        return MarianTransformer.pretrained() \
            .setInputCols("document") \
            .setOutputCol("marian")

    @staticmethod
    def get_pretrained_model(name, language):
        return MarianTransformer.pretrained(name, language) \
            .setInputCols("document") \
            .setOutputCol("marian")



