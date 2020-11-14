from sparknlp.annotator import *

class Yake:
    @staticmethod
    def get_default_model():
        return YakeModel() \
                   .setInputCols("token") \
                   .setOutputCol("keywords") \
                   .setMinNGrams(1) \
                   .setMaxNGrams(3) \
                   .setNKeywords(3) \


    @staticmethod
    def get_pretrained_model(name, language):
        return Yake.get_default_model()


