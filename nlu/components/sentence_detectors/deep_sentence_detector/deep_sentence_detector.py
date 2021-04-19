from sparknlp.annotator import *

class SentenceDetectorDeep:
    @staticmethod
    def get_default_model():
        return SentenceDetectorDLModel\
            .pretrained()\
            .setInputCols(["document"]) \
            .setOutputCol("sentence")


    @staticmethod
    def get_pretrained_model(name,lang):
        return SentenceDetectorDLModel.pretrained(name,lang) \
            .pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")


    @staticmethod
    def get_trainable_model():
        return SentenceDetectorDLApproach \
            .setInputCol("document") \
            .setOutputCol("sentence")
