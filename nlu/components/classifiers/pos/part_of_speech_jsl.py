from sparknlp.annotator import *


class PartOfSpeechJsl:
    @staticmethod
    def get_default_model():
        return PerceptronModel.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos")


    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return PerceptronModel.pretrained(name,language,bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos")


    @staticmethod
    def get_default_trainable_model():
        return PerceptronApproach() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos") \
            .setPosCol("y") \
            .setNIterations(2)

