import nlu.pipe_components
from sparknlp.annotator import *


class PartOfSpeechJsl:
    @staticmethod
    def get_default_model():
        return PerceptronModel.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos")

    @staticmethod
    def get_pretrained_model(name, language):
        return PerceptronModel.pretrained(name,language) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos")


    @staticmethod
    def get_default_trainable_model():
        return PerceptronApproach() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("pos") \
            .setPosCol("pos_train") \
            .setNIterations(2)
