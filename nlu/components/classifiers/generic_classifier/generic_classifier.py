from sparknlp_jsl.annotator import  GenericClassifierModel, GenericClassifierApproach
from sparknlp_jsl.base import *
class GenericClassifier:
    @staticmethod
    def get_default_model():
        return GenericClassifierModel.pretrained() \
            .setInputCols("feature_vector") \
            .setOutputCol("generic_classification") \


    @staticmethod
    def get_pretrained_model(name, language):
        return GenericClassifierModel.pretrained(name,language) \
            .setInputCols("feature_vector") \
            .setOutputCol("generic_classification") \


    @staticmethod
    def get_default_trainable_model():
        return GenericClassifierApproach() \
            .setInputCols("feature_vector") \
            .setOutputCol("generic_classification") \
            .setLabelColumn("y") \
            .setEpochsNumber(2)
