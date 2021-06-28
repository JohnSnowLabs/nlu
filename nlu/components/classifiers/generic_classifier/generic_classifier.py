from sparknlp_jsl.annotator import  GenericClassifierModel, GenericClassifierApproach
from sparknlp_jsl.base import *
class SentimentDl:
    @staticmethod
    def get_default_model():
        return GenericClassifierModel.pretrained() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \


    @staticmethod
    def get_pretrained_model(name, language):
        return GenericClassifierModel.pretrained(name,language) \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \


    @staticmethod
    def get_default_trainable_model():
        return GenericClassifierApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("y") \
            .setMaxEpochs(2) \
            .setEnableOutputLogs(True)