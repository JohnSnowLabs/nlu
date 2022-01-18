from sparknlp.annotator import *

class ViveknSentiment:
    @staticmethod
    def get_default_model():
        return ViveknSentimentModel.pretrained() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("sentiment")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return ViveknSentimentModel.pretrained(name,language,bucket) \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("sentiment")


    @staticmethod
    def get_default_trainable_model():
        return ViveknSentimentApproach() \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("sentiment")\
            .setSentimentCol("sentiment_label")