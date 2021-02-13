from sparknlp.annotator import *

class SentimentDl:
    @staticmethod
    def get_default_model():
        return   SentimentDLModel.pretrained() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \


    @staticmethod
    def get_pretrained_model(name, language):
        return   SentimentDLModel.pretrained(name,language) \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \


    @staticmethod
    def get_default_trainable_model():
        return SentimentDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("y") \
            .setMaxEpochs(2) \
            .setEnableOutputLogs(True)