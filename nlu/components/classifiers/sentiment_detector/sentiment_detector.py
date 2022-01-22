from sparknlp.annotator import *
SentimentDetectorModel

class Sentiment:
    @staticmethod
    def get_default_model():
        return SentimentDetectorModel() \
            .setInputCols("lemma", "document") \
            .setOutputCol("sentiment") \


    @staticmethod
    def get_default_trainable_model():
        return SentimentDetector() \
            .setInputCols("lemma", "document") \
            .setOutputCol("sentiment") \
            .setDictionary("dict_todo???")
