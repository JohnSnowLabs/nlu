from sparknlp.annotator import *


class BertZeroShotClassifier:
    @staticmethod
    def get_default_model():
        return BertForZeroShotClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return BertForZeroShotClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)
