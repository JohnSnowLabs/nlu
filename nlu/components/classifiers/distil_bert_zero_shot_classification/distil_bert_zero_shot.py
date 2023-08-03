from sparknlp.annotator import *


class DistilBertZeroShotClassifier:
    @staticmethod
    def get_default_model():
        return DistilBertForZeroShotClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return DistilBertForZeroShotClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)
