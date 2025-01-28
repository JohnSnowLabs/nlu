from sparknlp.annotator import *


class AlbertZeroShotClassifier:
    @staticmethod
    def get_default_model():
        return AlbertForZeroShotClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return AlbertForZeroShotClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)
