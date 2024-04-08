from sparknlp.annotator import *


class BartZeroShotClassifier:
    @staticmethod
    def get_default_model():
        return BartForZeroShotClassification.pretrained() \
            .setInputCols(["token", "document"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return BartForZeroShotClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "document"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)
