from sparknlp.annotator import *


class XlmRobertaZeroShotClassifier:
    @staticmethod
    def get_default_model():
        return XlmRoBertaForZeroShotClassification.pretrained() \
            .setInputCols(["token", "document"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return XlmRoBertaForZeroShotClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "document"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)
