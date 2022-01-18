from sparknlp.annotator import *

class SparkNLPNormalizer:
    @staticmethod
    def get_default_model():
        return Normalizer() \
            .setInputCols(["token"]) \
            .setOutputCol("normalized")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        # There are no pretrained normalizers
        return Normalizer() \
            .setInputCols(["token"]) \
            .setOutputCol("normalized")
