from sparknlp.annotator import *
class SparkNLPBartTransformer:
    @staticmethod
    def get_default_model():
        return BartTransformer.pretrained()

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return BartTransformer.pretrained(name, language, bucket)
