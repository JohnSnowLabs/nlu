from sparknlp.annotator import *

class M2M100:
    @staticmethod
    def get_default_model():
        return M2M100Transformer.pretrained() \
                .setInputCols("document") \
                .setOutputCol("generation")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return M2M100Transformer.pretrained(name, language, bucket) \
                .setInputCols("document") \
                .setOutputCol("generation")