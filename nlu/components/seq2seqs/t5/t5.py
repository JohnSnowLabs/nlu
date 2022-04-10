from sparknlp.annotator import *

class T5:
    @staticmethod
    def get_default_model():
        return T5Transformer.pretrained() \
        .setInputCols("document") \
        .setOutputCol("T5")

    @staticmethod
    def get_pretrained_model(name, language,bucket=None):
        return T5Transformer.pretrained(name, language,bucket) \
            .setInputCols("document") \
            .setOutputCol("T5")






