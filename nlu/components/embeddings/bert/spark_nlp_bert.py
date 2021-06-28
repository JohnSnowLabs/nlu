from sparknlp.annotator import *


class SparkNLPBert:
    @staticmethod
    def get_default_model():
        return BertEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("bert")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return BertEmbeddings.pretrained(name,language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("bert")



