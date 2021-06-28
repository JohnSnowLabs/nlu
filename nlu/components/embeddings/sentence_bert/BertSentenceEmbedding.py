from sparknlp.annotator import *


class BertSentence:
    @staticmethod
    def get_default_model():
        return BertSentenceEmbeddings.pretrained() \
        .setInputCols("sentence") \
        .setOutputCol("sentence_embeddings")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return BertSentenceEmbeddings.pretrained(name,language,bucket) \
            .setInputCols('sentence') \
            .setOutputCol("sentence_embeddings")



