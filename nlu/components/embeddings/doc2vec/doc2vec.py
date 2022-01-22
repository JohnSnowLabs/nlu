from sparknlp.annotator import *


class Doc2Vec:
    @staticmethod
    def get_default_model():
        return Doc2VecModel.pretrained() \
        .setInputCols("token") \
        .setOutputCol("sentence_embeddings")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return Doc2VecModel.pretrained(name,language,bucket) \
            .setInputCols("token") \
            .setOutputCol("sentence_embeddings")
    @staticmethod
    def get_trainable_model():
        return Doc2VecApproach()\
            .setInputCols("token") \
            .setOutputCol("sentence_embeddings")


