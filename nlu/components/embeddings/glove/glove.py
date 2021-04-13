from sparknlp.annotator import *

class Glove:
    @staticmethod
    def get_default_model():
        return WordEmbeddingsModel.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("word_embeddings")


    @staticmethod
    def get_pretrained_model(name, language, bucket = None):
        return WordEmbeddingsModel.pretrained(name,language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("word_embeddings")



