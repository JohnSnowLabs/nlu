from sparknlp.annotator import *

class SparkNLPElmo:
    @staticmethod
    def get_default_model():
        return ElmoEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("word_embeddings")



    @staticmethod
    def get_pretrained_model(name, language):
        return ElmoEmbeddings.pretrained(name,language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("word_embeddings")

