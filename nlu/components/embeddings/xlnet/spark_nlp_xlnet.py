from sparknlp.annotator import *

class SparkNLPXlnet:
    @staticmethod
    def get_default_model():
        return XlnetEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("word_embeddings")



    @staticmethod
    def get_pretrained_model(name, language):
        return XlnetEmbeddings.pretrained(name,language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("word_embeddings")
