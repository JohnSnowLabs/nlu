from sparknlp.annotator import *

class OpenaiEmbeddings:
    @staticmethod
    def get_default_model():
        return OpenAIEmbeddings() \
                .setInputCols("document") \
                .setOutputCol("embeddings")


    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return  OpenAIEmbeddings() \
                .setInputCols("document") \
                .setOutputCol("embeddings")




