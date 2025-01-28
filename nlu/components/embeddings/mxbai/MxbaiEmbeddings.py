from sparknlp.annotator import *


class MxbaiEmbeddings:
    @staticmethod
    def get_default_model():
        from sparknlp.annotator import MxbaiEmbeddings
        return MxbaiEmbeddings.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("mxbai_embeddings")

    # @staticmethod
    # def get_pretrained_model(name, language, bucket=None):
    #     return MxbaiEmbeddings.pretrained(name,language,bucket) \
    #         .setInputCols(["document"]) \
    #         .setOutputCol("sentence_embeddings")



