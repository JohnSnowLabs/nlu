import sparknlp
from sparknlp.annotator import BGEEmbeddings


class BGE:
    @staticmethod
    def get_default_model():
        return BGEEmbeddings.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("bge_embeddings")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return BGEEmbeddings.pretrained(name,language,bucket) \
            .setInputCols(["document"]) \
            .setOutputCol("bge_embeddings")
