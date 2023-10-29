from sparknlp.annotator import E5Embeddings


class E5:
    @staticmethod
    def get_default_model():
        return E5Embeddings.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("e5_embeddings")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return E5Embeddings.pretrained(name,language,bucket) \
            .setInputCols(["document"]) \
            .setOutputCol("e5_embeddings")
