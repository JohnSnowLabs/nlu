from sparknlp.annotator import ElmoEmbeddings

class SparkNLPElmo:
    @staticmethod
    def get_default_model():
        return ElmoEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("word_embeddings")



    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return ElmoEmbeddings.pretrained(name,language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("word_embeddings")

