from sparknlp.annotator import Word2VecModel, Word2VecApproach


class Word2Vec:
    @staticmethod
    def get_default_model():
        return Word2VecModel.pretrained() \
        .setInputCols("token") \
        .setOutputCol("word_embeddings")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return Word2VecModel.pretrained(name,language,bucket) \
            .setInputCols("token") \
            .setOutputCol("word_embeddings")
    @staticmethod
    def get_trainable_model():
        return Word2VecApproach()\
            .setInputCols("token") \
            .setOutputCol("word_embeddings")


