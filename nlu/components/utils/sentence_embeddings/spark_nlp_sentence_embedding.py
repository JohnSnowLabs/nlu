from sparknlp.annotator import *

class SparkNLPSentenceEmbeddings:
    @staticmethod
    def get_default_model():
        return SentenceEmbeddings() \
            .setInputCols(["document", "word_embeddings"]) \
            .setOutputCol("sentence_embeddings") \
            .setPoolingStrategy("AVERAGE")
