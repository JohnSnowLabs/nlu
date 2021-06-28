from sparknlp.annotator import *

class ChunkEmbedder:
    @staticmethod
    def get_default_model():
        return ChunkEmbeddings() \
            .setInputCols(["entities", "word_embeddings"]) \
            .setOutputCol("chunk_embeddings") \
            .setPoolingStrategy("AVERAGE")