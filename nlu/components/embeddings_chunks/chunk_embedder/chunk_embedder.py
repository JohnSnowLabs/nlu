import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class ChunkEmbedder:
    @staticmethod
    def get_default_model():
        return   ChunkEmbeddings() \
            .setInputCols(["chunk", "glove_embeddings"]) \
            .setOutputCol("chunk_embeddings") \
            .setPoolingStrategy("AVERAGE")