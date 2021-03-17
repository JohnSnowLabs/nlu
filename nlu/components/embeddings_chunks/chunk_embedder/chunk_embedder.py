import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class ChunkEmbedder:
    @staticmethod
    def get_default_model():
        return   ChunkEmbeddings() \
            .setInputCols(["entities", "work_embeddings"]) \
            .setOutputCol("word_embeddings") \
            .setPoolingStrategy("AVERAGE")