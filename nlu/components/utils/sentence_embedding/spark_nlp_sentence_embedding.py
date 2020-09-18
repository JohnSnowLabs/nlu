import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class SparkNLPSentenceEmbeddinge:
    @staticmethod
    def get_default_model():
        return  SentenceEmbeddings() \
            .setInputCols(["document", "embeddings"]) \
            .setOutputCol("sentence_embeddings") \
            .setPoolingStrategy("AVERAGE")
