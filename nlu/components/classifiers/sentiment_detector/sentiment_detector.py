import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *
SentimentDetectorModel

class Sentiment:
    @staticmethod
    def get_default_model():
        return SentimentDetectorModel() \
            .setInputCols("lemma", "sentence_embeddings") \
            .setOutputCol("sentiment") \


    @staticmethod
    def get_default_trainable_model():
        return SentimentDetector() \
            .setInputCols("lemma", "sentence_embeddings") \
            .setOutputCol("sentiment") \
            .setDictionary("dict_todo???")
