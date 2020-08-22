import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *


class SentimentDl:
    @staticmethod
    def get_default_model():  # TODO cannot runw ithouth a dictionary!
        return SentimentDetectorModel() \
            .setInputCols("lemma", "sentence_embeddings") \
            .setOutputCol("sentiment") \
 \
    @staticmethod
    def get_default_trainable_model():
        return SentimentDetector() \
            .setInputCols("lemma", "sentence_embeddings") \
            .setOutputCol("sentiment") \
            .setDictionary("dcit_TODO???")
