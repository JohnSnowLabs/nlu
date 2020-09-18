import nlu.pipe_components
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *

class SentenDetectorDeep:
    @staticmethod
    def get_default_model():
        return DeepSentenceDetector() \
            .setInputCols(["document", "token", "ner_chunk"]) \
            .setOutputCol("sentence") \
            .setIncludePragmaticSegmenter(True) \
            .setEndPunctuation([".", "?"])
