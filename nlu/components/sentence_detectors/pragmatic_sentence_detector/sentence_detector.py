from sparknlp.annotator import *

class PragmaticSentenceDetector:
    @staticmethod
    def get_default_model():
        return SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")



