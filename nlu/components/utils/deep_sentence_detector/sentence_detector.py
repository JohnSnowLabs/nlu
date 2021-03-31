from sparknlp.annotator import *

class SparkNLPSentenceDetector:
    @staticmethod
    def get_default_model():
        return SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")



