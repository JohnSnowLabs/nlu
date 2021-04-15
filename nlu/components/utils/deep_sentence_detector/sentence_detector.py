from sparknlp.annotator import *

class SparkNLPDeepSentenceDetector:
    @staticmethod
    def get_default_model():
        return SentenceDetectorDLModel.pretrained() \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")



