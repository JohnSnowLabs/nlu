from sparknlp.annotator import *

class NLUStopWordcleaner:
    @staticmethod
    def get_default_model():
        return StopWordsCleaner.pretrained() \
            .setInputCols(["token"]) \
            .setOutputCol("stopword_less")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return StopWordsCleaner.pretrained(name, language) \
            .setInputCols(["token"]) \
            .setOutputCol("stopword_less")


