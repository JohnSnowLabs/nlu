from sparknlp.annotator import *

class LanguageDetector:
    @staticmethod
    def get_default_model():
        return LanguageDetectorDL.pretrained()\
        .setInputCols("document") \
        .setOutputCol("language")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return LanguageDetectorDL.pretrained(name,language,bucket) \
            .setInputCols("document") \
            .setOutputCol("language")
