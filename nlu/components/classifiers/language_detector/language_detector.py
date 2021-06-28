from sparknlp.annotator import *

class LanguageDetector:
    @staticmethod
    def get_default_model():
        return LanguageDetectorDL.pretrained()\
        .setInputCols("document") \
        .setOutputCol("language")

    @staticmethod
    def get_pretrained_model(name, language):
        return LanguageDetectorDL.pretrained(name,language) \
            .setInputCols("document") \
            .setOutputCol("language")


    @staticmethod
    def get_default_trainable_model():
        print ("TRAINING LANGUAGE DETECTORS NOT SUPPORTED")
        return -1  # todo throw rexception