from sparknlp.annotator import *

class WordSegmenter:
    @staticmethod
    def get_default_model():
        return WordSegmenterModel()\
            .setInputCols(["document"]) \
            .setOutputCol("seg_token")


    @staticmethod
    def get_pretrained_model(name, language):
        return WordSegmenterModel.pretrained(name,language) \
            .setInputCols(["document"]) \
            .setOutputCol("seg_token")
