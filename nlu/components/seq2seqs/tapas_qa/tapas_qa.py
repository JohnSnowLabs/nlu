from sparknlp.annotator import *


class TapasQA:
    @staticmethod
    def get_default_model():
        return TapasForQuestionAnswering.pretrained() \
            .setInputCols('table', "document") \
            .setOutputCol("answer")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return TapasForQuestionAnswering.pretrained(name, language, bucket) \
            .setInputCols('table', "document") \
            .setOutputCol("answer")
