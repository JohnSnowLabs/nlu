from sparknlp.annotator import *

class OpenaiCompletion:
    @staticmethod
    def get_default_model():
        return OpenAICompletion() \
                .setInputCols("document") \
                .setOutputCol("completion")


    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return  OpenAICompletion() \
                .setInputCols("document") \
                .setOutputCol("completion")




