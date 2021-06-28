from sparknlp.annotator import *

class XLM:
    @staticmethod
    def get_default_model():
        return XlmRoBertaEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("roberta")

    @staticmethod
    def get_pretrained_model(name, language):
        return XlmRoBertaEmbeddings.pretrained(name, language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("roberta")



