import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *


class BertSentence:
    @staticmethod
    def get_default_model():
        return   BertSentenceEmbeddings.pretrained() \
        .setInputCols("document") \
        .setOutputCol("bert_sentence")

    @staticmethod
    def get_pretrained_model(name, language):
        return   BertSentenceEmbeddings.pretrained(name,language) \
            .setInputCols('document') \
            .setOutputCol("bert_sentence")



