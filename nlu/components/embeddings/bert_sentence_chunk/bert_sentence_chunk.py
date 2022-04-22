from sparknlp.annotator import *


class BertSentenceChunkEmbeds:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import BertSentenceChunkEmbeddings
        return BertSentenceChunkEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("bert")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        from sparknlp_jsl.annotator import BertSentenceChunkEmbeddings
        return BertSentenceChunkEmbeddings.pretrained(name,language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("bert")



