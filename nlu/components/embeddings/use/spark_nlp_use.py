from sparknlp.annotator import *

class SparkNLPUse:
    @staticmethod
    def get_default_model():
        return UniversalSentenceEncoder.pretrained() \
            .setInputCols("sentence") \
            .setOutputCol("use_embeddings")

    @staticmethod
    def get_pretrained_model(name, language):
        return UniversalSentenceEncoder.pretrained(name, language) \
            .setInputCols("sentence") \
            .setOutputCol("use_embeddings")

