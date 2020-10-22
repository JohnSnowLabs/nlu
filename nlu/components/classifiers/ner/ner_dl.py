import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class NERDL:
    @staticmethod
    def get_default_model():  \
        return NerDLModel.pretrained(name='ner_dl_bert', lang='en') \
            .setInputCols(["sentence", "token", "word_embeddings"]) \
            .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language):
        return NerDLModel.pretrained(name,language) \
            .setInputCols(["sentence", "token", "word_embeddings"]) \
            .setOutputCol("ner")

    @staticmethod
    def get_default_trainable_model():
        return NerDLApproach() \
            .setInputCols(["sentence", "token", "embeddings"]) \
            .setLabelColumn("label") \
            .setOutputCol("ner") \
            .setMaxEpochs(10) \
            .setRandomSeed(0) \
            .setVerbose(2)
