import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class NERDLCRF:
    @staticmethod
    def get_default_model():
        return NerCrfModel.pretrained(name="ner_dl_bert") \
            .setInputCols(["sentence", "token", "pos", "word_embeddings"]) \
            .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language):
        return NerCrfModel.pretrained(name,language) \
            .setInputCols(["sentence", "token", "pos", "word_embeddings"]) \
            .setOutputCol("ner")

    @staticmethod
    def get_default_trainable_model():
        return NerCrfApproach() \
            .setInputCols(["sentence", "token", "pos"]) \
            .setLabelColumn("label") \
            .setOutputCol("ner") \
            .setMinEpochs(1) \
            .setMaxEpochs(20) \
            .setLossEps(1e-3) \
            .setDicts(["ner-corpus/dict.txt"]) \
            .setL2(1) \
            .setC0(1250000) \
            .setRandomSeed(0) \
            .setVerbose(2)