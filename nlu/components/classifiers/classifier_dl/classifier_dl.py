import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class ClassifierDl:
    @staticmethod
    def get_default_model():
        return ClassifierDLModel.pretrained() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language):
        return ClassifierDLModel.pretrained(name,language) \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category")



    @staticmethod
    def get_default_trainable_model():
        return ClassifierDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("label") \
            .setBatchSize(64) \
            .setMaxEpochs(20) \
            .setLr(0.5) \
            .setDropout(0.5)