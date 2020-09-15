import nlu.pipe_components
import sparknlp
from sparknlp.annotator import *

class MultiClassifier:
    @staticmethod
    def get_default_model():
        return MultiClassifierDLModel.pretrained() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language):
        return MultiClassifierDLModel.pretrained(name,language) \
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