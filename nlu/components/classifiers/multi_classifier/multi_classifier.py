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
            .setOutputCol("multi_category")
    



    @staticmethod
    def get_default_trainable_model():
        return MultiClassifierDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("multi_category") \
            .setLabelColumn("y") \
            .setEnableOutputLogs(True) \
            .setMaxEpochs(2)
