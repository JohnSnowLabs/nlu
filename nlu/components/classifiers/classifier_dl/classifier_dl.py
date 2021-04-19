from sparknlp.annotator import *
class ClassifierDl:
    @staticmethod
    def get_default_model():
        return ClassifierDLModel.pretrained() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return ClassifierDLModel.pretrained(name,language,bucket) \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category")




    @staticmethod
    def get_trainable_model():
        return ClassifierDLApproach() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("y") \
            .setMaxEpochs(3) \
           .setEnableOutputLogs(True)

    @staticmethod
    def get_offline_model(path):
        return ClassifierDLModel.load() \
            .setInputCols("sentence_embeddings") \
            .setOutputCol("category") \
            .setLabelColumn("label") \
            .setEnableOutputLogs(True)

