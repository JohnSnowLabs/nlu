from sparknlp.annotator import *

class NERDL:
    @staticmethod
    def get_default_model():  \
        return NerDLModel.pretrained(name='ner_dl_bert', lang='en') \
            .setInputCols(["sentence", "token", "word_embeddings"]) \
            .setOutputCol("ner") \
            .setIncludeConfidence(True)



    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return NerDLModel.pretrained(name,language,bucket) \
            .setInputCols(["sentence", "token", "word_embeddings"]) \
            .setOutputCol("ner") \
            .setIncludeConfidence(True)

    @staticmethod
    def get_default_trainable_model():
        return NerDLApproach() \
            .setInputCols(["sentence", "token", "word_embeddings"]) \
            .setLabelColumn("y") \
            .setOutputCol("ner") \
            .setMaxEpochs(2) \
            .setVerbose(0) \
            .setIncludeConfidence(True)