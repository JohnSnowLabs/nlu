from sparknlp_jsl.annotator import AssertionDLModel,AssertionDLApproach

class AssertionDL:
    @staticmethod
    def get_default_model():
        return AssertionDLModel.pretrained() \
                               .setInputCols(["sentence", "entities", "word_embeddings"]) \
                               .setOutputCol("assertion")


    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        return AssertionDLModel.pretrained(name, language,bucket) \
            .setInputCols(["sentence", "entities", "word_embeddings"]) \
            .setOutputCol("assertion")

    def get_default_trainable_model(self):
        return AssertionDLApproach()\
            .setInputCols(["sentence", "entities", "word_embeddings"]) \
            .setOutputCol("assertion")\
            .setLabelCol('label')
