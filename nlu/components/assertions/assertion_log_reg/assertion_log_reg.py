

class AssertionLogReg:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import AssertionLogRegModel
        return AssertionLogRegModel.pretrained() \
                               .setInputCols(["sentence", "entities", "word_embeddings"]) \
                               .setOutputCol("assertion")


    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        from sparknlp_jsl.annotator import AssertionLogRegModel
        return AssertionLogRegModel.pretrained(name, language,bucket) \
            .setInputCols(["sentence", "entities", "word_embeddings"]) \
            .setOutputCol("assertion")

    def get_default_trainable_model(self):
        from sparknlp_jsl.annotator import  AssertionLogRegApproach
        return AssertionLogRegApproach()\
            .setInputCols(["sentence", "entities", "word_embeddings"]) \
            .setOutputCol("assertion")\
            .setLabelCol('label')
