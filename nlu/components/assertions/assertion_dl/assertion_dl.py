
class AssertionDL:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import AssertionDLModel

        return AssertionDLModel.pretrained() \
                               .setInputCols(["sentence", "entities", "word_embeddings"]) \
                               .setOutputCol("assertion")


    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        from sparknlp_jsl.annotator import AssertionDLModel
        return AssertionDLModel.pretrained(name, language,bucket) \
            .setInputCols(["sentence", "entities", "word_embeddings"]) \
            .setOutputCol("assertion")

    @staticmethod
    def get_default_trainable_model():
        from sparknlp_jsl.annotator import AssertionDLApproach
        return AssertionDLApproach()\
            .setInputCols(["sentence", "entities", "word_embeddings"]) \
            .setOutputCol("assertion")\
            .setLabelCol('label')
