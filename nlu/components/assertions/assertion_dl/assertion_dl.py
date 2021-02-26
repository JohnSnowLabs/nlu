from sparknlp_jsl.annotator import AssertionDLModel,AssertionDLApproach

class AssertionDL:
    @staticmethod
    def get_default_model():
        return AssertionDLModel.pretrained(remote_loc='clinical/models') \
                               .setInputCols(["sentence", "chunk", "word_embeddings"]) \
                               .setOutputCol("AssertDLpos")


    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        return AssertionDLModel.pretrained(name, language,bucket) \
            .setInputCols(["sentence", "chunk", "word_embeddings"]) \
            .setOutputCol("AssertDLpos")

    def get_default_trainable_model(self):
        return AssertionDLApproach()\
            .setInputCols(["sentence", "chunk", "word_embeddings"]) \
            .setOutputCol("AssertDLpos")\
            .setLabelCol('label')
