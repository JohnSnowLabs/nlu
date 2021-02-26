from sparknlp_jsl.annotator import SentenceEntityResolverModel,SentenceEntityResolverApproach

class SentenceResolver:
    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        return SentenceEntityResolverModel.pretrained(name, language,bucket) \
            .setInputCols(["sentence_embeddings"])\
            .setOutputCol("resolution")

    @staticmethod
    def get_default_trainable_model(self):
        return SentenceEntityResolverApproach() \
        .setInputCols("sentence_embeddings") \
        .setOutputCol("resolution")\
        .setLabelCol('label')
