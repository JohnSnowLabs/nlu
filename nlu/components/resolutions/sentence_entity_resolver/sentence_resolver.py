from sparknlp_jsl.annotator import SentenceEntityResolverModel,SentenceEntityResolverApproach

class SentenceResolver:
    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        return SentenceEntityResolverModel.pretrained(name, language,bucket) \
            .setInputCols(["sentence_embeddings"]) \
            .setDistanceFunction("COSINE") \
            .setNeighbours(3) \
            .setOutputCol("sentence_resolution")

    @staticmethod
    def get_default_trainable_model():
        return SentenceEntityResolverApproach() \
        .setInputCols("sentence_embeddings") \
        .setOutputCol("sentence_resolution") \
        .setDistanceFunction("COSINE") \
        .setNeighbours(1) \
        .setLabelCol('label')
