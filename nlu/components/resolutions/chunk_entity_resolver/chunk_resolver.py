from sparknlp_jsl.annotator import ChunkEntityResolverModel,ChunkEntityResolverApproach

class ChunkResolver:
    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        return ChunkEntityResolverModel.pretrained(name, language,bucket) \
            .setInputCols(["token","chunk_embeddings"]) \
            .setNeighbours(3) \
            .setOutputCol("chunk_resolution")

    @staticmethod
    def get_default_trainable_model():
        return ChunkEntityResolverApproach() \
        .setInputCols("token","chunk_embeddings") \
        .setOutputCol("chunk_resolution") \
                .setLabelCol('y')\
        .setNormalizedCol("_y") \
            .setNeighbours(1000) \
            .setAlternatives(25) \
            .setEnableWmd(True).setEnableTfidf(True).setEnableJaccard(True) \
            .setEnableSorensenDice(True).setEnableJaroWinkler(True).setEnableLevenshtein(True) \
            .setDistanceWeights([1, 2, 2, 1, 1, 1]) \
            .setAllDistancesMetadata(True) \
            .setPoolingStrategy("MAX") \
            .setThreshold(1e32)
        # .setDistanceFunction("COSINE") \
