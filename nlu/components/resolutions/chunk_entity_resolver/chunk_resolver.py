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
        .setDistanceFunction("COSINE") \
            .setNeighbours(1) \
            .setLabelCol('label')
