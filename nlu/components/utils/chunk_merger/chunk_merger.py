
class ChunkMerger:
    @staticmethod

    def get_default_model():
        from sparknlp_jsl.annotator import ChunkMergeModel
        return ChunkMergeModel() \
            .setInputCol("entities") \
            .setOutputCol("merged_entities")


