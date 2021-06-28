from sparknlp_jsl.annotator import ChunkMergeModel

class ChunkMerger:
    @staticmethod
    def get_default_model():
        return ChunkMergeModel() \
            .setInputCol("entities") \
            .setOutputCol("merged_entities")


