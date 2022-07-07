class ChunkMapper:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import ChunkMapperModel
        return ChunkMapperModel() \
            .setInputCols("chunk") \
            .setOutputCol("mapped_chunk")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        from sparknlp_jsl.annotator import ChunkMapperModel
        return ChunkMapperModel.pretrained(name, language, bucket) \
            .setInputCols('mapped_chunk') \
            .setOutputCol("mapped_chunk")
