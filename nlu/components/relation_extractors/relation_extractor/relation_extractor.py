class RelationExtraction:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import RelationExtractionModel
        return RelationExtractionModel.pretrained() \
            .setInputCols(["word_embeddings", "pos", "chunk", "dependency"]) \
            .setOutputCol("relations")

    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        from sparknlp_jsl.annotator import RelationExtractionModel
        if name == "posology_re": bucket = None
        return RelationExtractionModel.pretrained(name, language, bucket) \
            .setInputCols(["word_embeddings", "pos", "entities", "unlabeled_dependency"]) \
            .setOutputCol("relations")

    @staticmethod
    def get_default_trainable_model():
        from sparknlp_jsl.annotator import RelationExtractionApproach
        return RelationExtractionApproach() \
            .setInputCols(["word_embeddings", "pos", "entities", "unlabeled_dependency"]) \
            .setOutputCol("relations") \
            .setLabelCol('label')
