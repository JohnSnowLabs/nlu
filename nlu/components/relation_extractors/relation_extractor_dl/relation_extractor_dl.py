from sparknlp_jsl.annotator import RelationExtractionDLModel

class RelationExtractionDL:
    @staticmethod
    def get_default_model():
        return RelationExtractionDLModel.pretrained(name = 'redl_bodypart_direction_biobert') \
                   .setInputCols(["entities", "sentence"]) \
                   .setOutputCol("relations")


    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        return RelationExtractionDLModel.pretrained(name, language,bucket) \
            .setInputCols(["entities", "sentence"]) \
            .setOutputCol("relations")

    # def get_default_trainable_model():
    #     return RelationExtractionApproach()\
    #         .setInputCols(["word_embeddings", "pos", "chunk", "unlabeled_dependency"]) \
    #         .setOutputCol("relations") \
    #         .setLabelCol('label')
