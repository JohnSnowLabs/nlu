from sparknlp_jsl.annotator import DeIdentificationModel

class Deidentifier:
    @staticmethod
    def get_default_model():
        return DeIdentificationModel.pretrained(name = 'redl_bodypart_direction_biobert') \
            .setInputCols(["entities", "sentence", "token"]) \
            .setOutputCol("deidentified")


    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        return DeIdentificationModel.pretrained(name, language,bucket) \
            .setInputCols(["entities", "sentence", "token"]) \
            .setOutputCol("deidentified")

    # def get_default_trainable_model():
    #     return RelationExtractionApproach()\
    #         .setInputCols(["word_embeddings", "pos", "chunk", "unlabeled_dependency"]) \
    #         .setOutputCol("relations") \
    #         .setLabelCol('label')
