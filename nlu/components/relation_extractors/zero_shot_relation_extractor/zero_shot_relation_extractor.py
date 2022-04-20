

class ZeroShotRelationExtractor:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import ZeroShotRelationExtractionModel
        return ZeroShotRelationExtractionModel.pretrained(name = 'redl_bodypart_direction_biobert') \
                   .setInputCols(["entities", "sentence"]) \
                   .setOutputCol("relations")


    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        from sparknlp_jsl.annotator import ZeroShotRelationExtractionModel
        return ZeroShotRelationExtractionModel.pretrained(name, language,bucket) \
            .setInputCols(["entities", "sentence"]) \
            .setOutputCol("relations")

