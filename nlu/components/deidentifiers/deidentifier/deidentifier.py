
class Deidentifier:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import DeIdentificationModel
        return DeIdentificationModel.pretrained(name = 'redl_bodypart_direction_biobert') \
            .setInputCols(["entities", "sentence", "token"]) \
            .setOutputCol("deidentified")


    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        from sparknlp_jsl.annotator import DeIdentificationModel
        return DeIdentificationModel.pretrained(name, language,bucket) \
            .setInputCols(["entities", "sentence", "token"]) \
            .setOutputCol("deidentified")


    @staticmethod
    def get_trainable_model():
        from sparknlp_jsl.annotator import DeIdentification
        return DeIdentification\
            .setInputCols(["entities", "sentence", "token"]) \
            .setOutputCol("deidentified")
