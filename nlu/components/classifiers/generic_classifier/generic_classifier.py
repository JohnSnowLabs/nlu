class GenericClassifier:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import  GenericClassifierModel
        return GenericClassifierModel.pretrained() \
            .setInputCols("feature_vector") \
            .setOutputCol("generic_classification") \


    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        from sparknlp_jsl.annotator import  GenericClassifierModel
        return GenericClassifierModel.pretrained(name,language,bucket) \
            .setInputCols("feature_vector") \
            .setOutputCol("generic_classification") \


    @staticmethod
    def get_default_trainable_model():
        from sparknlp_jsl.annotator import  GenericClassifierApproach
        return GenericClassifierApproach() \
            .setInputCols("feature_vector") \
            .setOutputCol("generic_classification") \
            .setLabelColumn("y") \
            .setEpochsNumber(2)
