class TokenBertHealthcare:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import MedicalBertForTokenClassifier
        return MedicalBertForTokenClassifier.pretrained() \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        from sparknlp_jsl.annotator import MedicalBertForTokenClassifier
        return MedicalBertForTokenClassifier.pretrained(name, language, bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")
