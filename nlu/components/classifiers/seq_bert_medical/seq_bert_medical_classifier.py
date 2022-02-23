class SeqBertMedicalClassifier:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import MedicalBertForSequenceClassification
        return MedicalBertForSequenceClassification.pretrained()

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        from sparknlp_jsl.annotator import MedicalBertForSequenceClassification
        return MedicalBertForSequenceClassification.pretrained(name, language, bucket)





