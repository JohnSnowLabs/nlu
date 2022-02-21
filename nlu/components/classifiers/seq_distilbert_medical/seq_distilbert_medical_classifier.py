class SeqDilstilBertMedicalClassifier:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import MedicalDistilBertForSequenceClassification
        return MedicalDistilBertForSequenceClassification.pretrained()
    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        from sparknlp_jsl.annotator import MedicalDistilBertForSequenceClassification
        return MedicalDistilBertForSequenceClassification.pretrained(name, language, bucket)





