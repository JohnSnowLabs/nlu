class MedSummarizer:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import MedicalSummarizer
        return MedicalSummarizer.pretrained()

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        from sparknlp_jsl.annotator import MedicalSummarizer
        return MedicalSummarizer.pretrained(name, language, bucket)
