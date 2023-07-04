class MedTextGenerator:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import MedicalTextGenerator
        return MedicalTextGenerator.pretrained()

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        from sparknlp_jsl.annotator import MedicalTextGenerator
        return MedicalTextGenerator.pretrained(name, language, bucket)
