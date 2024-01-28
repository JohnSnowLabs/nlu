class SpanMedical:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import MedicalQuestionAnswering

        return MedicalQuestionAnswering.pretrained() \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")



    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        from sparknlp_jsl.annotator import MedicalQuestionAnswering

        return MedicalQuestionAnswering.pretrained(name, language, bucket) \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")
