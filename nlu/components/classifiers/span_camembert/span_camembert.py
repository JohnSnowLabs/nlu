from sparknlp.annotator import CamemBertForQuestionAnswering


class SpanCamemBert:
    @staticmethod
    def get_default_model():
        return CamemBertForQuestionAnswering.pretrained() \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return CamemBertForQuestionAnswering.pretrained(name, language, bucket) \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")
