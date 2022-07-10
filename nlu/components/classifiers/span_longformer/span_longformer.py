from sparknlp.annotator import LongformerForQuestionAnswering


class SpanLongFormerClassifier:
    @staticmethod
    def get_default_model():
        return LongformerForQuestionAnswering.pretrained() \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return LongformerForQuestionAnswering.pretrained(name, language, bucket) \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")
