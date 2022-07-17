from sparknlp.annotator import DeBertaForQuestionAnswering


class SpanDeBertaClassifier:
    @staticmethod
    def get_default_model():
        return DeBertaForQuestionAnswering.pretrained() \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return DeBertaForQuestionAnswering.pretrained(name, language, bucket) \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")
