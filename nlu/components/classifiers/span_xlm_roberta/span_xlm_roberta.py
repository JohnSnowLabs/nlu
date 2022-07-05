from sparknlp.annotator import XlmRoBertaForQuestionAnswering


class SpanXlmRobertaClassifier:
    @staticmethod
    def get_default_model():
        return XlmRoBertaForQuestionAnswering.pretrained() \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return XlmRoBertaForQuestionAnswering.pretrained(name, language, bucket) \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")
