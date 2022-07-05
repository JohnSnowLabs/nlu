from sparknlp.annotator import RoBertaForQuestionAnswering


class SpanRobertaClassifier:
    @staticmethod
    def get_default_model():
        return RoBertaForQuestionAnswering.pretrained() \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return RoBertaForQuestionAnswering.pretrained(name, language, bucket) \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")
