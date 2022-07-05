from sparknlp.annotator import AlbertForQuestionAnswering


class SpanAlbertClassifier:
    @staticmethod
    def get_default_model():
        return AlbertForQuestionAnswering.pretrained() \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return AlbertForQuestionAnswering.pretrained(name, language, bucket) \
            .setInputCols(["document_question", "context"]) \
            .setOutputCol("answer")
