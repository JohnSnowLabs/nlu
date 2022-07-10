from sparknlp.annotator import SpanBertCorefModel


class CorefBert:
    @staticmethod
    def get_default_model():
        return SpanBertCorefModel.pretrained() \
            .setInputCols(["document", "token"]) \
            .setOutputCol("coref")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return SpanBertCorefModel.pretrained(name, language, bucket) \
            .setInputCols(["document", "token"]) \
            .setOutputCol("c oref")
