from sparknlp.annotator import LongformerForSequenceClassification


class SeqLongformerClassifier:
    @staticmethod
    def get_default_model():
        return LongformerForSequenceClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return LongformerForSequenceClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")
