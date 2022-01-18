from sparknlp.annotator import XlnetForSequenceClassification


class SeqXlnetClassifier:
    @staticmethod
    def get_default_model():
        return XlnetForSequenceClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return XlnetForSequenceClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")
