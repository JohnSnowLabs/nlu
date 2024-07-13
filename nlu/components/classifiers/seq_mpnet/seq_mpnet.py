from sparknlp.annotator import MPNetForSequenceClassification


class SeqMPNetClassifier:
    @staticmethod
    def get_default_model():
        return MPNetForSequenceClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return MPNetForSequenceClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")
