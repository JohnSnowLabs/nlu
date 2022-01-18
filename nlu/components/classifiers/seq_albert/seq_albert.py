from sparknlp.annotator import AlbertForSequenceClassification


class SeqAlbertClassifier:
    @staticmethod
    def get_default_model():
        return AlbertForSequenceClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return AlbertForSequenceClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")
