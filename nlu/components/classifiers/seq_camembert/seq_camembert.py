from sparknlp.annotator import CamemBertForSequenceClassification


class SeqCamembertClassifier:
    @staticmethod
    def get_default_model():
        return CamemBertForSequenceClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return CamemBertForSequenceClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")
