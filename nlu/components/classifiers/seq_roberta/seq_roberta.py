from sparknlp.annotator import RoBertaForSequenceClassification


class SeqRobertaClassifier:
    @staticmethod
    def get_default_model():
        return RoBertaForSequenceClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return RoBertaForSequenceClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")
