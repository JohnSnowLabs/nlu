from sparknlp.annotator import DeBertaForSequenceClassification
class SeqDebertaClassifier:
    @staticmethod
    def get_default_model():
        return DeBertaForSequenceClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return DeBertaForSequenceClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)





