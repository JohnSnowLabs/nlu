from sparknlp.annotator import *
class SeqBertClassifier:
    @staticmethod
    def get_default_model():
        return BertForSequenceClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return BertForSequenceClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)





