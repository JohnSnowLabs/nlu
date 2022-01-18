from sparknlp.annotator import XlmRoBertaForSequenceClassification


class SeqXlmRobertaClassifier:
    @staticmethod
    def get_default_model():
        return XlmRoBertaForSequenceClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return XlmRoBertaForSequenceClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category")
