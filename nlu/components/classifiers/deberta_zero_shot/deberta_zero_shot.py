from sparknlp.annotator import DeBertaForZeroShotClassification


class DeBertaZeroShotClassifier:
    @staticmethod
    def get_default_model():
        return DeBertaForZeroShotClassification.pretrained() \
            .setInputCols(["token", "document"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return DeBertaForZeroShotClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "document"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)