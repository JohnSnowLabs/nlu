import sparknlp.annotator


class RoBertaForZeroShotClassification:
    @staticmethod
    def get_default_model():
        return sparknlp.annotator.RoBertaForZeroShotClassification.pretrained() \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return sparknlp.annotator.RoBertaForZeroShotClassification.pretrained(name, language, bucket) \
            .setInputCols(["token", "sentence"]) \
            .setOutputCol("category") \
            .setCaseSensitive(True)
