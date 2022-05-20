from sparknlp.annotator import DeBertaForTokenClassification

class TokenDeBerta:
    @staticmethod
    def get_default_model():
        return DeBertaForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return DeBertaForTokenClassification.pretrained(name, language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")



