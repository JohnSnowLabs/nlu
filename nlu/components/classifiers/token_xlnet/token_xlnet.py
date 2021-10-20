from sparknlp.annotator import XlnetForTokenClassification

class TokenXlnet:
    @staticmethod
    def get_default_model():
        return XlnetForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return XlnetForTokenClassification.pretrained(name, language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")



