from sparknlp.annotator import CamemBertForTokenClassification

class TokenCamembert:
    @staticmethod
    def get_default_model():
        return CamemBertForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return CamemBertForTokenClassification.pretrained(name, language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")



