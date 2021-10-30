from sparknlp.annotator import DistilBertForTokenClassification

class TokenDistilBert:
    @staticmethod
    def get_default_model():
        return DistilBertForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return DistilBertForTokenClassification.pretrained(name, language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")



