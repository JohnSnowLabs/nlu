from sparknlp.annotator import BertForTokenClassification

class TokenBert:
    @staticmethod
    def get_default_model():
        return BertForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return BertForTokenClassification.pretrained(name, language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")



