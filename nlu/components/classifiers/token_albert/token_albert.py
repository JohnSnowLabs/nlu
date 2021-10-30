from sparknlp.annotator import AlbertForTokenClassification

class TokenAlbert:
    @staticmethod
    def get_default_model():
        return AlbertForTokenClassification.pretrained() \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return AlbertForTokenClassification.pretrained(name, language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")



