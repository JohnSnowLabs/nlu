from sparknlp.annotator import BertForTokenClassification

class TokenBert:
    @staticmethod
    def get_default_model():
        return BertForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("tokenbert")

    @staticmethod
    def get_pretrained_model(name, language):
        return BertForTokenClassification.pretrained(name, language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("tokenbert")



