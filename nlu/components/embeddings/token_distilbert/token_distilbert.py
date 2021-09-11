from sparknlp.annotator import DistilBertForTokenClassification

class TokenDistilBert:
    @staticmethod
    def get_default_model():
        return DistilBertForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("tokendistilbert")

    @staticmethod
    def get_pretrained_model(name, language):
        return DistilBertForTokenClassification.pretrained(name, language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("tokendistilbert")



