from sparknlp.annotator import RoBertaForTokenClassification

class TokenRoBerta:
    @staticmethod
    def get_default_model():
        return RoBertaForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return RoBertaForTokenClassification.pretrained(name, language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")



