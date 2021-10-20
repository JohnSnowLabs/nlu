from sparknlp.annotator import XlmRoBertaForTokenClassification

class TokenXlmRoBerta:
    @staticmethod
    def get_default_model():
        return XlmRoBertaForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return XlmRoBertaForTokenClassification.pretrained(name, language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")



