from sparknlp.annotator import LongformerForTokenClassification

class TokenLongFormer:
    @staticmethod
    def get_default_model():
        return LongformerForTokenClassification.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("ner")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return LongformerForTokenClassification.pretrained(name, language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("ner")



