from sparknlp.annotator import LongformerEmbeddings

class Longformer:
    @staticmethod
    def get_default_model():
        return LongformerEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("longformer")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return LongformerEmbeddings.pretrained(name, language) \
            .setInputCols("sentence", "token") \
            .setOutputCol("longformer")



