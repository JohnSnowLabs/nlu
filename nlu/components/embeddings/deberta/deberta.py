from sparknlp.annotator import DeBertaEmbeddings


class Deberta:
    @staticmethod
    def get_default_model():
        return DeBertaEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("deberta")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return DeBertaEmbeddings.pretrained(name,language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("deberta")



