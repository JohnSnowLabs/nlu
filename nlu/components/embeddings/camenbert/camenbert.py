from sparknlp.annotator import CamemBertEmbeddings


class CamemBert:
    @staticmethod
    def get_default_model():
        return CamemBertEmbeddings.pretrained() \
        .setInputCols("sentence", "token") \
        .setOutputCol("bert")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return CamemBertEmbeddings.pretrained(name,language,bucket) \
            .setInputCols("sentence", "token") \
            .setOutputCol("bert")



