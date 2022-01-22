from sparknlp.annotator import MarianTransformer

class Marian:

    @staticmethod
    def get_default_model():
        return MarianTransformer.pretrained() \
            .setInputCols("document") \
            .setOutputCol("marian")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return MarianTransformer.pretrained(name, language) \
            .setInputCols("document") \
            .setOutputCol("marian")



