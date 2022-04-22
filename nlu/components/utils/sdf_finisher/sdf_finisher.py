from sparknlp.base import Finisher



class SdfFinisher:
    @staticmethod
    def get_default_model():
        return Finisher() \
            .setInputCol("text") \
            .setOutputCol("document")

