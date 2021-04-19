from sparknlp.base import *

class SparkNlpDocumentAssembler:
    @staticmethod
    def get_default_model():
        return DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document") \
            .setCleanupMode("shrink")


