from sparknlp.annotator import *

class SparkNLPDocumentNormalizer:
    @staticmethod
    def get_default_model():
        return DocumentNormalizer() \
            .setInputCols(["document"]) \
            .setOutputCol("normalized_document")\
            .setAction('clean')

