from sparknlp_jsl.annotator import DrugNormalizer

class DrugNorm:
    @staticmethod
    def get_default_model():
        return DrugNormalizer() \
            .setInputCols(["document"]) \
            .setOutputCol("normalized_drugs")

