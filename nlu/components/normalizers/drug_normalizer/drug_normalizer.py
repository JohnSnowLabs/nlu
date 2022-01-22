

class DrugNorm:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import DrugNormalizer
        return DrugNormalizer() \
            .setInputCols(["document"]) \
            .setOutputCol("normalized_drugs")

