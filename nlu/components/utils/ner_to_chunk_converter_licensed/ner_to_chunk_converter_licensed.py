
class NerToChunkConverterLicensed:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import NerConverterInternal
        return NerConverterInternal() \
            .setInputCols(["sentence", "token", "ner"]) \
            .setOutputCol("entities") 
