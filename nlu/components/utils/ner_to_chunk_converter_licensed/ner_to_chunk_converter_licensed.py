from sparknlp_jsl.annotator import NerConverterInternal

class NerToChunkConverterLicensed:
    @staticmethod
    def get_default_model():
        return NerConverterInternal() \
            .setInputCols(["sentence", "token", "ner"]) \
            .setOutputCol("entities") 
