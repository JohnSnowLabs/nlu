
class FewShotAssertionClassifierModel:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import FewShotAssertionClassifierModel

        return FewShotAssertionClassifierModel.pretrained() \
                               .setInputCols(["sentence", "ner_chunk"]) \
                               .setOutputCol("assertion")


    @staticmethod
    def get_pretrained_model(name, language, bucket='clinical/models'):
        from sparknlp_jsl.annotator import FewShotAssertionClassifierModel
        return FewShotAssertionClassifierModel.pretrained(name, language,bucket) \
            .setInputCols(["sentence", "ner_chunk"]) \
            .setOutputCol("assertion")