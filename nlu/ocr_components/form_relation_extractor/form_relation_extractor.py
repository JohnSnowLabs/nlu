
class FormRelationExtractor:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import FormRelationExtractor
        return FormRelationExtractor() \
            .setInputCol("text_entity") \
            .setOutputCol("ocr_relations")
