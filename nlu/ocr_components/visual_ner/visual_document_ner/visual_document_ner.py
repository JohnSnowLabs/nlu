class VisualDocumentNer:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import VisualDocumentNer
        return VisualDocumentNer()\
            .pretrained("lilt_roberta_funsd_v1", "en", "clinical/ocr")\
            .setInputCols(["text_tokenized", "image"])\
            .setOutputCol("text_entity")
