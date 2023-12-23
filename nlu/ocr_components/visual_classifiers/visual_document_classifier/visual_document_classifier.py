class VisualDocClassifier:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import VisualDocumentClassifier
        return VisualDocumentClassifier.pretrained("visual_document_classifier_tobacco3482", "en", "clinical/ocr") \
            .setMaxSentenceLength(128) \
            .setInputCol("hocr") \
            .setLabelCol("prediction") \
            .setConfidenceCol("conf")
