class Img2Text:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageToText
        return ImageToText() \
            .setInputCol("ocr_image") \
            .setOutputCol("text") \
            .setIgnoreResolution(False) \
            .setPageIteratorLevel(PageIteratorLevel.SYMBOL) \
            .setPageSegMode(PageSegmentationMode.SPARSE_TEXT) \
            .setConfidenceThreshold(70)

