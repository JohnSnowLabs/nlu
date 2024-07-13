class PDF2Image:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import PdfToImage
        return PdfToImage() \
            .setPartitionNum(12) \
            .setInputCol("content") \
            .setOutputCol("ocr_image") \
            .setKeepInput(False)
