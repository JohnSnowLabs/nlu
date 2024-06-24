class Image2PDF:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageToPdf
        return ImageToPdf() \
            .setInputCol("image_with_regions") \
            .setOutputCol("content")
