class Image2Hocr:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageToHocr
        return ImageToHocr() \
            .setInputCol("image") \
            .setOutputCol("hocr")
