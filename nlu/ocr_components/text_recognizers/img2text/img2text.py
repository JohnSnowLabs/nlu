class Img2Text:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import ImageToText
        return ImageToText() \
            .setInputCol("image") \
            .setOutputCol("text") \
            .setOcrParams(["preserve_interword_spaces=1", ])

