from sparknlp.annotator import *
from sparknlp.base import *


class VisionEncoderDecoder:
    @staticmethod
    def get_default_model():
        return VisionEncoderDecoderForImageCaptioning \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("caption")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return VisionEncoderDecoderForImageCaptioning \
            .pretrained(name, language, bucket) \
            .setInputCols("image_assembler") \
            .setOutputCol("caption")





