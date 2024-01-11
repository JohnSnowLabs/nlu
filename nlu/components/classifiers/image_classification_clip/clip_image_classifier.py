from sparknlp.annotator import *
from sparknlp.base import *


class ClipZeroShotImageClassifier:
    @staticmethod
    def get_default_model():
        return CLIPForZeroShotClassification \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("classes")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return CLIPForZeroShotClassification \
            .pretrained(name, language, bucket) \
            .setInputCols("image_assembler") \
            .setOutputCol("class")





