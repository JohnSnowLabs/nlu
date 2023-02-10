from sparknlp.annotator import *
from sparknlp.base import *


class SwinImageClassifier:
    @staticmethod
    def get_default_model():
        return SwinForImageClassification \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("classes")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return SwinForImageClassification \
            .pretrained(name, language, bucket) \
            .setInputCols("image_assembler") \
            .setOutputCol("class")





