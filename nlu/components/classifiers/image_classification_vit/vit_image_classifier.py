from sparknlp.annotator import *
class VitImageClassifier:
    @staticmethod
    def get_default_model():
        return ViTForImageClassification \
            .pretrained() \
            .setInputCols("image_assembler") \
            .setOutputCol("class")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return ViTForImageClassification \
            .pretrained(name, language, bucket) \
            .setInputCols("image_assembler") \
            .setOutputCol("class")




