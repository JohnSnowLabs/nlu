from sparknlp.base import *

class SparkNlpImageAssembler:
    @staticmethod
    def get_default_model():
        return ImageAssembler() \
            .setInputCol("image") \
            .setOutputCol("image_assembler")

