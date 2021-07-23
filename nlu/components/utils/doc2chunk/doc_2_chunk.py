from sparknlp.base import *


class Doc_2_Chunk:
    @staticmethod
    def get_default_model():
        return Doc2Chunk() \
            .setInputCols("sentence") \
            .setOutputCol("entities")



