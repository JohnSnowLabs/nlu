from sparknlp.base import *


class Chunk_2_Doc:
    @staticmethod
    def get_default_model():
        return Chunk2Doc() \
            .setInputCols("entities") \
            .setOutputCol("chunk2doc")



