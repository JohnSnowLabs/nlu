from sparknlp.annotator import Chunk2Doc

class Chunk_2_Doc:
    @staticmethod
    def get_default_model():
        return Chunk2Doc() \
            .setInputCols("entities") \
            .setOutputCol("chunk2doc")



