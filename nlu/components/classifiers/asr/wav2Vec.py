from sparknlp.annotator import *


class Wav2Vec:
    @staticmethod
    def get_default_model():
        return Wav2Vec2ForCTC.pretrained() \
            .setInputCols("audio_assembler") \
            .setOutputCol("text")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return Wav2Vec2ForCTC.pretrained(name, language, bucket) \
            .setInputCols("audio_assembler") \
            .setOutputCol("text")
