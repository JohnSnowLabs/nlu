from sparknlp.annotator import *


class Whisper:
    @staticmethod
    def get_default_model():
        return WhisperForCTC.pretrained() \
            .setInputCols("audio_assembler") \
            .setOutputCol("text")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return WhisperForCTC.pretrained(name, language, bucket) \
            .setInputCols("audio_assembler") \
            .setOutputCol("text")