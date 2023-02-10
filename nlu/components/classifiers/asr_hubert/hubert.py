from sparknlp.annotator import *


class Hubert:
    @staticmethod
    def get_default_model():
        return HubertForCTC.pretrained() \
            .setInputCols("audio_assembler") \
            .setOutputCol("text")

    @staticmethod
    def get_pretrained_model(name, language, bucket=None):
        return HubertForCTC.pretrained(name, language, bucket) \
            .setInputCols("audio_assembler") \
            .setOutputCol("text")
