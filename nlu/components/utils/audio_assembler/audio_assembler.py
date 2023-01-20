from sparknlp import AudioAssembler


class AudioAssembler_:
    @staticmethod
    def get_default_model():
        return AudioAssembler() \
            .setInputCol("audio_content") \
            .setOutputCol("audio_assembler")
