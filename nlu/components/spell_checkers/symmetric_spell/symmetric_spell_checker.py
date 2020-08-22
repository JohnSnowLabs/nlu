from sparknlp.annotator import *

class SymmetricSpellChecker:
    @staticmethod
    def get_default_model():
        return SymmetricDeleteModel.pretrained() \
            .setInputCols(["token"]) \
            .setOutputCol("spell")

    @staticmethod
    def get_pretrained_model(name, language):
        return SymmetricDeleteModel.pretrained(name, language) \
            .setInputCols(["token"]) \
            .setOutputCol("spell")

    @staticmethod
    def get_default_trainable_model():
        return SymmetricDeleteApproach() \
            .setInputCols(["token"]) \
            .setOutputCol("spell")
