from sparknlp.annotator import *

class WordSegmenter:
    @staticmethod
    def get_default_model():
        return WordSegmenterModel()\
            .setInputCols(["document"]) \
            .setOutputCol("token")


    @staticmethod
    def get_pretrained_model(name, language):
        return WordSegmenterModel.pretrained(name,language) \
            .setInputCols(["document"]) \
            .setOutputCol("token")

    @staticmethod
    def get_default_model_for_lang(language):
        name = WordSegmenter.get_default_word_seg_for_lang(language)

        return WordSegmenterModel.pretrained(name,language) \
            .setInputCols(["document"]) \
            .setOutputCol("token")


    @staticmethod
    def get_default_word_seg_for_lang(language):
        import nlu
        # get default reference
        return nlu.Spellbook.pretrained_models_references[language][language + '.' + 'segment_words']