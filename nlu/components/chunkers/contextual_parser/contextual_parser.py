from sparknlp_jsl.annotator import ContextualParserModel,ContextualParserApproach

class ContextualParser:
    @staticmethod
    def get_default_model():
        return ContextualParserModel() \
            .setInputCols(["document","token"]) \
            .setOutputCol("parsed_context")

    @staticmethod
    def get_trainable_model():
        return ContextualParserApproach() \
            .setInputCols(["document","token"]) \
            .setOutputCol("parsed_context")

