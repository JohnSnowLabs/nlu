
class ContextualParser:
    @staticmethod
    def get_default_model():
        from sparknlp_jsl.annotator import ContextualParserModel
        return ContextualParserModel() \
            .setInputCols(["document","token"]) \
            .setOutputCol("parsed_context")

    @staticmethod
    def get_trainable_model():
        from sparknlp_jsl.annotator import ContextualParserApproach
        return ContextualParserApproach() \
            .setInputCols(["document","token"]) \
            .setOutputCol("parsed_context")

