from sparknlp.base import MultiDocumentAssembler

class SparkNlpMultiDocumentAssembler:
    @staticmethod
    def get_default_model():
        return MultiDocumentAssembler() \
            .setInputCols(["question", "context"]) \
            .setOutputCols(["document_question", "document_context"])
