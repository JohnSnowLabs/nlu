class PositionFinder:
    @staticmethod
    def get_default_model():
        from sparkocr.transformers import PositionFinder
        return PositionFinder() \
            .setInputCols("ner_chunk_subentity") \
            .setOutputCol("ocr_positions") \
            .setPageMatrixCol("positions")
