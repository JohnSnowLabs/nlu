from sparknlp.base import TableAssembler

class SparkNlpTableAssembler:
    @staticmethod
    def get_default_model():
        return TableAssembler() \
            .setInputCols('raw_table') \
            .setOutputCol("assembled_table")
