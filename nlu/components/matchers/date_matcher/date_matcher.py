import sparknlp

class DateMatcher:
    @staticmethod
    def get_default_model():
        return   sparknlp.annotator.DateMatcher()\
            .setInputCols("document") \
            .setOutputCol("date") \
            .setDateFormat("yyyyMM")




