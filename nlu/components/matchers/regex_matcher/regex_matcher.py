import sparknlp

class RegexMatcher:
    @staticmethod
    def get_default_model():
        return   sparknlp.annotator.RegexMatcherModel() \
            .setStrategy("MATCH_ALL") \
            .setInputCols("document") \
            .setOutputCol("entity") \


    @staticmethod
    def get_pretrained_model(name, language):
        return   RegexMatcher.get_default_model() 
        # sparknlp.annotator.TextMatcherModel.pretrained(name,language) \
        #     .setInputCols("document") \
        #     .setOutputCol("entity") \


