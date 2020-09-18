from nlu.pipe_components import SparkNLUComponent

class Matcher(SparkNLUComponent):
    def __init__(self,component_name='date_matcher', language = 'en', component_type='matcher', get_default=True, model = None,sparknlp_reference='',dataset='' ):
        if '_matcher' not in component_name : component_name+='_matcher' 
        SparkNLUComponent.__init__(self,component_name,component_type)


        if model != None : self.model = model
        else :
            if 'text' in component_name:
                from nlu import TextMatcher
                if get_default : self.model =  TextMatcher.get_default_model()
                else : self.model = TextMatcher.get_pretrained_model(sparknlp_reference, language)
            elif 'date' in component_name:
                from nlu import DateMatcher
                if get_default : self.model =  DateMatcher.get_default_model()
            elif 'regex' in component_name :
                from nlu import RegexMatcher
                if get_default : self.model = RegexMatcher.get_default_model()
                else : self.model = RegexMatcher.get_pretrained_model(sparknlp_reference, language)

