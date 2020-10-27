from nlu.pipe_components import SparkNLUComponent

class Matcher(SparkNLUComponent):
    def __init__(self, annotator_class='date_matcher', language ='en', component_type='matcher', get_default=False,nlp_ref ='',model = None, nlu_ref='', dataset='' ):

        if 'date' in nlp_ref or 'date' in nlu_ref : annotator_class= 'date_matcher'
        elif 'regex' in nlp_ref or 'regex' in nlu_ref : annotator_class= 'regex_matcher'
        elif 'text' in nlp_ref or 'text' in nlu_ref : annotator_class= 'text_matcher'
        elif '_matcher' not in annotator_class: annotator_class= annotator_class  + '_matcher'



        if model != None : self.model = model
        else :
            if 'text' in annotator_class:
                from nlu import TextMatcher
                if get_default : self.model =  TextMatcher.get_default_model()
                else : self.model = TextMatcher.get_pretrained_model(nlu_ref, language)
            elif 'date' in annotator_class:
                from nlu import DateMatcher
                if get_default : self.model =  DateMatcher.get_default_model()
            elif 'regex' in annotator_class :
                from nlu import RegexMatcher
                if get_default : self.model = RegexMatcher.get_default_model()
                else : self.model = RegexMatcher.get_pretrained_model(nlu_ref, language)

        SparkNLUComponent.__init__(self, annotator_class, component_type)
