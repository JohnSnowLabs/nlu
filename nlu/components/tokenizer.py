from nlu.pipe_components import SparkNLUComponent, NLUComponent

class Tokenizer(SparkNLUComponent):

    def __init__(self, annotator_class='default_tokenizer', language='en', component_type='tokenizer', get_default = True, nlp_ref='', nlu_ref='', model=None):

        if 'token' in annotator_class and not 'regex' in annotator_class: annotator_class = 'default_tokenizer'

        if model != None : self.model = model
        else:
            from nlu import DefaultTokenizer
            if get_default : self.model =  DefaultTokenizer.get_default_model()
            else : self.model =  DefaultTokenizer.get_default_model()  # there are no pretrained tokenizrs, only default 1
        SparkNLUComponent.__init__(self, annotator_class, component_type)
