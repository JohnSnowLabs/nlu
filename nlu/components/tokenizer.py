from nlu.pipe_components import SparkNLUComponent, NLUComponent

class Tokenizer(SparkNLUComponent):

    def __init__(self,component_name='default_tokenizer', language='en', component_type='tokenizer', get_default = True,sparknlp_reference=''):
        if 'token' in component_name : component_name = 'default_tokenizer'
        SparkNLUComponent.__init__(self,component_name,component_type)
        if component_name == 'default_tokenizer' or 'token' in component_name:
            from nlu import DefaultTokenizer
            if get_default : self.model =  DefaultTokenizer.get_default_model()
            else : self.model =  DefaultTokenizer.get_default_model()  # there are no pretrained tokenizrs, only default 1
