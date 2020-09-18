from nlu.pipe_components import SparkNLUComponent, NLUComponent

class Stemmer(SparkNLUComponent):
    def __init__(self,component_name='stemmer', component_type='stemmer',model = None):
        if model != None : self.model = model
        else :

            SparkNLUComponent.__init__(self,component_name,component_type)
            if component_name == 'stemmer':
                from nlu import SparkNLPStemmer 
                self.model =  SparkNLPStemmer.get_default_model()
