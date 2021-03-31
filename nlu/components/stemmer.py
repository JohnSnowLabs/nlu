from nlu.pipe.pipe_components import SparkNLUComponent


class Stemmer(SparkNLUComponent):
    def __init__(self, annotator_class='stemmer', component_type='stemmer', model = None, nlu_ref ='', nlp_ref=''):
        if model != None : self.model = model
        else :
            if annotator_class == 'stemmer':
                from nlu import SparkNLPStemmer 
                self.model =  SparkNLPStemmer.get_default_model()
        SparkNLUComponent.__init__(self, annotator_class, component_type)
