from nlu.pipe.pipe_component import SparkNLUComponent


class Stemmer(SparkNLUComponent):
    def __init__(self, annotator_class='stemmer', component_type='stemmer', model = None, nlu_ref ='', nlp_ref='',loaded_from_pretrained_pipe=False):
        if model != None : self.model = model
        else :
            if annotator_class == 'stemmer':
                from nlu import SparkNLPStemmer 
                self.model =  SparkNLPStemmer.get_default_model()
        SparkNLUComponent.__init__(self, annotator_class, component_type,loaded_from_pretrained_pipe=loaded_from_pretrained_pipe)
