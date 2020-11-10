from nlu.pipe_components import SparkNLUComponent

class Normalizer(SparkNLUComponent):
    def __init__(self, annotator_class='normalizer', language='en', component_type='normalizer', get_default=True, nlp_ref='',model=None):
        if model != None :
            self.model = model

        elif annotator_class == 'normalizer':
            from nlu import SparkNLPNormalizer
            if get_default : self.model =  SparkNLPNormalizer.get_default_model()
            else : self.model =  SparkNLPNormalizer.get_pretrained_model(nlp_ref, language) # there is no pretrained API for Normalizer in SparkNLP yet


        SparkNLUComponent.__init__(self, annotator_class, component_type)
