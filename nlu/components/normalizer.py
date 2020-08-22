from nlu.pipe_components import SparkNLUComponent

class Normalizer(SparkNLUComponent):
    def __init__(self,component_name='normalizer', language='en', component_type='normalizer', get_default=True,sparknlp_reference=''):
        SparkNLUComponent.__init__(self,component_name,component_type)
        if component_name == 'normalizer':
            from nlu import SparkNLPNormalizer
            if get_default : self.model =  SparkNLPNormalizer.get_default_model()
            else : self.model =  SparkNLPNormalizer.get_pretrained_model(sparknlp_reference, language) # there is no pretrained API for Normalizer in SparkNLP yet