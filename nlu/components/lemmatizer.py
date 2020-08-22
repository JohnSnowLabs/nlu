from nlu import *
from nlu.pipe_components import SparkNLUComponent
from sparknlp.annotator import *

class Lemmatizer(SparkNLUComponent):

    def __init__(self,component_name='lemma', language='en', component_type='lemmatizer', get_default=False,model = None, sparknlp_reference=''):
        component_name = 'lemmatizer'
        SparkNLUComponent.__init__(self,component_name,component_type)
        # component_name = utils.lower_case(component_name) TODO

        if model != None : self.model = model
        else :
            if 'lemma' in component_name :
                from nlu import SparkNLPLemmatizer
                if get_default : self.model =  SparkNLPLemmatizer.get_default_model()
                else : self.model =  SparkNLPLemmatizer.get_pretrained_model(sparknlp_reference,language)
