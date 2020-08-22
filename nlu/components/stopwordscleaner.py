from nlu import *
from nlu.pipe_components import SparkNLUComponent
from sparknlp.annotator import *

class StopWordsCleaner(SparkNLUComponent):

    def __init__(self,component_name='stopwordcleaner', language='en', component_type='stopwordscleaner', get_default=False,model = None, sparknlp_reference=''):
        SparkNLUComponent.__init__(self,component_name,component_type)
        # component_name = utils.lower_case(component_name) TODO

        if model != None : self.model = model
        else :
            if 'stop' in component_name :
                from nlu import NLUStopWordcleaner
                if get_default : self.model =  NLUStopWordcleaner.get_default_model()
                else : self.model =  NLUStopWordcleaner.get_pretrained_model(sparknlp_reference,language)
