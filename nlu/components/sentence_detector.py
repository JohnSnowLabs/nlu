from nlu.pipe_components import SparkNLUComponent, NLUComponent

class NLUSentenceDetector(SparkNLUComponent):

    def __init__(self,component_name='pragmatic_sentence_detector', language='en', component_type='sentence_detector',  get_default=False,model = None, sparknlp_reference=''):
        if component_name == 'sentence_detector' : component_name = 'deep_sentence_detector' #default
        SparkNLUComponent.__init__(self,component_name,component_type)
        if model != None : self.model = model
        else:
            if component_name == 'deep_sentence_detector' or 'ner_dl' in sparknlp_reference:
                from nlu import SentenDetectorDeep # wierd import issue ... does not work when outside scoped.
                if get_default : self.model =  SentenDetectorDeep.get_default_model()
            elif component_name == 'pragmatic_sentence_detector' :
                from nlu import PragmaticSentenceDetector
                if get_default : self.model =  PragmaticSentenceDetector.get_default_model()
