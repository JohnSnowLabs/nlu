from nlu.pipe_components import SparkNLUComponent, NLUComponent

class NLUSentenceDetector(SparkNLUComponent):
# always gets deep
    def __init__(self, annotator_class='sentence_detector', language='en', component_type='sentence_detector', get_default=True, model = None, nlp_ref='', nlu_ref=''):
        if annotator_class == 'sentence_detector' : annotator_class = 'deep_sentence_detector' #default
        SparkNLUComponent.__init__(self, annotator_class, component_type)
        if model != None : self.model = model
        else:
            if annotator_class == 'deep_sentence_detector' or 'ner_dl' in nlp_ref:
                from nlu import SentenDetectorDeep # wierd import issue ... does not work when outside scoped.
                if get_default : self.model =  SentenDetectorDeep.get_default_model()
            elif annotator_class == 'pragmatic_sentence_detector' :
                from nlu import PragmaticSentenceDetector
                if get_default : self.model =  PragmaticSentenceDetector.get_default_model()
