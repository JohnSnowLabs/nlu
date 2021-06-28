from nlu.pipe.pipe_components import SparkNLUComponent

class StopWordsCleaner(SparkNLUComponent):

    def __init__(self, annotator_class='stopwordcleaner', lang='en', component_type='stopwordscleaner', get_default=False, model = None, nlp_ref='', nlu_ref='',loaded_from_pretrained_pipe=False,is_licensed=False):

        if model != None : self.model = model
        else :
            if 'stop' in annotator_class :
                from nlu import NLUStopWordcleaner
                if get_default : self.model =  NLUStopWordcleaner.get_default_model()
                else : self.model =  NLUStopWordcleaner.get_pretrained_model(nlp_ref, lang)
        SparkNLUComponent.__init__(self, annotator_class, component_type, nlu_ref, nlp_ref, lang,loaded_from_pretrained_pipe , is_licensed)
