from nlu.pipe.pipe_components import SparkNLUComponent

class NLUSentenceDetector(SparkNLUComponent):
    def __init__(self, annotator_class='sentence_detector', language='en', component_type='sentence_detector', get_default=True, model = None, nlp_ref='', nlu_ref='', trainable=False, is_licensed=False,lang='en',loaded_from_pretrained_pipe=False):
        if annotator_class == 'sentence_detector' and 'pragmatic' not in nlu_ref: annotator_class = 'deep_sentence_detector' #default
        else : annotator_class = 'pragmatic_sentence_detector'
        if model != None : self.model = model
        else:
            if annotator_class == 'deep_sentence_detector' or 'ner_dl' in nlp_ref:
                from nlu import SentenceDetectorDeep
                if trainable : self.model = SentenceDetectorDeep.get_trainable_model()
                elif get_default : self.model =  SentenceDetectorDeep.get_default_model()
                else : self.model = SentenceDetectorDeep.get_pretrained_model(nlp_ref,language)
            elif annotator_class == 'pragmatic_sentence_detector' :
                from nlu import PragmaticSentenceDetector
                if get_default : self.model =  PragmaticSentenceDetector.get_default_model()
        SparkNLUComponent.__init__(self, annotator_class, component_type, nlu_ref, lang,loaded_from_pretrained_pipe )
