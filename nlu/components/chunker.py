from nlu.pipe.pipe_components import SparkNLUComponent


class Chunker(SparkNLUComponent):

    def __init__(self, annotator_class='default_chunker', language='en', component_type='chunker', get_default = True, nlp_ref='', nlu_ref='',  model=None, lang='en',loaded_from_pretrained_pipe=False,is_licensed=False):
        if model != None : self.model = model
        else : 
            if annotator_class == 'default_chunker':
                from nlu import DefaultChunker
                if get_default : self.model =  DefaultChunker.get_default_model()
                else : self.model =  DefaultChunker.get_default_model()  # there are no pretrained chunkers, only default 1
            if annotator_class == 'ngram':
                from nlu import NGram
                if get_default : self.model =  NGram.get_default_model()
                else : self.model =  NGram.get_default_model()  # there are no pretrained chunkers, only default 1
            if annotator_class == 'contextual_parser':
                from nlu.components.chunkers.contextual_parser.contextual_parser import ContextualParser
                if get_default : self.model =  ContextualParser.get_default_model()
                else: self.model =  ContextualParser.get_default_model()


        SparkNLUComponent.__init__(self, annotator_class, component_type, nlu_ref, nlp_ref, lang,loaded_from_pretrained_pipe , is_licensed)
