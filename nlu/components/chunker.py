from nlu.pipe_components import SparkNLUComponent, NLUComponent

class Chunker(SparkNLUComponent):

    def __init__(self, annotator_class='default_chunker', language='en', component_type='chunker', get_default = True, nlp_ref='', nlu_ref='',  model=None):
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
        SparkNLUComponent.__init__(self, annotator_class, component_type)
