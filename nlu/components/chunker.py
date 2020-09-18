from nlu.pipe_components import SparkNLUComponent, NLUComponent

class Chunker(SparkNLUComponent):

    def __init__(self,component_name='default_chunker', language='en', component_type='chunker', get_default = True,sparknlp_reference='', model=None):
        SparkNLUComponent.__init__(self,component_name,component_type)
        if model != None : self.model = model
        else : 
            if component_name == 'default_chunker':
                from nlu import DefaultChunker
                if get_default : self.model =  DefaultChunker.get_default_model()
                else : self.model =  DefaultChunker.get_default_model()  # there are no pretrained chunkers, only default 1
            if component_name == 'ngram':
                from nlu import NGram
                if get_default : self.model =  NGram.get_default_model()
                else : self.model =  NGram.get_default_model()  # there are no pretrained chunkers, only default 1
