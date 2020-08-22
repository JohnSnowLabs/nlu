from nlu.pipe_components import SparkNLUComponent, NLUComponent

class EmbeddingsChunker(SparkNLUComponent):

    def __init__(self,component_name='chunk_embedder', language='en', component_type='embeddings_chunk', get_default = True,sparknlp_reference='', model=None):
        SparkNLUComponent.__init__(self,component_name,component_type)
        if model != None : self.model = model
        else : 
            if component_name == 'chunk_embedder' :
                from nlu import ChunkEmbedder
                if get_default : self.model =  ChunkEmbedder.get_default_model()
                else : self.model =  ChunkEmbedder.get_default_model()  # there are no pretrained chunkers, only default 1
