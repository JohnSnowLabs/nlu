from nlu.pipe_components import SparkNLUComponent, NLUComponent

class EmbeddingsChunker(SparkNLUComponent):

    def __init__(self, annotator_class='chunk_embedder', language='en', component_type='embeddings_chunk', get_default = True, nlp_ref='', model=None, nlu_ref=''):
        if model != None : self.model = model
        else : 
            if annotator_class == 'chunk_embedder' :
                from nlu import ChunkEmbedder
                if get_default : self.model =  ChunkEmbedder.get_default_model()
                else : self.model =  ChunkEmbedder.get_default_model()  # there are no pretrained chunkers, only default 1
        SparkNLUComponent.__init__(self, annotator_class, component_type)
