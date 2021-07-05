from nlu.pipe.pipe_components import SparkNLUComponent

class Util(SparkNLUComponent):

    def __init__(self, annotator_class='document_assembler', component_type='util', model = None, loaded_from_pretrained_pipe = False, nlu_ref='',nlp_ref='',lang='en',is_licensed=False):
        # super(Tokenizer,self).__init__(annotator_class = annotator_class, component_type = component_type)
        if annotator_class == 'ner_converter':
            annotator_class = 'ner_to_chunk_converter'
        if model != None : self.model = model
        else :
            if annotator_class == 'document_assembler':
                from nlu import SparkNlpDocumentAssembler
                self.model =  SparkNlpDocumentAssembler.get_default_model()
            elif annotator_class == 'deep_sentence_detector' :
                from nlu import SentenceDetectorDeep
                self.model =  SentenceDetectorDeep.get_default_model()
            elif annotator_class == 'sentence_detector' :
                from nlu import SparkNLPSentenceDetector
                self.model =  SparkNLPSentenceDetector.get_default_model()
            elif annotator_class == 'ner_to_chunk_converter' :
                from nlu import NerToChunkConverter
                self.model =  NerToChunkConverter.get_default_model()
            elif annotator_class == 'sentence_embeddings':
                from nlu import SparkNLPSentenceEmbeddings
                self.model = SparkNLPSentenceEmbeddings.get_default_model()
            elif annotator_class == 'feature_assembler':
                from nlu.components.utils.feature_assembler.feature_assembler import SparkNLPFeatureAssembler
                self.model = SparkNLPFeatureAssembler.get_default_model()
            elif annotator_class == 'ner_to_chunk_converter_licensed':
                from nlu.components.utils.ner_to_chunk_converter_licensed.ner_to_chunk_converter_licensed import NerToChunkConverterLicensed
                self.model = NerToChunkConverterLicensed.get_default_model()
            elif annotator_class == 'chunk_merger':
                from nlu.components.utils.chunk_merger.chunk_merger import ChunkMerger
                self.model = ChunkMerger.get_default_model()
            elif annotator_class == 'doc2chunk':
                from nlu.components.utils.doc2chunk.doc_2_chunk import Doc_2_Chunk
                self.model = Doc_2_Chunk.get_default_model()


        SparkNLUComponent.__init__(self, annotator_class, component_type, nlu_ref, lang,loaded_from_pretrained_pipe )
