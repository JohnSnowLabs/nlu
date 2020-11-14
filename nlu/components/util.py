from nlu.pipe_components import SparkNLUComponent, NLUComponent

class Util(SparkNLUComponent):

    def __init__(self, annotator_class='document_assembler', component_type='util', model = None):
        # super(Tokenizer,self).__init__(annotator_class = annotator_class, component_type = component_type)
        if annotator_class == 'ner_converter':
            annotator_class = 'ner_to_chunk_converter'
        if model != None : self.model = model
        else :
            if annotator_class == 'document_assembler':
                from nlu import SparkNlpDocumentAssembler
                self.model =  SparkNlpDocumentAssembler.get_default_model()
            elif annotator_class == 'sentence_detector' :
                from nlu import SparkNLPSentenceDetector
                self.model =  SparkNLPSentenceDetector.get_default_model()
            elif annotator_class == 'sentence_detector_deep' :
                from nlu import SparkNLPSentenceDetector
                self.model =  SparkNLPSentenceDetector.get_default_model()
            elif annotator_class == 'ner_to_chunk_converter' :
                from nlu import NerToChunkConverter
                self.model =  NerToChunkConverter.get_default_model()
        SparkNLUComponent.__init__(self, annotator_class, component_type)
