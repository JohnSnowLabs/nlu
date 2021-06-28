from nlu.pipe.pipe_components import SparkNLUComponent
class Relation(SparkNLUComponent):
    def __init__(self, annotator_class='relation_extractor', lang='en', component_type='relation_extractor', get_default=True, model = None, nlp_ref ='', nlu_ref='', trainable=False, is_licensed=False,loaded_from_pretrained_pipe=False):

        if 're_' in nlp_ref  : annotator_class = 'relation_extractor'
        if 'redl' in nlp_ref : annotator_class = 'relation_extractor_dl'

        if model != None : self.model = model
        else :
            if annotator_class == 'relation_extractor':
                from nlu.components.relation_extractors.relation_extractor.relation_extractor import RelationExtraction
                if trainable : self.model = RelationExtraction.get_default_trainable_model()
                else : self.model = RelationExtraction.get_pretrained_model(nlp_ref, lang, 'clinical/models')

            elif annotator_class == 'relation_extractor_dl':
                from nlu.components.relation_extractors.relation_extractor_dl.relation_extractor_dl import RelationExtractionDL
                # if trainable : self.model = RelationExtractionDL.get_default_trainable_model()
                self.model = RelationExtractionDL.get_pretrained_model(nlp_ref, lang, 'clinical/models')


        SparkNLUComponent.__init__(self, annotator_class, component_type, nlu_ref, nlp_ref, lang,loaded_from_pretrained_pipe , is_licensed)
