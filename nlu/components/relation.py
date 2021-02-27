from nlu.pipe_components import SparkNLUComponent
class Relation(SparkNLUComponent):
    def __init__(self, annotator_class='sentence_entity_resolver', language='en', component_type='relation_extractor', get_default=True, model = None, nlp_ref ='', nlu_ref='',trainable=False, is_licensed=False):

        if model != None : self.model = model
        else :
            if annotator_class == 'sentence_entity_resolver':
                from nlu.components.relation_extractors.relation_extractor.relation_extractor import RelationExtraction

                if trainable : self.model = RelationExtraction.get_default_trainable_model()
                else : self.model = RelationExtraction.get_pretrained_model(nlp_ref, language,'clinical/models')

            elif annotator_class == 'chunk_entity_resolver': pass


        SparkNLUComponent.__init__(self, annotator_class, component_type)
