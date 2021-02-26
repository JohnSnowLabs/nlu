from nlu.pipe_components import SparkNLUComponent
class Resolver(SparkNLUComponent):
    def __init__(self, annotator_class='sentence_entity_resolver', language='en', component_type='resolution', get_default=True, model = None, nlp_ref ='', nlu_ref='',trainable=False, is_licensed=False):

        if model != None : self.model = model
        else :
            if annotator_class == 'sentence_entity_resolver':
                # from nlu.components.resolutions.assertion_dl.assertion_dl import AssertionDL
                from nlu.components.resolutions.sentence_entity_resolver.sentence_resolver import SentenceResolver
                if trainable : self.model = SentenceResolver.get_default_trainable_model()
                elif get_default : self.model = SentenceResolver.get_default_model()
                else : self.model = SentenceResolver.get_pretrained_model(nlp_ref, language)

            elif annotator_class == 'chunk_entity_resolver': pass


        SparkNLUComponent.__init__(self, annotator_class, component_type)
