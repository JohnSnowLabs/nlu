from nlu.pipe_components import SparkNLUComponent
class Asserter(SparkNLUComponent):
    def __init__(self, annotator_class='assertion_dl', language='en', component_type='assertion', get_default=True, model = None, nlp_ref ='', nlu_ref='',trainable=False, is_licensed=False):

        if model != None : self.model = model
        else :
            if annotator_class == 'assertion_dl':
                from nlu.components.assertions.assertion_dl.assertion_dl import AssertionDL
                if trainable : self.model = AssertionDL.get_default_trainable_model()
                elif get_default : self.model = AssertionDL.get_default_model()
                else : self.model = AssertionDL.get_pretrained_model(nlp_ref, language)

            elif annotator_class == 'assertion_log_reg': pass

        SparkNLUComponent.__init__(self, annotator_class, component_type)
