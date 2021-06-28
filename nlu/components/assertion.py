from nlu.pipe.pipe_components import SparkNLUComponent
class Asserter(SparkNLUComponent):
    def __init__(self, annotator_class='assertion_dl', lang='en', component_type='assertion', get_default=True, model = None, nlp_ref ='', nlu_ref='', trainable=False, is_licensed=False, loaded_from_pretrained_pipe=False):

        if model != None : self.model = model
        else :
            if annotator_class == 'assertion_dl':
                from nlu.components.assertions.assertion_dl.assertion_dl import AssertionDL
                if trainable : self.model = AssertionDL.get_default_trainable_model()
                elif get_default : self.model = AssertionDL.get_default_model()
                else : self.model = AssertionDL.get_pretrained_model(nlp_ref, lang)

            elif annotator_class == 'assertion_log_reg':
                from nlu.components.assertions.assertion_log_reg.assertion_log_reg import AssertionLogReg
                if trainable : self.model = AssertionLogReg.get_default_trainable_model()
                elif get_default : self.model = AssertionLogReg.get_default_model()
                else : self.model = AssertionLogReg\
                    .get_pretrained_model(nlp_ref, lang)


        SparkNLUComponent.__init__(self, annotator_class, component_type, nlu_ref, nlp_ref, lang,loaded_from_pretrained_pipe , is_licensed)
