from nlu.pipe.pipe_components import SparkNLUComponent
class Deidentification(SparkNLUComponent):
    def __init__(self, annotator_class='deidentifier', lang='en', component_type='deidentifier', get_default=False, model = None, nlp_ref ='', nlu_ref='', trainable=False, is_licensed=True,loaded_from_pretrained_pipe=False):
        annotator_class= 'deidentifier'
        if model != None : self.model = model
        else :
            if annotator_class == 'deidentifier':
                from nlu.components.deidentifiers.deidentifier.deidentifier import Deidentifier
                if get_default : self.model = Deidentifier.get_default_model()
                else : self.model = Deidentifier.get_pretrained_model(nlp_ref, lang)

        SparkNLUComponent.__init__(self, annotator_class, component_type, nlu_ref, nlp_ref, lang,loaded_from_pretrained_pipe , is_licensed)
