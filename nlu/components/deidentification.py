from nlu.pipe_components import SparkNLUComponent
class Deidentification(SparkNLUComponent):
    def __init__(self, annotator_class='deidentifier', language='en', component_type='deidentifier', get_default=False, model = None, nlp_ref ='', nlu_ref='',trainable=False, is_licensed=True):
        annotator_class= 'deidentifier'
        if model != None : self.model = model
        else :
            if annotator_class == 'deidentifier':
                from nlu.components.deidentifiers.deidentifier.deidentifier import Deidentifier
                if get_default : self.model = Deidentifier.get_default_model()
                else : self.model = Deidentifier.get_pretrained_model(nlp_ref, language)

        print('model')
        SparkNLUComponent.__init__(self, annotator_class, component_type)
