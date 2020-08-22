from nlu.pipe_components import SparkNLUComponent

class Embeddings(SparkNLUComponent):

    def __init__(self,component_name='glove', language ='en', component_type='embedding', get_default=True,model = None, sparknlp_reference =''):
        if 'use' in component_name or 'embed_sentence' in component_name or 'tfhub_use' in sparknlp_reference: component_name = 'use'
        elif 'bert' in component_name and 'albert' not in component_name: component_name='bert'
        elif 'tfhub' in component_name: component_name='use'
        elif 'glove' in component_name : component_name = 'glove'
        elif 'albert' in component_name : component_name = 'albert'
        elif 'xlnet' in component_name : component_name = 'xlnet'

        SparkNLUComponent.__init__(self,component_name,component_type)
        if model != None : self.model = model
        else :
            if 'albert' in component_name :
                from nlu import SparkNLPAlbert
                if get_default: self.model =  SparkNLPAlbert.get_default_model()
                else : self.model = SparkNLPAlbert.get_pretrained_model(sparknlp_reference,language)
            elif 'bert' in component_name  :
                from nlu import SparkNLPBert
                if get_default : self.model =  SparkNLPBert.get_default_model()
                else : self.model = SparkNLPBert.get_pretrained_model(sparknlp_reference,language)
            elif 'elmo' in component_name  :
                from nlu import SparkNLPElmo
                if get_default : self.model = SparkNLPElmo.get_default_model()
                else : self.model =SparkNLPElmo.get_pretrained_model(sparknlp_reference, language)
            elif  'xlnet' in component_name  :
                from nlu import SparkNLPXlnet
                if get_default : self.model = SparkNLPXlnet.get_default_model()
                else : self.model = SparkNLPXlnet.get_pretrained_model(sparknlp_reference, language)
            elif 'use' in component_name   :
                from nlu import SparkNLPUse
                if get_default : self.model = SparkNLPUse.get_default_model()
                else : self.model = SparkNLPUse.get_pretrained_model(sparknlp_reference, language)
            elif 'glove' in component_name   :
                from nlu import Glove
                if component_name == 'glove' : self.model = Glove.get_default_model()
                else :
                    if get_default : self.model = Glove.get_default_model()
                    else :
                        if sparknlp_reference=='glove_840B_300' or  sparknlp_reference=='glove_6B_300':
                            if language=='en': #special case
                                language = 'xx'
                                self.model = Glove.get_pretrained_model(sparknlp_reference, language)
                        else :
                            self.model = Glove.get_pretrained_model(sparknlp_reference, language)
