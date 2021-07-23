from nlu.pipe.pipe_components import SparkNLUComponent

class Normalizer(SparkNLUComponent):
    def __init__(self, annotator_class='normalizer', language='en', component_type='normalizer', get_default=True, nlp_ref='',nlu_ref='',model=None, is_licensed=False):
        if model != None :self.model = model
        else :
            if 'norm_document' in nlu_ref : annotator_class = 'document_normalizer'
            elif 'drug' in nlu_ref : annotator_class = 'drug_normalizer'
            elif 'norm' in nlu_ref : annotator_class = 'normalizer'


            if annotator_class == 'normalizer':
                from nlu import SparkNLPNormalizer
                if get_default : self.model =  SparkNLPNormalizer.get_default_model()
                else : self.model =  SparkNLPNormalizer.get_pretrained_model(nlp_ref, language) # there is no pretrained API for Normalizer in SparkNLP yet
            elif annotator_class == 'document_normalizer':
                from nlu import SparkNLPDocumentNormalizer
                if get_default : self.model =  SparkNLPDocumentNormalizer.get_default_model()
                else : self.model =  SparkNLPDocumentNormalizer.get_pretrained_model(nlp_ref, language) # there is no pretrained API for Normalizer in SparkNLP yet
            elif annotator_class == 'drug_normalizer':
                from nlu.components.normalizers.drug_normalizer.drug_normalizer import DrugNorm
                is_licensed = True
                if get_default : self.model =  DrugNorm.get_default_model()


        SparkNLUComponent.__init__(self, annotator_class, component_type)
