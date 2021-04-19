from nlu.pipe.pipe_components import SparkNLUComponent

class Seq2Seq(SparkNLUComponent):

    def __init__(self, annotator_class='t5', language ='en', component_type='seq2seq', get_default=True, model = None, nlp_ref ='', nlu_ref ='',dataset='', configs='', is_licensed=False):
        if 't5' in nlu_ref or 't5' in nlp_ref: annotator_class = 't5'
        elif 'marian' in nlu_ref or 'marian' in nlp_ref: annotator_class = 'marian'
        elif 'translate_to' in nlu_ref or 'translate_to' in nlp_ref or 'translate_to' in annotator_class: annotator_class = 'marian'


        if model != None : self.model = model
        else :
            if 't5' in annotator_class :
                from nlu import T5
                if is_licensed : self.model = T5.get_pretrained_model(nlp_ref, language, bucket='clinical/models')
                elif get_default: self.model =  T5.get_default_model()
                elif configs !='' : self.model = T5.get_preconfigured_model(nlp_ref,language,configs)
                else : self.model = T5.get_pretrained_model(nlp_ref, language)

            elif 'marian' in annotator_class  :
                from nlu import Marian
                if get_default : self.model =  Marian.get_default_model()
                else : self.model = Marian.get_pretrained_model(nlp_ref, language)
        SparkNLUComponent.__init__(self, annotator_class, component_type)
