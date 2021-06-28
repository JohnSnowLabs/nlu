from nlu.pipe.pipe_components import SparkNLUComponent


class Embeddings(SparkNLUComponent):

    def __init__(self, annotator_class='glove', lang ='en', component_type='embedding', get_default=True, model = None, nlp_ref ='', nlu_ref ='', is_licensed=False, resolution_ref='',loaded_from_pretrained_pipe=False,do_ref_checks=True ):
        if do_ref_checks:
            if 'use' in nlu_ref or 'tfhub_use' in nlp_ref: annotator_class = 'use'
            # first check for sentence then token embeddings.

            elif 'xlm'     in nlu_ref or 'xlm' in nlp_ref : annotator_class = 'xlm'
            elif 'roberta' in nlu_ref or 'roberta' in nlp_ref : annotator_class = 'roberta'
            elif 'distil'  in nlu_ref or 'distil' in nlp_ref : annotator_class = 'distil_bert'


            elif 'bert' in nlp_ref and 'albert' not in nlp_ref and 'sent' in nlp_ref : annotator_class= 'sentence_bert'
            elif 'bert' in nlu_ref and 'albert' not in nlu_ref and 'sent' in nlu_ref : annotator_class= 'sentence_bert'

            elif 'elmo' in nlp_ref  : annotator_class= 'elmo'
            elif 'elmo' in nlu_ref  : annotator_class= 'elmo'


            elif 'electra' in nlp_ref and 'sent' in nlp_ref : annotator_class= 'sentence_bert'
            elif 'electra' in nlu_ref and 'sent' in nlu_ref : annotator_class= 'sentence_bert'

            elif 'bert' in nlu_ref and 'albert' not in nlu_ref: annotator_class= 'bert'
            elif 'bert' in nlp_ref and 'albert' not in nlp_ref: annotator_class= 'bert'

            elif 'electra' in nlu_ref or 'electra' in nlp_ref:  annotator_class= 'bert'
            elif 'labse' in nlu_ref   or 'labse' in nlp_ref:  annotator_class= 'sentence_bert'

            elif 'tfhub' in nlu_ref or 'tfhub' in nlp_ref: annotator_class= 'use'
            elif 'glove' in nlu_ref or 'glove' in nlp_ref : annotator_class = 'glove'
            elif 'cc_300d' in nlu_ref or 'cc_300d' in nlp_ref : annotator_class = 'glove'

            elif 'albert' in nlu_ref or 'albert' in nlp_ref : annotator_class = 'albert'
            elif 'xlnet' in nlu_ref or 'xlnet' in nlp_ref : annotator_class = 'xlnet'

            # Default component models for nlu actions that dont specify a particular model
            elif 'embed_sentence' in nlu_ref : annotator_class = 'glove'
            elif 'embed' in nlu_ref          : annotator_class = 'glove'

        if model != None : self.model = model
        else :

            # Check if this lang has embeddings, if NOT set to multi lang xx!
            multi_lang_embeds = ['th']
            if lang in multi_lang_embeds : lang ='xx'

            if 'xlm' in annotator_class :
                from nlu import XLM
                if get_default: self.model =  XLM.get_default_model()
                else : self.model = XLM.get_pretrained_model(nlp_ref, lang)

            elif 'roberta' in annotator_class :
                from nlu import Roberta
                if get_default: self.model =  Roberta.get_default_model()
                else : self.model = Roberta.get_pretrained_model(nlp_ref, lang)


            elif 'distil_bert' in annotator_class :
                from nlu import DistilBert
                if get_default: self.model =  DistilBert.get_default_model()
                else : self.model = DistilBert.get_pretrained_model(nlp_ref, lang)

            elif 'albert' in annotator_class :
                from nlu import SparkNLPAlbert
                if get_default: self.model =  SparkNLPAlbert.get_default_model()
                else : self.model = SparkNLPAlbert.get_pretrained_model(nlp_ref, lang)
            elif 'bert' in annotator_class and 'sent' in annotator_class  :
                from nlu import BertSentence
                if get_default : self.model =  BertSentence.get_default_model()
                elif is_licensed : self.model = BertSentence.get_pretrained_model(nlp_ref, lang,'clinical/models' )
                else : self.model = BertSentence.get_pretrained_model(nlp_ref, lang)
            elif 'electra' in annotator_class and 'sent' in annotator_class  :
                from nlu import BertSentence
                if get_default : self.model =  BertSentence.get_default_model()
                elif is_licensed : self.model = BertSentence.get_pretrained_model(nlp_ref, lang,'clinical/models' )
                else : self.model = BertSentence.get_pretrained_model(nlp_ref, lang)
            elif 'bert' in annotator_class :
                from nlu import SparkNLPBert
                if get_default : self.model =  SparkNLPBert.get_default_model()
                elif is_licensed : self.model = SparkNLPBert.get_pretrained_model(nlp_ref, lang,'clinical/models' )
                else : self.model = SparkNLPBert.get_pretrained_model(nlp_ref, lang)
            elif 'elmo' in annotator_class  :
                from nlu import SparkNLPElmo
                if get_default : self.model = SparkNLPElmo.get_default_model()
                else : self.model =SparkNLPElmo.get_pretrained_model(nlp_ref, lang)
            elif  'xlnet' in annotator_class  :
                from nlu import SparkNLPXlnet
                if get_default : self.model = SparkNLPXlnet.get_default_model()
                else : self.model = SparkNLPXlnet.get_pretrained_model(nlp_ref, lang)
            elif 'use' in annotator_class   :
                from nlu import SparkNLPUse
                if get_default : self.model = SparkNLPUse.get_default_model()
                else : self.model = SparkNLPUse.get_pretrained_model(nlp_ref, lang)
            elif 'glove' in annotator_class   :
                from nlu import Glove
                if annotator_class == 'glove' and get_default==True: self.model = Glove.get_default_model()
                else :
                    if get_default : self.model = Glove.get_default_model()
                    elif is_licensed : self.model = Glove.get_pretrained_model(nlp_ref,lang,'clinical/models')
                    else :
                        if nlp_ref == 'glove_840B_300' or  nlp_ref== 'glove_6B_300':
                            # if lang=='en' and nlp_ref=='glove_6B_300': #special case
                            lang = 'xx' # For these particular Glove embeddings, anyreference to them is actually the reference to the multilingual onces
                            self.model = Glove.get_pretrained_model(nlp_ref, lang)
                        else :
                            self.model = Glove.get_pretrained_model(nlp_ref, lang)

        SparkNLUComponent.__init__(self, annotator_class, component_type,nlu_ref,nlp_ref,lang)
