from nlu.pipe.pipe_components import SparkNLUComponent

class SpellChecker(SparkNLUComponent):
    def __init__(self, annotator_class='context_spell', language ='en', component_type='spell_checker', get_default=True, model = None, nlp_ref='', dataset='', nlu_ref ='', is_licensed=False):
        if annotator_class == 'context' or annotator_class == 'norvig' or annotator_class == 'symmetric':
            annotator_class = annotator_class + '_spell'
        if dataset != '':annotator_class = dataset + '_spell'
        if 'spellcheck_dl' in nlp_ref : annotator_class ='context_spell'
        if 'spell.med' in nlu_ref : annotator_class ='context'
        if 'spell.clinical' in nlu_ref : annotator_class ='context'
        if '.med' in nlu_ref : annotator_class ='context'

        if model != None : self.model = model
        else :
            if 'context' in annotator_class:
                from nlu import ContextSpellChecker
                if is_licensed : self.model =  ContextSpellChecker.get_pretrained_model(nlp_ref, language,bucket='clinical/models')
                elif get_default : self.model =  ContextSpellChecker.get_default_model()
                else : self.model = ContextSpellChecker.get_pretrained_model(nlp_ref, language)
            elif 'norvig' in annotator_class:
                from nlu import NorvigSpellChecker
                if get_default : self.model =  NorvigSpellChecker.get_default_model()
                else : self.model = NorvigSpellChecker.get_pretrained_model(nlp_ref, language)
            elif 'symmetric' in annotator_class :
                from nlu import SymmetricSpellChecker
                if get_default : self.model = SymmetricSpellChecker.get_default_model()
                else : self.model = SymmetricSpellChecker.get_pretrained_model(nlp_ref, language)

        SparkNLUComponent.__init__(self, annotator_class, component_type)
