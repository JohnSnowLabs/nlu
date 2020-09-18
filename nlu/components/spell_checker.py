from nlu.pipe_components import SparkNLUComponent

class SpellChecker(SparkNLUComponent):
    def __init__(self,component_name='context_spell', language = 'en', component_type='spell_checker', get_default=True, model = None,sparknlp_reference='',dataset='' ):
        if component_name == 'context' or component_name == 'norvig' or component_name == 'symmetric':
            component_name = component_name+'_spell'
        if dataset != '':component_name = dataset+'_spell'
        SparkNLUComponent.__init__(self,component_name,component_type)


        if model != None : self.model = model
        else :
            if 'context' in component_name:
                from nlu import ContextSpellChecker
                if get_default : self.model =  ContextSpellChecker.get_default_model()
                else : self.model = ContextSpellChecker.get_pretrained_model(sparknlp_reference, language)
            elif 'norvig' in component_name:
                from nlu import NorvigSpellChecker
                if get_default : self.model =  NorvigSpellChecker.get_default_model()
                else : self.model = NorvigSpellChecker.get_pretrained_model(sparknlp_reference, language)
            elif 'symmetric' in component_name :
                from nlu import SymmetricSpellChecker
                if get_default : self.model = SymmetricSpellChecker.get_default_model()
                else : self.model = SymmetricSpellChecker.get_pretrained_model(sparknlp_reference, language)

