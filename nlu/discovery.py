from nlu.spellbook import Spellbook
import nlu
from nlu.info import AllComponentsInfo
all_components_info = AllComponentsInfo()

class Discoverer:
    """Various methods that help discover nlu_refs and functionality"""
    def __init__(self):
        ''' Initialize every NLU component_to_resolve info object and provide access to them'''
        self.nlu_info = {}

    @staticmethod
    def get_components(m_type='', include_pipes=False, lang='', licensed=False, get_all=False,include_aliases=True):
        """Filter all NLU components

        m_type : Component/Model type to filter for
        include_pipes : Weather to include pipelines in the result or not
        lang : Which languages to include. By default lang='' will get every lang
        licensed : Wether to include licensed models or not
        get_all: If set to true, will ignore other params and gets EVERY NLU_ref from defined name spaces
        """
        nlu_refs_of_type = []
        model_universe = nlu.Spellbook.pretrained_models_references
        for lang_, models in model_universe.items():
            if lang != '' :
                if lang_!= lang : continue
            for nlu_ref, nlp_ref in model_universe[lang_].items():
                if m_type in nlu_ref or get_all: nlu_refs_of_type.append(nlu_ref)

        if include_pipes :
            model_universe = nlu.Spellbook.pretrained_pipe_references
            for lang_, models in model_universe.items():
                if lang != '':
                    if lang_!= lang : continue
                for nlu_ref, nlp_ref in model_universe[lang_].items():
                    if m_type in nlu_ref or get_all: nlu_refs_of_type.append(nlu_ref)
        if include_aliases :
            model_universe = nlu.Spellbook.component_alias_references
            for nlu_ref, nlp_ref in model_universe.items():
                if m_type in nlu_ref or get_all: nlu_refs_of_type.append(nlu_ref)

        if licensed:
            model_universe = nlu.Spellbook.pretrained_healthcare_model_references
            for lang_, models in model_universe.items():
                if lang != '':
                    if lang_!= lang : continue
                for nlu_ref, nlp_ref in model_universe[lang_].items():
                    if m_type in nlu_ref or get_all: nlu_refs_of_type.append(nlu_ref)

        return list(set(nlu_refs_of_type))





    @staticmethod
    def print_all_languages():
        ''' Print all languages which are available in NLU Spark NLP pointer '''
        print('Languages available in NLU : \n ')
        for lang in all_components_info.all_languages: print(lang)



    @staticmethod
    def print_all_nlu_components_for_lang(lang='en', c_type='classifier'):
        #todo parse for lang
        '''Print all NLU components available for a language Spark NLP pointer'''
        if lang in all_components_info.all_languages:
            # print("All Pipelines for language"+ lang+ "\n"+)
            for nlu_reference in Spellbook.pretrained_pipe_references[lang]:
                print("nlu.load('" + nlu_reference + "') returns Spark NLP Pipeline:" +
                      Spellbook.pretrained_pipe_references[lang][nlu_reference])

            for nlu_reference in Spellbook.pretrained_models_references[lang]:
                print("nlu.load('" + nlu_reference + "') returns Spark NLP Model: " +
                      Spellbook.pretrained_models_references[lang][nlu_reference])

        else:
            print(
                "Language " + lang + " Does not exsist in NLU. Please check the docs or nlu.print_all_languages() for supported language references")


    @staticmethod
    def print_components(lang='', action=''):
        '''
        Print every single NLU reference for models and pipeliens and their Spark NLP pointer
        :param lang: Language requirements for the components filterd. See nlu.languages() for supported languages
        :param action: Components that will be filterd.
        :return: None. This method will print its results.
        '''
        if lang != '' and action == '':
            nlu.Discoverer().print_all_nlu_components_for_lang(lang)
            return

        if lang != '' and action != '':
            nlu.Discoverer().print_all_model_kinds_for_action_and_lang(lang, action)
            return

        if lang == '' and action != '':
            nlu.Discoverer().print_all_model_kinds_for_action(action)
            return

        # Print entire Namespace below
        for nlu_reference in nlu.Spellbook.component_alias_references.keys():
            component_type = nlu.Spellbook.component_alias_references[nlu_reference][1][0],  # component_list or model_anno_obj
            print("nlu.load('" + nlu_reference + "') '  returns Spark NLP " + str(component_type) + ': ' +
                  nlu.Spellbook.component_alias_references[nlu_reference][0])

        for lang in nlu.Spellbook.pretrained_pipe_references.keys():
            for nlu_reference in nlu.Spellbook.pretrained_pipe_references[lang]:
                print("nlu.load('" + nlu_reference + "') for lang" + lang + " returns model_anno_obj Spark NLP model_anno_obj:" +
                      nlu.Spellbook.pretrained_pipe_references[lang][nlu_reference])

        for lang in nlu.Spellbook.pretrained_models_references.keys():
            for nlu_reference in nlu.Spellbook.pretrained_models_references[lang]:
                print("nlu.load('" + nlu_reference + "')' for lang" + lang + " returns model_anno_obj Spark NLP model_anno_obj: " +
                      nlu.Spellbook.pretrained_models_references[lang][nlu_reference])


    @staticmethod
    def print_component_types():
        ''' Prints all unique component_to_resolve types in NLU'''
        types = []
        for key, val in nlu.all_components_info.all_components.items(): types.append(val.type)

        types = set(types)
        print("Provided component_to_resolve types in this NLU version are : ")
        for i, type in enumerate(types):
            print(i, '. ', type)


    @staticmethod
    def print_all_model_kinds_for_action(action):
        for lang, lang_models in nlu.Spellbook.pretrained_models_references.items():
            lang_printed = False
            for nlu_reference, nlp_reference in lang_models.items():
                ref_action = nlu_reference.split('.')
                if len(ref_action) > 1:
                    ref_action = ref_action[1]

                if ref_action == action:
                    if not lang_printed:
                        print('For language <' + lang + '> NLU provides the following Models : ')
                        lang_printed = True
                    print("nlu.load('" + nlu_reference + "') returns Spark NLP model_anno_obj " + nlp_reference)


    @staticmethod
    def print_all_model_kinds_for_action_and_lang(lang, action):
        lang_candidates = nlu.Spellbook.pretrained_models_references[lang]
        print("All NLU components for lang ", lang, " that peform action ", action)
        for nlu_reference, nlp_reference in lang_candidates.items():
            ref_action = nlu_reference.split('.')
            if len(ref_action) > 1: ref_action = ref_action[1]
            if ref_action == action:
                print("nlu.load('" + nlu_reference + "') returns Spark NLP model_anno_obj " + nlp_reference)

    @staticmethod
    def print_trainable_components():
        '''
        # todo update
        Print every trainable Algorithm/Model
        :return: None
        '''
        i = 1
        print('The following models can be trained with a dataset that provides a label column and matching dataset')
        for name, infos in nlu.all_components_info.all_components.items() :
            if infos.trainable == True :
                print(f' {i}. {name}')
                i+=1
