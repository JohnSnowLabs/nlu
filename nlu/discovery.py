from nlu.namespace import NameSpace
import nlu
all_components_info = nlu.AllComponentsInfo()

class Discoverer:
    def __init__(self):
        ''' Initialize every NLU component info object and provide access to them'''
        self.nlu_info = {}

    # Functionality discovery methods
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
            for nlu_reference in NameSpace.pretrained_pipe_references[lang]:
                print("nlu.load('" + nlu_reference + "') returns Spark NLP Pipeline:" +
                      NameSpace.pretrained_pipe_references[lang][nlu_reference])

            for nlu_reference in NameSpace.pretrained_models_references[lang]:
                print("nlu.load('" + nlu_reference + "') returns Spark NLP Model: " +
                      NameSpace.pretrained_models_references[lang][nlu_reference])

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
        for nlu_reference in nlu.NameSpace.component_alias_references.keys():
            component_type = nlu.NameSpace.component_alias_references[nlu_reference][1][0],  # pipe or model
            print("nlu.load('" + nlu_reference + "') '  returns Spark NLP " + str(component_type) + ': ' +
                  nlu.NameSpace.component_alias_references[nlu_reference][0])

        for lang in nlu.NameSpace.pretrained_pipe_references.keys():
            for nlu_reference in nlu.NameSpace.pretrained_pipe_references[lang]:
                print("nlu.load('" + nlu_reference + "') for lang" + lang + " returns model Spark NLP model:" +
                      nlu.NameSpace.pretrained_pipe_references[lang][nlu_reference])

        for lang in nlu.NameSpace.pretrained_models_references.keys():
            for nlu_reference in nlu.NameSpace.pretrained_models_references[lang]:
                print("nlu.load('" + nlu_reference + "')' for lang" + lang + " returns model Spark NLP model: " +
                      nlu.NameSpace.pretrained_models_references[lang][nlu_reference])


    @staticmethod
    def print_component_types():
        ''' Prints all unique component types in NLU'''
        types = []
        for key, val in nlu.all_components_info.all_components.items(): types.append(val.type)

        types = set(types)
        print("Provided component types in this NLU version are : ")
        for i, type in enumerate(types):
            print(i, '. ', type)


    @staticmethod
    def print_all_model_kinds_for_action(action):
        for lang, lang_models in nlu.NameSpace.pretrained_models_references.items():
            lang_printed = False
            for nlu_reference, nlp_reference in lang_models.items():
                ref_action = nlu_reference.split('.')
                if len(ref_action) > 1:
                    ref_action = ref_action[1]

                if ref_action == action:
                    if not lang_printed:
                        print('For language <' + lang + '> NLU provides the following Models : ')
                        lang_printed = True
                    print("nlu.load('" + nlu_reference + "') returns Spark NLP model " + nlp_reference)


    @staticmethod
    def print_all_model_kinds_for_action_and_lang(lang, action):
        lang_candidates = nlu.NameSpace.pretrained_models_references[lang]
        print("All NLU components for lang ", lang, " that peform action ", action)
        for nlu_reference, nlp_reference in lang_candidates.items():
            ref_action = nlu_reference.split('.')
            if len(ref_action) > 1: ref_action = ref_action[1]
            if ref_action == action:
                print("nlu.load('" + nlu_reference + "') returns Spark NLP model " + nlp_reference)

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
