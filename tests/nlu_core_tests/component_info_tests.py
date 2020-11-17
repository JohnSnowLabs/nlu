import unittest
import nlu
import pandas as pd
import numpy as np


class TestComponentInfo(unittest.TestCase):
    def test_list_all_names(self):
        a = nlu.AllComponentsInfo()
        a.list_all_components()
        a.DEBUG_list_all_components()

    def test_print_all_default_components_as_markdown(self):
        print(pd.DataFrame(nlu.NameSpace.component_alias_references).T.to_markdown())

    def test_print_all_models_as_markdown(self):
        rows = []
        for lang in nlu.NameSpace.pretrained_models_references.keys():
            for nlu_reference in nlu.NameSpace.pretrained_models_references[lang].keys():
                rows.append((lang,nlu_reference,nlu.NameSpace.pretrained_models_references[lang][nlu_reference]))
        
        print(pd.DataFrame(rows).to_markdown())
        
    def test_print_all_pipes_as_markdown(self):
        rows = []
        for lang in nlu.NameSpace.pretrained_pipe_references.keys():
            for nlu_reference in nlu.NameSpace.pretrained_pipe_references[lang].keys():
                rows.append((lang,nlu_reference,nlu.NameSpace.pretrained_pipe_references[lang][nlu_reference]))

        print(pd.DataFrame(rows).to_markdown())


    def test_get_count_for_every_component_type(self):
        component_counts = {}
        for lang in nlu.NameSpace.pretrained_models_references:
            for nlu_ref, nlp_ref in nlu.NameSpace.pretrained_models_references[lang].items():
                c_type = nlu_ref.split('.')
                
                if len(c_type) <2 : continue
                c_type = c_type[1]
                
                if c_type not in component_counts : component_counts[c_type] = []
                if (lang+nlp_ref) not in component_counts[c_type] : component_counts[c_type].append(lang+nlp_ref)
                    
        
        for c_type, components in component_counts.items(): 
            print(c_type, len(components))
  

    def test_get_language_count_for_every_component_type(self):
        component_counts = {}
        for lang in nlu.NameSpace.pretrained_models_references:
            for nlu_ref, nlp_ref in nlu.NameSpace.pretrained_models_references[lang].items():
                c_type = nlu_ref.split('.')

                if len(c_type) <2 : continue
                c_type = c_type[1]

                if c_type not in component_counts : component_counts[c_type] = []
                if (lang+nlp_ref) not in component_counts[c_type] : component_counts[c_type].append(lang+nlp_ref)


        for c_type, components in component_counts.items():
            print(c_type, len(components))

    
    def test_get_count_of_unique_spark_nlp_references(self):
        spark_nlp_references = []

        # # get refs from default        
        # for key, value in nlu.NameSpace.component_alias_references.items():
        #     if value[0] not in spark_nlp_references : spark_nlp_references.append(value[0])

        # get refs from pipes
        pipes = []
        for lang in nlu.NameSpace.pretrained_pipe_references.keys() :
            for key,value in nlu.NameSpace.pretrained_pipe_references[lang].items():
                if lang+value not in spark_nlp_references : 
                    pipes.append(value)
                    spark_nlp_references.append(lang+value)

        models = []
        # get refs from models
        for lang in nlu.NameSpace.pretrained_models_references.keys() :
            for key,value in nlu.NameSpace.pretrained_models_references[lang].items():
                if lang+value not in spark_nlp_references :
                    models.append(value)
                    spark_nlp_references.append(lang+value)
        print("Num overall references", len(spark_nlp_references))
        print("Num Model references", len(models))
        print("Num Pipe references", len(pipes))
        
        
        print(spark_nlp_references)
        
    def test_get_count_of_unique_languages(self):
        print('num languages in NLU : ', len(nlu.all_components_info.all_languages))
    
    
    def test_print_pipe_info(self):
        pipe = nlu.load('sentiment')
        
        pipe.generate_class_metadata_table()
        
    def test_print_all_components_for_lang(self):
        # Test printing of all components for one specific language
        nlu.print_components( 'de')


    def test_print_all_component_types(self):
        # Test printing of all components types
        nlu.print_component_types()
        

        
    def test_print_all_components_for_action_in_lang(self):
        # Test printing of all components for one specific action and language
        nlu.print_components( lang='en', action='classify')

    def test_print_all_components_for_action(self):
        # Test printing of all components for one specific type
        nlu.print_components(action='lemma')


    def test_print_all_components(self):
        nlu.print_components()
    def test_print_all_trainable_components(self):
        nlu.print_trainable_components()



    
if __name__ == '__main__':
    TestComponentInfo().test_entities_config()
