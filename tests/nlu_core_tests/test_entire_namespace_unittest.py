# content of test_expectation.py
import pytest
import nlu
from nose2.tools import params
import unittest
from parameterized import parameterized, parameterized_class

'''
Test every component in the NLU namespace. 
This can take very long
'''
all_default_references = []
i=0
for nlu_reference in nlu.NameSpace.component_alias_references.keys():
    print('Adding default namespace test ', nlu_reference)
    all_default_references.append((nlu_reference,i))
    i+=1

all_pipe_references = []
for lang in nlu.NameSpace.pretrained_pipe_references.keys():
    for nlu_reference in nlu.NameSpace.pretrained_pipe_references[lang] :
        print('Adding pipe namespace test ', nlu_reference, ' and lang', lang)
        all_pipe_references.append((nlu_reference,i))
        i+=1

all_model_references = []

#because of pytest memory issues for large test suites we have to do tests in batches
skip_to_test=118 #skips all test

for lang in nlu.NameSpace.pretrained_models_references.keys():
    for nlu_reference in nlu.NameSpace.pretrained_models_references[lang] :
        print('Adding model namespace test ', nlu_reference, ' and lang', lang)
        all_model_references.append((nlu_reference,i))
        i+=1

class Test(unittest.TestCase):
    # @params(all_default_references)
    # @params(all_default_references)
    @parameterized.expand(all_default_references)
    def test_every_default_component(self,nlu_reference, id):
        import nlu
        print('TESTING NLU REFERENCE : ', nlu_reference)
        df = nlu.load(nlu_reference).predict('What a wonderful day!')
        print(df)
        print(df.columns)
        print('TESTING DONE FOR NLU REFERENCE : ', nlu_reference)

if __name__ == '__main__':
    unittest.main()


# @pytest.mark.parametrize("nlu_ref,id",all_pipe_references)
# def test_every_default_component(nlu_ref,id):
#     import nlu
#     gc.collect()
#     print( 'param =', nlu_ref)
#     print('TESTING NLU REFERENCE : ', nlu_ref)
#     if id < skip_to_test : return
#     df = nlu.load(nlu_ref).predict('What a wonderful day!')
#     print(df)
#     print(df.columns)
#
#     print('TESTING DONE FOR NLU REFERENCE : ', nlu_ref)
#
# @pytest.mark.parametrize("nlu_ref,id",all_model_references)
# def test_every_default_component(nlu_ref,id):
#     import nlu
#     gc.collect()
#     print( 'param =', nlu_ref)
#     print('TESTING NLU REFERENCE : ', nlu_ref)
#     if id < skip_to_test : return
#     df = nlu.load(nlu_ref).predict('What a wonderful day!')
#     print(df)
#     print(df.columns)
#     print('TESTING DONE FOR NLU REFERENCE : ', nlu_ref)
#
