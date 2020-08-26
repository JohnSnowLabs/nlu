# content of test_expectation.py
import pytest
import nlu
'''
Test every component in the NLU namespace. 
This can take very long
'''
all_default_references = []
i=0
for nlu_reference in nlu.NameSpace.default_pretrained_component_references.keys():
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

def setup_module(module):
    """ setup any state specific to the execution of the given module."""
    import nlu
def teardown_module(module):
    """ teardown any state that was previously setup with a setup_module
    method.
    """
    import gc
    gc.collect()
import gc

@pytest.mark.parametrize("nlu_reference,id",all_default_references)
def test_every_default_component(nlu_reference, id):
    gc.collect()

    print( 'param =', nlu_reference)
    print('TESTING NLU REFERENCE : ', nlu_reference)
    if id < skip_to_test : return
    df = nlu.load(nlu_reference).predict('What a wonderful day!')
    print(df)
    print('TESTING DONE FOR NLU REFERENCE : ', nlu_reference)

@pytest.mark.parametrize("nlu_reference,id",all_pipe_references)
def test_every_default_component(nlu_reference,id):
    gc.collect()
    print( 'param =', nlu_reference)
    print('TESTING NLU REFERENCE : ', nlu_reference)
    if id < skip_to_test : return
    df = nlu.load(nlu_reference).predict('What a wonderful day!')
    print(df)
    print('TESTING DONE FOR NLU REFERENCE : ', nlu_reference)

@pytest.mark.parametrize("nlu_reference,id",all_model_references)
def test_every_default_component(nlu_reference,id):
    gc.collect()
    print( 'param =', nlu_reference)
    print('TESTING NLU REFERENCE : ', nlu_reference)
    if id < skip_to_test : return
    df = nlu.load(nlu_reference).predict('What a wonderful day!')
    print(df)
    print('TESTING DONE FOR NLU REFERENCE : ', nlu_reference)

