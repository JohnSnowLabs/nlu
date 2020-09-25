# 1. create meta file for all classes
# 2. Create meta file for all open source models
# 3. create meta file for all licenced classes
# 4. create meta file for all licenced models
import nlu
import sparknlp
sparknlp.start()
import pandas as pd
from nlu.components.classifier import Classifier
from nlu.components.unlabeled_dependency_parser import UnlabeledDependencyParser
from nlu.components.labeled_dependency_parser import LabeledDependencyParser


from nlu.components import chunker
nlu_components = []
#
#all chunker trained classses
nlu_components.append( nlu.chunker.Chunker('default_chunker'))
nlu_components.append( nlu.chunker.Chunker('ngram'))

#all classifier trained classes

nlu_components.append(nlu.components.classifier.Classifier('classifier_dl') )
nlu_components.append(nlu.components.classifier.Classifier('e2e') )
nlu_components.append(nlu.components.classifier.Classifier(sparknlp_reference='sentimentdl') )
nlu_components.append(nlu.components.classifier.Classifier('vivekn') )
nlu_components.append(nlu.components.classifier.Classifier('yake') )
nlu_components.append(nlu.components.classifier.Classifier('wiki_') )
nlu_components.append(nlu.components.classifier.Classifier('ner') )
nlu_components.append(nlu.components.classifier.Classifier('pos') )


# Dep typed and untyped
nlu_components.append(UnlabeledDependencyParser())
nlu_components.append(LabeledDependencyParser())

# embeddings
nlu_components.append(nlu.Embeddings('bert'))
nlu_components.append(nlu.Embeddings('albert'))
nlu_components.append(nlu.Embeddings('glove'))
nlu_components.append(nlu.Embeddings('use'))

nlu_components.append(nlu.Embeddings('sentence_bert'))
nlu_components.append(nlu.Embeddings('elmo'))
nlu_components.append(nlu.Embeddings('xlnet'))

# chunk embeds skipped cuz there are no pretrained afaik

nlu_components.append(nlu.Matcher('text'))
nlu_components.append(nlu.Matcher('date'))
nlu_components.append(nlu.Matcher('regex'))
nlu_components.append(nlu.Normalizer())
nlu_components.append(nlu.NLUSentenceDetector('deep_sentence_detector'))
nlu_components.append(nlu.NLUSentenceDetector('pragmatic_sentence_detector'))

nlu_components.append(nlu.SpellChecker('context_spell'))
nlu_components.append(nlu.SpellChecker('norvig_spell'))
nlu_components.append(nlu.SpellChecker('context_spell'))
nlu_components.append(nlu.Stemmer())
nlu_components.append(nlu.StopWordsCleaner(get_default=True))
nlu_components.append(nlu.Lemmatizer(get_default=True))

pipe = nlu.NLUPipeline()
for c in nlu_components : pipe.add(c)


df = pd.DataFrame()
def print_info(model_dict ,):
    '''
    Print out information about every component currently loaded in the pipe and their configurable parameters
    :return: None
    '''


    all_outputs = []
    final_df = {}

    for i, component_key in enumerate(model_dict.keys()) :


        # get comp info of model
        c_info = None
        for c in model_dict.pipe_components:
            if c.component_info.name == component_key :
                c_info = c.component_info
                break
        if c_info == None : print('CCOULD NOT FIND INFO FOR ', component_key)

        final_df[component_key] = {
            'model_class' : type(model_dict[component_key]).__name__,
            'class_description' : 'TODO',
            'inputs' : c_info.outputs,
            'outputs' : c_info.inputs,
            # 'label' : [], only for approaches
            'class_parameters' : [],
            'class_license' : c_info.license,
            'dataset_schema' : 'TODO',
            'class_annotation_sample' : 'TODO',
        }

        print(component_key)
        p_map = model_dict[component_key].extractParamMap()
        all_param_dicts ={}

        # get all param info
        for key in p_map.keys():
            if 'lazyAnnotator' in key.name: continue

            param_dict = {}
            param_dict['param_name'] = key.name
            param_dict['param_description'] = key.doc
            param_dict['param_default_value'] = str(p_map[key])
            param_dict['param_setter_method'] = 'TODO'
            param_dict['param_getter_method'] = 'TODO'

            if type(p_map[key]) == str :
                s1 = "model.set"+ str( key.name[0].capitalize())+ key.name[1:]+"('"+str(p_map[key])+"')"
            else :
                s1 = "model.set"+ str( key.name[0].capitalize())+ key.name[1:]+"("+str(p_map[key])+")"


            if type(p_map[key]) == str :
                s2 = "model.get"+ str( key.name[0].capitalize())+ key.name[1:]+"()"
            else :
                s2 = "model.get"+ str( key.name[0].capitalize())+ key.name[1:]+"()"


            param_dict['param_setter_method'] = s1
            param_dict['param_getter_method'] = s2
            final_df[component_key]['class_parameters'].append(param_dict)



    return final_df


model_df = pd.DataFrame(print_info(pipe))
model_df = model_df.T
print(model_df)
model_df.to_csv('./models.csv')
print(1+1)

# class_parameter col is a list of Dictionaries.
# Each dict  in the class_parameter list describes a param with the following values :
# - param_name
# - param_description
# - param_default_value
# - param_setter_method
# - param_getter_method