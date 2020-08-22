---
layout: article
title: NLU Concepts
permalink: /docs/en/concepts
key: docs-concepts
modify_date: "2020-05-08"
---


The NLU library provides 2 simple methods with which most NLU tasks can be solved while achieving state of the art results.    
The **load** and **predict** method.     

When building a NLU programm you will usually go through the following steps :  

 1. Pick a model/pipeline/component you want to create from the [NLU namespace](model_namespace)
 2. Call the nlu.load(component) method which returns a NLU model pipeline object
 3. call model.predict() on some String input
 
These 3 steps have been boiled down to **just 1 line**
```python
import nlu
nlu.load('sentiment').predict('How does this witchcraft work?')
```


## NLU components
NLU defines a universe of NLU components which can be viewed as stackable and interchangable parts, inspired by methodology of category theory.          
Inside of this NLU universe, abitrary machine learning pipelines can be constructed from its elements.      

NLU currently defines **13 components types** in its universe.     
Each component type embelishes one of many **component kinds**.   
Each component kind embelished one of many **NLU algorithms** 
NLU algorithms are represented by pretrained models or pipelines.      
A **pretrained model** could be a Deep Neural Network or a simple word matcher.    
A **pipeline** consists of a stack of pretrained models.     

### NLU component types

Any of these component types can be passed as a string to nlu.load() and will return you the default model for that component type.  
You can further specify your model selection by placing a '.' behind your component selection.
After the '.' you can specify the model you want via metioning a dataset or model version.    
See [The NLU components namespace](model_namespace) and [The load f](load_api)


- classifiers
- embeddings
- labeled dependency parsers
- unlabled dependency parsers
- lemmatizers
- matchers 
- normalizers
- sentence detectors 
- spell checkers 
- stemmers
- stopwords cleaners
- tokenizers



## Component Name Space
The NLU name space describes the collection of all models, pipelines and components available in NLU and supported by the nlu.load() method.        
You can view it on the [Name Space page](model_namespace)

NLU also provides a few handy funtions to gain insight into the NLU namespace.

**Print all supported languages:**
```python
import nlu
nlu.print_all_nlu_supported_languages()
```

**Print every component for one specific language:**
```python
import nlu
nlu.print_all_nlu_components_for_lang(lang='de')
```

**Print the entire NLU namespace offering:**
```python
import nlu
nlu.print_all_nlu_components()
```



