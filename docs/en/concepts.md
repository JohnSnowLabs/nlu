---
layout: docs
header: true
title: NLU Concepts
permalink: /docs/en/concepts
key: docs-concepts
modify_date: "2020-05-08"
---

<div class="main-docs" markdown="1">

<div class="h3-box" markdown="1">

The NLU library provides 2 simple methods with which most NLU tasks can be solved while achieving state of the art results.   
The **load** and **predict** method.    

When building a NLU programm you will usually go through the following steps : 

1. Pick a model/pipeline/component you want to create from the [NLU namespace](/docs/en/namespace)
2. Call the nlu.load(component) method which returns a NLU model pipeline object
3. Call model.predict() on some String input

These 3 steps have been boiled down to **just 1 line**
```python
import nlu
nlu.load('sentiment').predict('How does this witchcraft work?')
```

</div><div class="h3-box" markdown="1">

## NLU components
NLU defines a universe of NLU components which can be viewed as stackable and interchangeable parts, inspired by methodology of category theory.         
Inside of this NLU universe, arbitrary machine learning pipelines can be constructed from its elements.     

NLU currently defines **18 components types** in its universe.    
Each component type embelishes one of many **component kinds**.  
Each component kind embelished one of many **NLU algorithms**.        
NLU algorithms are represented by pretrained models or pipelines.     
A **pretrained model** could be a Deep Neural Network or a simple word matcher.   
A **pipeline** consists of a stack of pretrained models.    

</div><div class="h3-box" markdown="1">

### NLU component types

Any of these component types can be passed as a string to nlu.load() and will return you the default model for that component type. 
You can further specify your model selection by placing a '.' behind your component selection.        
After the '.' you can specify the model you want via specifying a dataset or model version.   
See [the NLU components namespace](https://nlu.johnsnowlabs.com/docs/en/namespace) and [The load function](https://nlu.johnsnowlabs.com/docs/en/load_api)

{:.table-model-big}
|Component type|  nlu.load() action reference  |
|--------------|--------------------------------|
|Named Entity Recognition(NER) | ner |
|Part of Speech (POS) | pos |
|Classifiers | classify |
|Word embeddings| embed|
|Sentence embeddings| embed_sentence|
|Chunk embeddings| embed_chunk|
|Labeled dependency parsers| dep
|Unlabeled dependency parsers| dep.untyped
|Lemmatizers| lemma|
|Matchers| match|
|Normalizers| norm|
|Sentence detectors| sentence_detector |
|Chunkers| chunk |
|Spell checkers|  spell |
|Stemmers|stem |
|Stopwords cleaners| stopwords |
|Cleaner| clean |
|Tokenizers| tokenize |

</div><div class="h3-box" markdown="1">

## Specifying language for an action

### Print all supported languages
Any of these are partial NLU references which can be prefixed to a request to specify a language
```python
nlu.languages()
```

</div><div class="h3-box" markdown="1">

### Print every component for one specific language
These are complete NLU references and can be passed to the nlu.load() method right away
```python
# Print every German NLU component
nlu.print_components(lang='de')
```

</div><div class="h3-box" markdown="1">

### Print every model for an action
These are complete NLU references and can be passed to the nlu.load() method right away
```python
# Print every lemmatizer for every language
nlu.print_components(action='lemma')
```

</div><div class="h3-box" markdown="1">

### Print every model kind for an action and a language
These are complete NLU references and can be passed to the nlu.load() method right away
```python
# Print all english classifiers
nlu.print_components(lang='en', action='classify')
```

</div><div class="h3-box" markdown="1">

### Print the entire NLU namespace offering
These are complete NLU references and can be passed to the nlu.load() method right away
```python
nlu.print_components()
```

</div></div>