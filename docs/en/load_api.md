---
layout: docs
header: true
seotitle: NLP | John Snow Labs
title: The nlp.load() function
permalink: /docs/en/load_api
key: docs-developers
modify_date: "2020-05-08"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

The nlp.load() method takes in one or multiple nlp pipeline, model or component references separated by whitespaces.     
See [the Model Namespace]( /docs/en/namespace) for an overview of all possible nlp references.     

NLP  will induce the following reference format for any query to the load method:       
**language.component_type.dataset.embeddings** i.e.: en.sentiment.twitter.use     
      
It is possible to omit many parts of the query and the nlp module will provide the best possible defaults, like embeddings for choosing a dataset.

The NLP Namespace also provides a few aliases which make referencing a model even easier!       
This makes it possible to get predictions by only referencing the component name       
Examples for aliases are nlp.load('bert') or nlp.load('sentiment')   

It is possible to omit the language prefix and start the query with :
**component_type.dataset.embeddings** the nlp module will automatically set the language to english in this case.


**The nlp.load() method returns a NLU pipeline object which provides predictions** :
```python
from johnsnowlabs import nlp 

pipeline = nlp.load('sentiment')
pipeline.predict("I love this Documentation! It's so good!")
``` 
**This is equal to:**
```python
from johnsnowlabs import nlp
nlp.load('sentiment').predict("I love this Documentation! It's so good!")
``` 

</div><div class="h3-box" markdown="1">

## Load Parameters
The load method provides for now just one parameter **verbose**.
Setting nlp.load(nlp_reference, verbose=True) will generate log outputs that can be helpful for troubleshooting.   
If you encounter any errors, please run Verbose mode and post your output on our Github Issues page.    

| Description                                                                              | Parameter name                  | 
|------------------------------------------------------------------------------------------|---------------------------------| 
| NLP reference of the model                                                               | request                         |
| Path to a locally stored Spark NLP Model or Pipeline                                     | path                            |
| Whether to load GPU jars or not. Set to `True` to enable.                                | gpu                             |
| Whether to load M1 jars or not. Set to `True` to enable.                                 | m1_chip                         |
| Whether to use caching for the nlp.display() functions or not.  Set to  `True` to enable | streamlit_caching               |


</div><div class="h3-box" markdown="1">

## Configuring loaded models
To configure your model or pipeline, first load a NLP component and use the print_components() function.   
The print outputs tell you at which index of the pipe_components attribute which NLP component is located.   
Via  setters which are named according to the parameter values a model can be configured


```python
# example for configuring the first element in the component_list
pipe = nlp.load('en.sentiment.twitter')
pipe.generate_class_metadata_table()
document_assembler_model = pipe.components[0].model
document_assembler_model.setCleanupMode('inplace')
```

This will print

```python 
-------------------------------------At pipe.pipe_components[0].model  : document_assembler with configurable parameters: --------------------------------------
Param Name [ cleanupMode ] :  Param Info : possible values: disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full  currently Configured as :  disabled
--------------------------------------------At pipe.pipe_components[1].model  : glove with configurable parameters: --------------------------------------------
Param Name [ dimension ] :  Param Info : Number of embedding dimensions  currently Configured as :  512
----------------------------------------At pipe.pipe_components[2].model  : sentiment_dl  with configurable parameters: ----------------------------------------
Param Name [ threshold ] :  Param Info : The minimum threshold for the final result otherwise it will be neutral  currently Configured as :  0.6
Param Name [ thresholdLabel ] :  Param Info : In case the score is less than threshold, what should be the label. Default is neutral.  currently Configured as :  neutral
Param Name [ classes ] :  Param Info : get the tags used to trained this NerDLModel  currently Configured as :  ['positive', 'negative']
```

</div><div class="h3-box" markdown="1">

## Namespace
The NLP name space describes the collection of all models, pipelines and components available in NLP and supported by the nlp.load() method.       
You can view it on the [Name Space page](https://nlu.johnsnowlabs.com/docs/en/load_api)


</div></div>