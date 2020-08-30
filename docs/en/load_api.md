---
layout: article
title: The NLU Load function
permalink: /docs/en/load_api
key: docs-developers
modify_date: "2020-05-08"
---

The nlu.load() method takes in one or multiple NLU pipeline, model or component references separated by whitespaces.     
See [the NLU namespace]( /docs/en/namespace) for an overview of all possible NLU references.     

NLU  will induce the following reference format for any query to the load method:       
**language.component_type.dataset.embeddings** i.e.: en.sentiment.twitter.use     
      
It is possible to omit many parts of the query and NLU will provide the best possible defaults, like embeddings for choosing a dataset.

The NLU namespace also provides a few aliases which make referencing a model even easier!       
This makes it possible to get predictions by only referencing the component name       
Examples for aliases are nlu.load('bert') or nlu.load('sentiment')   

It is possible to omit the language prefix and start the query with :
**component_type.dataset.embeddings** NLU will automatically set the language to english in this case.


**The nlu.load() method returns a NLU pipeline object which provides predictions** :
```python
import nlu
pipeline = nlu.load('sentiment')
pipeline.predict("I love this Documentation! It's so good!")
``` 
**This is equal to:**
```python
import nlu
nlu.load(sentiment).predict("I love this Documentation! It's so good!")
``` 


# Load Parameter
The load method provides for now just one parameter **verbose**.
Setting nlu.load(nlu_reference, verbose=True) will generate log outputs that can be helpful for troubleshooting.   
If you encounter any errors, please run Verbose mode and post your output on our Github Issues page.    


# Configuring loaded models
To configure your model or pipeline, first load a NLU component and use the print_components() function.   
The print outputs tell you at which index of the pipe_components attribute which NLU component is located.   
Via  setters which are named according to the parameter values a model can be configured


```python
#example for configuring the first element in the pipe
pipe = nlu.load('en.sentiment.twitter')
pipe.print_info()
document_assembler_model = pipe.pipe_components[0].model
document_assembler_model.setCleanupMode('inplace')
```

This will print

```python 
-------------------------------------At pipe.pipe_components[0].model  : document_assembler with configurable parameters: --------------------------------------
Param Name [ cleanupMode ] :  Param Info : possible values: disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full  currently Configured as :  disabled
--------------------------------------------At pipe.pipe_components[1].model  : glove with configurable parameters: --------------------------------------------
Param Name [ lazyAnnotator ] :  Param Info : Whether this AnnotatorModel acts as lazy in RecursivePipelines  currently Configured as :  False
Param Name [ dimension ] :  Param Info : Number of embedding dimensions  currently Configured as :  512
Param Name [ storageRef ] :  Param Info : unique reference name for identification  currently Configured as :  tfhub_use
----------------------------------------At pipe.pipe_components[2].model  : sentiment_dl  with configurable parameters: ----------------------------------------
Param Name [ lazyAnnotator ] :  Param Info : Whether this AnnotatorModel acts as lazy in RecursivePipelines  currently Configured as :  False
Param Name [ threshold ] :  Param Info : The minimum threshold for the final result otherwise it will be neutral  currently Configured as :  0.6
Param Name [ thresholdLabel ] :  Param Info : In case the score is less than threshold, what should be the label. Default is neutral.  currently Configured as :  neutral
Param Name [ classes ] :  Param Info : get the tags used to trained this NerDLModel  currently Configured as :  ['positive', 'negative']
Param Name [ storageRef ] :  Param Info : unique reference name for identification  currently Configured as :  tfhub_use
```



## Component Namespace
The NLU name space describes the collection of all models, pipelines and components available in NLU and supported by the nlu.load() method.       
You can view it on the [Name Space page](https://nlu.johnsnowlabs.com/docs/en/load_api)

NLU also provides a few handy functions to gain insight into the NLU namespace.


