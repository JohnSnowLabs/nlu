---
layout: article
title: The NLU Load function
permalink: /docs/en/load_api
key: docs-developers
modify_date: "2020-05-08"
---

The nlu.load() method takes in one or multiple NLU pipeline, model or component references seperated by whitespaces.      
See [the NLU namespace](model_namespace) for an overview of all possibles NLU references.      

NLU  will induce the following reference format for any query to the load method:        
**language.component_type.dataset.embeddings** i.e.: en.sentiment.twitter.use      
        
It is possible to omit many parts of the query and NLU will provide the best possible defaults, like embeddings for choosing a dataset.

The NLU namespace also provides a few aliases which make referencing a model even easier!        
This makes it possible to get preditions by only referencing the component name        
Examples for aliases are nlu.load('bert') or nlu.load('sentiment')    

It is possible to omit the language prefix and start the query with :
**component_type.dataset.embeddings** NLU will automatically set the languge to english in this case.


**The nlu.load() method returns a NLU pipeline object which provides predictions** : 
```python
import nlu
pipeline = nlu.load('sentiment')
pipeline.predict("I love this Documentation! Its so good!")
```  
**This is equal to:**
```python
import nlu
nlu.load(sentiment).predict("I love this Documentation! Its so good!")
```  


# Load Parameter
The load method provides for now just one parameter **verbose**.
Setting nlu.load(nlu_reference, verbose=True) will generate log outputs that can be helpful for troubleshooting.    
If you encounter any errors, please run Verbose mode and post your output on our Github Issues page.     


# Configuring loaded models
To configure your model or pipeline, first load a NLU component and use the print_components() function.    
The print outputs tell you at which index of the pipe_components attribute which NLU component is located.    
Via  setters which ae named according to the parameter values a model can be configured
 

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
Param Name [ cleanupMode ] :  Param Info : possible values: disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full  currenlty Configured as :  disabled
--------------------------------------------At pipe.pipe_components[1].model  : glove with configurable parameters: --------------------------------------------
Param Name [ lazyAnnotator ] :  Param Info : Whether this AnnotatorModel acts as lazy in RecursivePipelines  currenlty Configured as :  False
Param Name [ dimension ] :  Param Info : Number of embedding dimensions  currenlty Configured as :  512
Param Name [ storageRef ] :  Param Info : unique reference name for identification  currenlty Configured as :  tfhub_use
----------------------------------------At pipe.pipe_components[2].model  : sentiment_dl  with configurable parameters: ----------------------------------------
Param Name [ lazyAnnotator ] :  Param Info : Whether this AnnotatorModel acts as lazy in RecursivePipelines  currenlty Configured as :  False
Param Name [ threshold ] :  Param Info : The minimum threshold for the final result otheriwse it will be neutral  currenlty Configured as :  0.6
Param Name [ thresholdLabel ] :  Param Info : In case the score is less than threshold, what should be the label. Default is neutral.  currenlty Configured as :  neutral
Param Name [ classes ] :  Param Info : get the tags used to trained this NerDLModel  currenlty Configured as :  ['positive', 'negative']
Param Name [ storageRef ] :  Param Info : unique reference name for identification  currenlty Configured as :  tfhub_use
```

# TODO outputs
pipe.components['sentiment'].setConfig(Bla)
pipe.components['tokenizer'].setConfig(bla)