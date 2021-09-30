---
layout: docs
header: true
seotitle: NLU | John Snow Labs
title: NLU under the hood
key: docs-examples
permalink: /docs/en/under_the_hood
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

<div class="h3-box" markdown="1">

This page acts as reference on the internal working and implementation of NLU.
It acts as a reference for internal development and open source contributers.

</div><div class="h3-box" markdown="1">

## How do NLU internals work?

NLU defines a universe of components which act as building blocks of user definable machine learning pipelines.    
The possibilities of creating unique and useful pipelines with NLU are only limited by onces imagination and  RAM.    

</div><div class="h3-box" markdown="1">

### The NLU component universe
There are many different types of Components in the NLU universe.
Each of the components acts as a wrapper around multiple different Spark NLP transformers.

</div><div class="h3-box" markdown="1">

## NLU spellbook
NLU defines a mapping for every of its model references to a specific Spark NLP model, pipeline or annotator.
You can view the mapping in the [TODO] file .
If no model is found, NLU will ping the John Snow Labs modelhub for any new models.
If the modelhub cannot resolve a Spark NLP reference for a NLU reference. NLU whill throw an exception, indicating that a component could not be resolved.
If the NLU reference points to a Spark NLP pipeline, it will unpack each model from the Spark NLP pipeline and and package it inside of corrosponding NLU components.

</div><div class="h3-box" markdown="1">

## NLU pipeline building steps
A NLU pipeline object cann either be created via the nlu.load('nlu.reference') API
or alternatively via the nlu.build([Component1,Component2]) API.

The pipeline will not start its building steps until the .predict() function is called for the first time on it.
When .predict() is called, the following steps occur
1. Check for every NLU component, wether all its inputs are satisfied.
I.e. if a user builds a pipeline with a classifier model in it but does not provide any embeddings. NLU will auto resolve the correct embeddings for the passed model
2.  Check and fix for every model if the input names align with the output names of the components they depend on.
3. Check and fix for every model that it is in the correct order in the pipeline. I.e. a sentence classifier must come after the sentence embeddings are generated in the pipeline, not before.

</div><div class="h3-box" markdown="1">

## NLU output generation steps
The .predict() method invokes a series of steps to ensure that the generated output is in the most usable format for further downstream ML tasks.

The steps are the following :
1. NLU converts the input data to a Spark Dataframe and lets the pipeline transform it to a new Spark Dataframe which contains all the features
2. If the output level is not set by the user, it will check what is the last component of the pipe and the infer from that what the output level should be. Each components default output level can be viewed in its corrosponding component.json file.
    Some components output level depend on their input, i.e classifiers can classify on sentences or documents. NLU does additional steps to infer the output level for these kinds of components.
3. Decide which columns to keep and which to drop
4. All the output columns that are at the same outputlevel as the pipe will be zipped and exploded.
5. All columns which are at diferent output level will be selected from the Spark Dataframe, which results in lists in the final output.

After all these steps the final pandas dataframe will be returned from the .predict() method

</div></div>