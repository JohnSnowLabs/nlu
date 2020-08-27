---
layout: article
title: Spark NLP release notes
permalink: /docs/en/release_notes
key: docs-release-notes
modify_date: "2020-06-12"
---

NLU release notes 

### 2.5.5
- Confidence extraction bugfix

### 2.5.4
- Fixed bug with bad conversion of datatypes


### 2.5.3
- metadata paraemter for predict function, prettier outputs
- Datatype consistency added for predictions

### 2.5.2
- Modin depenency bugfix

### 2.5.1
- Modin Support

### 2.5.0

- Support for Modin with Ray and Dask Backends
- Consisten input and outputs for predict() . If you input Spark Dataframe , you get Spark Dataframe Back. If you input Modin dataframe, you get Modin back. Analogus for predictions on Numpy and Pandas objects



### 2.5.0.rc1

The birth of a new Machine Learning library       
NLU provides out of the box

- 200+ pretrained models and pipelines for most NLU tasks ( Sentiment, Language Detection, NER, POS, Spellchecking)
- 60 languages
- Latest and greatest embeddings in different flavors (Elmo, Bert, Albert, Xlnert, Glove, Use)
- 13 Different types of NLU components 
