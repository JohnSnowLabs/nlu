---
layout: article
title: Spark NLP release notes
permalink: /docs/en/release_notes
key: docs-release-notes
modify_date: "2020-06-12"
---

NLU release notes

### 2.6
 - Added 100+ new models from Spark NLP 2.6
    - New YAKE model
    - New XYZ model
    - ...
 - Added examples for all new
 - Improved outputs for Chunk level components 
 - Integrated removal of IOB prefixes of NER tags
 - Integrated light pipeline which yields 10x speed up for predictions 
 - Easy and Copy pastable moel configs via pipe.print_info()
 - N new Notebooks
 - Recycling of Pandas indexes for predicting. No more ID columns, just pandas indexes.
 - 


### 2.5.6
 - Better Defaults for spell checking
 - Lots of bug fixes
 - Additional feature discovery via nlu.components()
 - Memory optimization
 - Refactoring
 - Docs and Examples updates
 
### 2.5.5
- Confidence extraction bugfix

### 2.5.4
- Fixed bug with bad conversion of datatypes


### 2.5.3
- metadata parameter for predict function, prettier outputs
- Datatype consistency added for predictions

### 2.5.2
- Modin dependency bugfix

### 2.5.1
- Modin Support

### 2.5.0

- Support for Modin with Ray and Dask Backends
- Consistent input and outputs for predict() . If you input Spark Dataframe , you get Spark Dataframe Back. If you input Modin dataframe, you get Modin back. Analogous for predictions on Numpy and Pandas objects



### 2.5.0.rc1

The birth of a new Machine Learning library      
NLU provides out of the box

- 200+ pretrained models and pipelines for most NLU tasks ( Sentiment, Language Detection, NER, POS, Spell Checking)
- 60 languages
- Latest and greatest embeddings in different flavors (Elmo, Bert, Albert, Xlnert, Glove, Use)
- 13 Different types of NLU components


