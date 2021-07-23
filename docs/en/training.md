---
layout: docs
header: true
title: Training Models with NLU
permalink: /docs/en/training
key: docs-developers
modify_date: "2020-05-08"
---

<div class="main-docs" markdown="1">

<div class="h3-box" markdown="1">

You can fit load a trainable NLU pipeline via ```nlu.load('train.<model>')``` 

# Binary Text Classifier Training
[Sentiment classification training demo](https://colab.research.google.com/drive/1f-EORjO3IpvwRAktuL4EvZPqPr2IZ_g8?usp=sharing)        
To train the a Sentiment classifier model, you must pass a dataframe with a ```text``` column and a ```y``` column for the label.
Uses a Deep Neural Network built in Tensorflow.       
By default *Universal Sentence Encoder Embeddings (USE)* are used as sentence embeddings.

```python
fitted_pipe = nlu.load('train.sentiment').fit(train_df)
preds = fitted_pipe.predict(train_df)
```
If you add a nlu sentence embeddings reference, before the train reference, NLU will use that Sentence embeddings instead of the default USE.

```python
#Train Classifier on BERT sentence embeddings
fitted_pipe = nlu.load('embed_sentence.bert train.classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

```python
#Train Classifier on ELECTRA sentence embeddings
fitted_pipe = nlu.load('embed_sentence.electra train.classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

# Multi Class Text Classifier Training
[Multi Class Text Classifier Training Demo](https://colab.research.google.com/drive/12FA2TVvvRWw4pRhxDnK32WAzl9dbF6Qw?usp=sharing)         
To train the Multi Class text classifier model, you must pass a dataframe with a ```text``` column and a ```y``` column for the label.        
By default *Universal Sentence Encoder Embeddings (USE)* are used as sentence embeddings. 

```python
fitted_pipe = nlu.load('train.classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

If you add a nlu sentence embeddings reference, before the train reference, NLU will use that Sentence embeddings instead of the default USE.

```python
#Train on BERT sentence emebddings
fitted_pipe = nlu.load('embed_sentence.bert train.classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

# Multi Label Classifier training
[ Train Multi Label Classifier on E2E dataset](https://colab.research.google.com/drive/15ZqfNUqliRKP4UgaFcRg5KOSTkqrtDXy?usp=sharing)       
[Train Multi Label  Classifier on Stack Overflow Question Tags dataset](https://drive.google.com/file/d/1Nmrncn-y559od3AKJglwfJ0VmZKjtMAF/view?usp=sharing)       
This model can predict multiple labels for one sentence.     
Uses a Bidirectional GRU with Convolution model that we have built inside TensorFlow and supports up to 100 classes.        
To train the Multi Class text classifier model, you must pass a dataframe with a ```text``` column and a ```y``` column for the label.   
The ```y``` label must be a string column where each label is seperated with a seperator.     
By default, ```,``` is assumed as line seperator.      
If your dataset is using a different label seperator, you must configure the ```label_seperator``` parameter while calling the ```fit()``` method.    

By default *Universal Sentence Encoder Embeddings (USE)* are used as sentence embeddings for training.

```python
fitted_pipe = nlu.load('train.multi_classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

If you add a nlu sentence embeddings reference, before the train reference, NLU will use that Sentence embeddings instead of the default USE.
```python
#Train on BERT sentence emebddings
fitted_pipe = nlu.load('embed_sentence.bert train.multi_classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

Configure a custom line seperator
```python
#Use ; as label seperator
fitted_pipe = nlu.load('embed_sentence.electra train.multi_classifier').fit(train_df, label_seperator=';')
preds = fitted_pipe.predict(train_df)
```



# Part of Speech (POS) Training

Your dataset must be in the form of universal dependencies [Universal Dependencies](https://universaldependencies.org/).
You must configure the dataset_path in the ```fit()``` method to point to the universal dependencies you wish to train on.       
You can configure the delimiter via the ```label_seperator``` parameter      
[POS training demo]](https://colab.research.google.com/drive/1CZqHQmrxkDf7y3rQHVjO-97tCnpUXu_3?usp=sharing)

```python
fitted_pipe = nlu.load('train.pos').fit(dataset_path=train_path, label_seperator='_')
preds = fitted_pipe.predict(train_df)
```



# Named Entity Recognizer (NER) Training
[NER training demo](https://colab.research.google.com/drive/1_GwhdXULq45GZkw3157fAOx4Wqo-fmFV?usp=sharing)        
You can train your own custom NER model with an [CoNLL 20003 IOB](https://www.aclweb.org/anthology/W03-0419.pdf) formatted dataset.      
By default *Glove 100d Token Embeddings* are used as features for the classifier.

```python
train_path = '/content/eng.train'
fitted_pipe = nlu.load('train.ner').fit(dataset_path=train_path)
```

If a NLU reference to a Token Embeddings model is added before the train reference, that Token Embedding will be used when training the NER model.

```python
# Train on BERT embeddigns
train_path = '/content/eng.train'
fitted_pipe = nlu.load('bert train.ner').fit(dataset_path=train_path)
```


# Chunk Entity Resolver Training
[Chunk Entity Resolver Training Tutorial Notebook]()
Named Entities are sub pieces in textual data which are labled with classes.    
These classes and strings are still ambious though and it is not possible to group semantically identically entities withouth any definition of `terminology`.
With the `Chunk Resolver` you can train a state of the art deep learning architecture to map entities to their unique terminological representation.

Train a chunk resolver on a dataset with columns named `y` , `_y` and `text`. `y` is a label, `_y` is an extra identifier label, `text` is the raw text

```python
import pandas as pd 
dataset = pd.DataFrame({
    'text': ['The Tesla company is good to invest is', 'TSLA is good to invest','TESLA INC. we should buy','PUT ALL MONEY IN TSLA inc!!'],
    'y': ['23','23','23','23']
    '_y': ['TESLA','TESLA','TESLA','TESLA'], 

})


trainable_pipe = nlu.load('train.resolve_chunks')
fitted_pipe  = trainable_pipe.fit(dataset)
res = fitted_pipe.predict(dataset)
fitted_pipe.predict(["Peter told me to buy Tesla ", 'I have money to loose, is TSLA a good option?'])
```

| entity_resolution_confidence   | entity_resolution_code   | entity_resolution   | document                                      |
|:-------------------------------|:-------------------------|:--------------------|:----------------------------------------------|
| '1.0000'                     | '23]                   | 'TESLA'           | Peter told me to buy Tesla                    |
| '1.0000'                     | '23]                   | 'TESLA'           | I have money to loose, is TSLA a good option? |


### Train with default glove embeddings
```python
untrained_chunk_resolver = nlu.load('train.resolve_chunks')
trained_chunk_resolver  =  untrained_chunk_resolver.fit(df)
trained_chunk_resolver.predict(df)
```

### Train with custom embeddings
```python
# Use Healthcare Embeddings
trainable_pipe = nlu.load('en.embed.glove.healthcare_100d train.resolve_chunks')
trained_chunk_resolver  =  untrained_chunk_resolver.fit(df)
trained_chunk_resolver.predict(df)
 ```



# Rule based NER with Context Matcher
[Rule based NER with context matching tutorial notebook](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/rule_based_named_entity_recognition_and_resolution/rule_based_NER_and_resolution_with_context_matching.ipynb)    
Define a rule based NER algorithm by providing Regex Patterns and resolution mappings.
The confidence value is computed  using a heuristic approach based on how many matches it has.    
A dictionary can be provided with setDictionary to map extracted entities to a unified representation. The first column of the dictionary file should be the representation with following columns the possible matches.


```python
import nlu
import json
# Define helper functions to write NER rules to file 
"""Generate json with dict contexts at target path"""
def dump_dict_to_json_file(dict, path): 
  with open(path, 'w') as f: json.dump(dict, f)

"""Dump raw text file """
def dump_file_to_csv(data,path):
  with open(path, 'w') as f:f.write(data)
sample_text = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting. Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Twenty days ago. Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . At birth the typical boy is growing slightly faster than the typical girl, but the velocities become equal at about seven months, and then the girl grows faster until four years. From then until adolescence no differences in velocity can be detected. 21-02-2020 21/04/2020 """

# Define Gender NER matching rules
gender_rules = {
    "entity": "Gender",
    "ruleScope": "sentence",
    "completeMatchRegex": "true"    }

# Define dict data in csv format
gender_data = '''male,man,male,boy,gentleman,he,him
female,woman,female,girl,lady,old-lady,she,her
neutral,neutral'''

# Dump configs to file 
dump_dict_to_json_file(gender_data, 'gender.csv')
dump_dict_to_json_file(gender_rules, 'gender.json')
gender_NER_pipe = nlu.load('match.context')
gender_NER_pipe.print_info()
gender_NER_pipe['context_matcher'].setJsonPath('gender.json')
gender_NER_pipe['context_matcher'].setDictionary('gender.csv', options={"delimiter":","})
gender_NER_pipe.predict(sample_text)
```

| context_match | context_match_confidence |
| :------------ | -----------------------: |
| female        |                     0.13 |
| she           |                     0.13 |
| she           |                     0.13 |
| she           |                     0.13 |
| she           |                     0.13 |
| boy           |                     0.13 |
| girl          |                     0.13 |
| girl          |                     0.13 |

### Context Matcher Parameters
You can define the following parameters in your rules.json file to define the entities to be matched

| Parameter             | Type                    | Description                                                  |
| --------------------- | ----------------------- | ------------------------------------------------------------ |
| entity                | `str   `                | The name of this rule                                        |
| regex                 | `Optional[str] `        | Regex Pattern to extract candidates                          |
| contextLength         | `Optional[int] `        | defines the maximum distance a prefix and suffix words can be away from the word to match,whereas context are words that must be immediately after or before the word to match |
| prefix                | `Optional[List[str]] `  | Words preceding the regex match, that are at most `contextLength` characters aways |
| regexPrefix           | `Optional[str]  `       | RegexPattern of words preceding the regex match, that are at most `contextLength` characters aways |
| suffix                | `Optional[List[str]]  ` | Words following the regex match, that are at most `contextLength` characters aways |
| regexSuffix           | `Optional[str] `        | RegexPattern of words following the regex match, that are at most `contextLength` distance aways |
| context               | `Optional[List[str]] `  | list of words that must be immediatly before/after a match   |
| contextException      | `Optional[List[str]] `  | ?? List of words that may not be immediatly before/after a match |
| exceptionDistance     | `Optional[int] `        | Distance exceptions must be away from a match                |
| regexContextException | `Optional[str] `        | Regex Pattern of exceptions that may not be within `exceptionDistance` range of the match |
| matchScope            | `Optional[str]`         | Either `token` or `sub-token` to match on character basis    |
| completeMatchRegex    | `Optional[str]`         | Wether to use complete or partial matching, either `"true"` or `"false"` |
| ruleScope             | `str`                   | currently only `sentence` supported                          |

# Saving a NLU pipeline to disk

```python
train_path = '/content/eng.train'
fitted_pipe = nlu.load('train.ner').fit(dataset_path=train_path)
stored_model_path = './models/classifier_dl_trained' 
fitted_pipe.save(stored_model_path)

```

# Loading a NLU pipeline from disk

```python
train_path = '/content/eng.train'
fitted_pipe = nlu.load('train.ner').fit(dataset_path=train_path)
stored_model_path = './models/classifier_dl_trained' 
fitted_pipe.save(stored_model_path)
hdd_pipe = nlu.load(path=stored_model_path)
```



# Loading a NLU pipeline as pyspark.ml.PipelineModel
```python
import pyspark
# load the NLU pipeline as pyspark pipeline
pyspark_pipe = pyspark.ml.PipelineModel.load(stored_model_path)
# Generate spark Df and transform it with the pyspark Pipeline
s_df = spark.createDataFrame(df)
pyspark_pipe.transform(s_df).show()
```


</div></div>