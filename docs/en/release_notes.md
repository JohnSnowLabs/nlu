---
layout: docs
header: true
layout: article
title: NLU release notes
permalink: /docs/en/release_notes
key: docs-release-notes
modify_date: "2020-06-12"
---

<div class="main-docs" markdown="1">


<div class="h3-box" markdown="1">



##  NLU 1.1.0 Release Notes 
We are incredibly excited to release NLU 1.1.0!
This release it integrates the 720+ new models from the latest [Spark-NLP 2.7.0 + releases](https://github.com/JohnSnowLabs/spark-nlp/releases)
You can now achieve state-of-the-art results with Sequence2Sequence transformers like for problems text summarization, question answering, translation between  192+ languages and extract Named Entity in various Right to Left written languages like Koreas, Japanese, Chinese and many more in 1 line of code!     
These new features are possible because of the integration of the [Google's T5 models](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) and [Microsoft's Marian models](https://marian-nmt.github.io/publications/)  transformers

NLU 1.1.0 has over 720+ new pretrained models and pipelines while extending the support of multi-lingual models to 192+ languages such as Chinese, Japanese, Korean, Arabic, Persian, Urdu, and Hebrew.     



### NLU 1.1.0  New Features
* **720+** new models you can find an overview of all NLU models [here](https://nlu.johnsnowlabs.com/docs/en/namespace) and further documentation in the [models hub](https://nlp.johnsnowlabs.com/models)
* **NEW:** Introducing MarianTransformer annotator for machine translation based on MarianNMT models. Marian is an efficient, free Neural Machine Translation framework mainly being developed by the Microsoft Translator team (646+ pretrained models & pipelines in 192+ languages)
* **NEW:** Introducing T5Transformer annotator for Text-To-Text Transfer Transformer (Google T5) models to achieve state-of-the-art results on multiple NLP tasks such as Translation, Summarization, Question Answering, Sentence Similarity, and so on
* **NEW:** Introducing brand new and refactored language detection and identification models. The new LanguageDetectorDL is faster, more accurate, and supports up to 375 languages
* **NEW:** Introducing WordSegmenter model for word segmentation of languages without any rule-based tokenization such as Chinese, Japanese, or Korean
* **NEW:** Introducing DocumentNormalizer component for cleaning content from HTML or XML documents, applying either data cleansing using an arbitrary number of custom regular expressions either data extraction following the different parameters


### NLU 1.1.0  New Notebooks, Tutorials and Articles
- [Translate between 192+ languages with marian](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/translation_demo.ipynb)
- [Try out the 18 Tasks like Summarization Question Answering and more on T5](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more)
- [Tokenize, extract POS and NER in Chinese](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/multilingual/chinese_ner_pos_and_tokenization.ipynb)
- [Tokenize, extract POS and NER in Korean](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/multilingual/korean_ner_pos_and_tokenization.ipynb)
- [Tokenize, extract POS and NER in Japanese](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/multilingual/japanese_ner_pos_and_tokenization.ipynb)
- [Normalize documents](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/text_pre_processing_and_cleaning/document_normalizer_demo.ipynb)
- [Aspect based sentiment NER sentiment for restaurants](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/named_entity_recognition_(NER)/aspect_based_ner_sentiment_restaurants.ipynb)

### NLU 1.1.0 New Training Tutorials
#### Binary Classifier training Jupyter tutorials
- [2 class Finance News sentiment classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_apple_twitter.ipynb)
- [2 class Reddit comment sentiment classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_reddit.ipynb)
- [2 class Apple Tweets sentiment classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_IMDB.ipynb)
- [2 class IMDB Movie sentiment classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_IMDB.ipynb)
- [2 class twitter classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_twitter.ipynb)

#### Multi Class text Classifier training Jupyter tutorials
- [5 class WineEnthusiast Wine review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_wine.ipynb) 
- [3 class Amazon Phone review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_amazon.ipynb)
- [5 class Amazon Musical Instruments review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_musical_instruments.ipynb)
- [5 class Tripadvisor Hotel review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_hotel_reviews.ipynb)
- [5 class Phone review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_hotel_reviews.ipynb)


### NLU 1.1.0 New Medium Tutorials

- [1 line to Glove Word Embeddings with NLU     with t-SNE plots](https://medium.com/spark-nlp/1-line-to-glove-word-embeddings-with-nlu-in-python-baed152fff4d)     
- [1 line to Xlnet Word Embeddings with NLU     with t-SNE plots](https://medium.com/spark-nlp/1-line-to-xlnet-word-embeddings-with-nlu-in-python-5efc57d7ac79)     
- [1 line to AlBERT Word Embeddings with NLU    with t-SNE plots](https://medium.com/spark-nlp/1-line-to-albert-word-embeddings-with-nlu-in-python-1691bc048ed1)     
- [1 line to CovidBERT Word Embeddings with NLU with t-SNE plots](https://medium.com/spark-nlp/1-line-to-covidbert-word-embeddings-with-nlu-in-python-e67396da2f78)     
- [1 line to Electra Word Embeddings with NLU   with t-SNE plots](https://medium.com/spark-nlp/1-line-to-electra-word-embeddings-with-nlu-in-python-25f749bf3e92)     
- [1 line to BioBERT Word Embeddings with NLU   with t-SNE plots](https://medium.com/spark-nlp/1-line-to-biobert-word-embeddings-with-nlu-in-python-7224ab52e131)     




## Translation
[Translation example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/translation_demo.ipynb)       
You can translate between more than 192 Languages pairs with the [Marian Models](https://marian-nmt.github.io/publications/)
You need to specify the language your data is in as `start_language` and the language you want to translate to as `target_language`.    
The language references must be [ISO language codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)

`nlu.load('<start_language>.translate.<target_language>')`

**Translate Turkish to English:**     
`nlu.load('tr.translate_to.fr')`

**Translate English to French:**     
`nlu.load('en.translate_to.fr')`


**Translate French to Hebrew:**     
`nlu.load('en.translate_to.fr')`

```python
translate_pipe = nlu.load('en.translate_to.fr')
df = translate_pipe.predict('Billy likes to go to the mall every sunday')
df
```

|	sentence|	translation|
|-----------|--------------|
|Billy likes to go to the mall every sunday	| Billy geht gerne jeden Sonntag ins Einkaufszentrum|






## T5
[Example of every T5 task](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more)
### Overview of every task available with T5
[The T5 model](https://arxiv.org/pdf/1910.10683.pdf) is trained on various datasets for 17 different tasks which fall into 8 categories.


1. Text summarization
2. Question answering
3. Translation
4. Sentiment analysis
5. Natural Language inference
6. Coreference resolution
7. Sentence Completion
8. Word sense disambiguation

### Every T5 Task with explanation:

|Task Name | Explanation | 
|----------|--------------|
|[1.CoLA](https://nyu-mll.github.io/CoLA/)                   | Classify if a sentence is gramaticaly correct|
|[2.RTE](https://dl.acm.org/doi/10.1007/11736790_9)                    | Classify whether if a statement can be deducted from a sentence|
|[3.MNLI](https://arxiv.org/abs/1704.05426)                   | Classify for a hypothesis and premise whether they contradict or contradict each other or neither of both (3 class).|
|[4.MRPC](https://www.aclweb.org/anthology/I05-5002.pdf)                   | Classify whether a pair of sentences is a re-phrasing of each other (semantically equivalent)|
|[5.QNLI](https://arxiv.org/pdf/1804.07461.pdf)                   | Classify whether the answer to a question can be deducted from an answer candidate.|
|[6.QQP](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)                    | Classify whether a pair of questions is a re-phrasing of each other (semantically equivalent)|
|[7.SST2](https://www.aclweb.org/anthology/D13-1170.pdf)                   | Classify the sentiment of a sentence as positive or negative|
|[8.STSB](https://www.aclweb.org/anthology/S17-2001/)                   | Classify the sentiment of a sentence on a scale from 1 to 5 (21 Sentiment classes)|
|[9.CB](https://ojs.ub.uni-konstanz.de/sub/index.php/sub/article/view/601)                     | Classify for a premise and a hypothesis whether they contradict each other or not (binary).|
|[10.COPA](https://www.aaai.org/ocs/index.php/SSS/SSS11/paper/view/2418/0)                   | Classify for a question, premise, and 2 choices which choice the correct choice is (binary).|
|[11.MultiRc](https://www.aclweb.org/anthology/N18-1023.pdf)                | Classify for a question, a paragraph of text, and an answer candidate, if the answer is correct (binary),|
|[12.WiC](https://arxiv.org/abs/1808.09121)                    | Classify for a pair of sentences and a disambigous word if the word has the same meaning in both sentences.|
|[13.WSC/DPR](https://www.aaai.org/ocs/index.php/KR/KR12/paper/view/4492/0)       | Predict for an ambiguous pronoun in a sentence what it is referring to.  |
|[14.Summarization](https://arxiv.org/abs/1506.03340)          | Summarize text into a shorter representation.|
|[15.SQuAD](https://arxiv.org/abs/1606.05250)                  | Answer a question for a given context.|
|[16.WMT1.](https://arxiv.org/abs/1706.03762)                  | Translate English to German|
|[17.WMT2.](https://arxiv.org/abs/1706.03762)                   | Translate English to French|
|[18.WMT3.](https://arxiv.org/abs/1706.03762)                   | Translate English to Romanian|

[refer to this notebook](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more) to see how to use every T5 Task.




## Question Answering
[Question answering example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more))

Predict an `answer` to a `question` based on input `context`.    
This is based on [SQuAD - Context based question answering](https://arxiv.org/abs/1606.05250)


|Predicted Answer | Question | Context | 
|-----------------|----------|------|
|carbon monoxide| What does increased oxygen concentrations in the patient’s lungs displace? | Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
|pie| What did Joey eat for breakfast?| Once upon a time, there was a squirrel named Joey. Joey loved to go outside and play with his cousin Jimmy. Joey and Jimmy played silly games together, and were always laughing. One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond. Joey woke up early in the morning to eat some food before they left. Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast. After he ate, he and Jimmy went to the pond. On their way there they saw their friend Jack Rabbit. They dove into the water and swam for several hours. The sun was out, but the breeze was cold. Joey and Jimmy got out of the water and started walking home. Their fur was wet, and the breeze chilled them. When they got home, they dried off, and Jimmy put on his favorite purple shirt. Joey put on a blue shirt with red and green dots. The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed,'|  

```python
# Set the task on T5
t5['t5'].setTask('question ') 


# define Data, add additional tags between sentences
data = ['''
What does increased oxygen concentrations in the patient’s lungs displace? 
context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
''']


#Predict on text data with T5
t5.predict(data)
```

### How to configure T5 task parameter for Squad Context based question answering and pre-process data
`.setTask('question:)` and prefix the context which can be made up of multiple sentences with `context:`

### Example pre-processed input for T5 Squad Context based question answering:
```
question: What does increased oxygen concentrations in the patient’s lungs displace? 
context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
```



## Text Summarization
[Summarization example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more)

`Summarizes` a paragraph into a shorter version with the same semantic meaning, based on [Text summarization](https://arxiv.org/abs/1506.03340)

```python
# Set the task on T5
pipe = nlu.load('summarize')

# define Data, add additional tags between sentences
data = [
'''
The belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth .
''',
'''  Calculus, originally called infinitesimal calculus or "the calculus of infinitesimals", is the mathematical study of continuous change, in the same way that geometry is the study of shape and algebra is the study of generalizations of arithmetic operations. It has two major branches, differential calculus and integral calculus; the former concerns instantaneous rates of change, and the slopes of curves, while integral calculus concerns accumulation of quantities, and areas under or between curves. These two branches are related to each other by the fundamental theorem of calculus, and they make use of the fundamental notions of convergence of infinite sequences and infinite series to a well-defined limit.[1] Infinitesimal calculus was developed independently in the late 17th century by Isaac Newton and Gottfried Wilhelm Leibniz.[2][3] Today, calculus has widespread uses in science, engineering, and economics.[4] In mathematics education, calculus denotes courses of elementary mathematical analysis, which are mainly devoted to the study of functions and limits. The word calculus (plural calculi) is a Latin word, meaning originally "small pebble" (this meaning is kept in medicine – see Calculus (medicine)). Because such pebbles were used for calculation, the meaning of the word has evolved and today usually means a method of computation. It is therefore used for naming specific methods of calculation and related theories, such as propositional calculus, Ricci calculus, calculus of variations, lambda calculus, and process calculus.'''
]


#Predict on text data with T5
pipe.predict(data)
```

| Predicted summary| Text | 
|------------------|-------|
| manchester united face newcastle in the premier league on wednesday . louis van gaal's side currently sit two points clear of liverpool in fourth . the belgian duo took to the dance floor on monday night with some friends .            | the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth . | 


## Binary Sentence similarity/ Paraphrasing
[Binary sentence similarity example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more)
Classify whether one sentence is a re-phrasing or similar to another sentence      
This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf) and based on [MRPC - Binary Paraphrasing/ sentence similarity classification ](https://www.aclweb.org/anthology/I05-5002.pdf)

```
t5 = nlu.load('en.t5.base')
# Set the task on T5
t5['t5'].setTask('mrpc ')

# define Data, add additional tags between sentences
data = [
''' sentence1: We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said .
sentence2: Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11 "
'''
,
'''  
sentence1: I like to eat peanutbutter for breakfast
sentence2: 	I like to play football.
'''
]

#Predict on text data with T5
t5.predict(data)
```
| Sentence1 | Sentence2 | prediction|
|------------|------------|----------|
|We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said .| Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11 " . | equivalent | 
| I like to eat peanutbutter for breakfast| I like to play football | not_equivalent | 


### How to configure T5 task for MRPC and pre-process text
`.setTask('mrpc sentence1:)` and prefix second sentence with `sentence2:`

### Example pre-processed input for T5 MRPC - Binary Paraphrasing/ sentence similarity

```
mrpc 
sentence1: We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said . 
sentence2: Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11",
```



## Regressive Sentence similarity/ Paraphrasing

Measures how similar two sentences are on a scale from 0 to 5 with 21 classes representing a regressive label.     
This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf) and based on[STSB - Regressive semantic sentence similarity](https://www.aclweb.org/anthology/S17-2001/) .

```python
t5 = nlu.load('en.t5.base')
# Set the task on T5
t5['t5'].setTask('stsb ') 

# define Data, add additional tags between sentences
data = [
             
              ''' sentence1:  What attributes would have made you highly desirable in ancient Rome?  
                  sentence2:  How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?'
              '''
             ,
             '''  
              sentence1: What was it like in Ancient rome?
              sentence2: 	What was Ancient rome like?
              ''',
              '''  
              sentence1: What was live like as a King in Ancient Rome??
              sentence2: 	What was Ancient rome like?
              '''

             ]



#Predict on text data with T5
t5.predict(data)

```

| Question1 | Question2 | prediction|
|------------|------------|----------|
|What attributes would have made you highly desirable in ancient Rome?        | How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER? | 0 | 
|What was it like in Ancient rome?  | What was Ancient rome like?| 5.0 | 
|What was live like as a King in Ancient Rome??       | What is it like to live in Rome? | 3.2 | 


### How to configure T5 task for stsb and pre-process text
`.setTask('stsb sentence1:)` and prefix second sentence with `sentence2:`




### Example pre-processed input for T5 STSB - Regressive semantic sentence similarity

```
stsb
sentence1: What attributes would have made you highly desirable in ancient Rome?        
sentence2: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?',
```





## Grammar Checking
[Grammar checking with T5 example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more))
Judges if a sentence is grammatically acceptable.    
Based on [CoLA - Binary Grammatical Sentence acceptability classification](https://nyu-mll.github.io/CoLA/)

```python
pipe = nlu.load('grammar_correctness')
# Set the task on T5
pipe['t5'].setTask('cola sentence: ')
# define Data
data = ['Anna and Mike is going skiing and they is liked is','Anna and Mike like to dance']
#Predict on text data with T5
pipe.predict(data)
```
|sentence  | prediction|
|------------|------------|
| Anna and Mike is going skiing and they is liked is | unacceptable |      
| Anna and Mike like to dance | acceptable | 


## Document Normalization
[Document Normalizer example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/text_pre_processing_and_cleaning/document_normalizer_demo.ipynb)     
The DocumentNormalizer extracts content from HTML or XML documents, applying either data cleansing using an arbitrary number of custom regular expressions either data extraction following the different parameters

```python
pipe = nlu.load('norm_document')
data = '<!DOCTYPE html> <html> <head> <title>Example</title> </head> <body> <p>This is an example of a simple HTML page with one paragraph.</p> </body> </html>'
df = pipe.predict(data,output_level='document')
df
```
|text|normalized_text|
|------|-------------|
| `<!DOCTYPE html> <html> <head> <title>Example</title> </head> <body> <p>This is an example of a simple HTML page with one paragraph.</p> </body> </html>`       |Example This is an example of a simple HTML page with one paragraph.|

## Word Segmenter
[Word Segmenter Example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/multilingual/japanese_ner_pos_and_tokenization.ipynb)     
The WordSegmenter segments languages without any rule-based tokenization such as Chinese, Japanese, or Korean
```python
pipe = nlu.load('ja.segment_words')
# japanese for 'Donald Trump and Angela Merkel dont share many opinions'
ja_data = ['ドナルド・トランプとアンゲラ・メルケルは多くの意見を共有していません']
df = pipe.predict(ja_data, output_level='token')
df

```

|	token|
|--------|
|	ドナルド|
|	・|
|	トランプ|
|	と|
|	アンゲラ|
|	・|
|	メルケル|
|	は|
|	多く|
|	の|
|	意見|
|	を|
|	共有|
|	し|
|	て|
|	い|
|	ませ|
|	ん|


## Installation

```bash
# PyPi
!pip install nlu pyspark==2.4.7
#Conda
# Install NLU from Anaconda/Conda
conda install -c johnsnowlabs nlu
```


# Additional NLU ressources
- [NLU Website](https://nlu.johnsnowlabs.com/)
- [All NLU Tutorial Notebooks](https://nlu.johnsnowlabs.com/docs/en/notebooks)
- [NLU Videos and Blogposts on NLU](https://nlp.johnsnowlabs.com/learn#pythons-nlu-library)
- [NLU on Github](https://github.com/JohnSnowLabs/nlu)


##  NLU 1.0.6 Release Notes
### Trainable Multi Label Classifiers, predict Stackoverflow Tags and much more in 1 Line of with NLU 1.0.6
We are glad to announce NLU 1.0.6 has been released!
NLU 1.0.6 comes with the Multi Label classifier, it can learn to map strings to multiple labels.
The Multi Label Classifier is using Bidirectional GRU and CNNs inside TensorFlow and supports up to 100 classes.

### NLU 1.0.6 New Features
- Multi Label Classifier
   - The Multi Label Classifier learns a 1 to many mapping between text and labels. This means it can predict multiple labels at the same time for a given input string. This is very helpful for tasks similar to content tag prediction (HashTags/RedditTags/YoutubeTags/Toxic/E2e etc..)
   - Support up to 100 classes
   - Pre-trained Multi Label Classifiers are already avaiable as [Toxic](https://nlu.johnsnowlabs.com/docs/en/examples#toxic-classifier) and [E2E](https://nlu.johnsnowlabs.com/docs/en/examples#e2e-classifier) classifiers

####  Multi Label Classifier
- [ Train Multi Label Classifier on E2E dataset Demo](https://colab.research.google.com/drive/15ZqfNUqliRKP4UgaFcRg5KOSTkqrtDXy?usp=sharing)
- [Train Multi Label  Classifier on Stack Overflow Question Tags dataset Demo](https://colab.research.google.com/drive/1Y0pYdUMKSs1ZP0NDcKgVECqkKD9ShIdc?usp=sharing)       
  This model can predict multiple labels for one sentence.
  To train the Multi Label text classifier model, you must pass a dataframe with a ```text``` column and a ```y``` column for the label.   
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


### NLU 1.0.6 Enhancements
- Improved outputs for Toxic and E2E Classifier.
  - by default, all predicted classes and their confidences which are above the threshold will be returned inside of a list in the Pandas dataframe
  - by configuring meta=True, the confidences for all classes will be returned.


### NLU 1.0.6 New Notebooks and Tutorials

- [ Train Multi Label Classifier on E2E dataset](https://colab.research.google.com/drive/15ZqfNUqliRKP4UgaFcRg5KOSTkqrtDXy?usp=sharing)
- [Train Multi Label  Classifier on Stack Overflow Question Tags dataset](https://drive.google.com/file/d/1Nmrncn-y559od3AKJglwfJ0VmZKjtMAF/view?usp=sharing)

### NLU 1.0.6 Bug-fixes
- Fixed a bug that caused ```en.ner.dl.bert``` to be inaccessible
- Fixed a bug that caused ```pt.ner.large``` to be inaccessible
- Fixed a bug that caused USE embeddings not properly beeing configured to document level output when using multiple embeddings at the same time


##  NLU 1.0.5 Release Notes 

### Trainable Part of Speech Tagger (POS), Sentiment Classifier with BERT/USE/ELECTRA sentence embeddings in 1 Line of code! Latest NLU Release 1.0.5
We are glad to announce NLU 1.0.5 has been released!       
This release comes with a **trainable Sentiment classifier** and a **Trainable Part of Speech (POS)** models!       
These Neural Network Architectures achieve the state of the art (SOTA) on most **binary Sentiment analysis** and **Part of Speech Tagging** tasks!       
You can train the Sentiment Model on any of the **100+ Sentence Embeddings** which include **BERT, ELECTRA, USE, Multi Lingual BERT Sentence Embeddings** and many more!       
Leverage this and achieve the state of the art in any of your datasets, all of this in **just 1 line of Python code**

### NLU 1.0.5 New Features
- Trainable Sentiment DL classifier
- Trainable POS

### NLU 1.0.5 New Notebooks and Tutorials 
- [Sentiment Classification Training Demo](https://colab.research.google.com/drive/1f-EORjO3IpvwRAktuL4EvZPqPr2IZ_g8?usp=sharing)
- [Part Of Speech Tagger Training demo](https://colab.research.google.com/drive/1CZqHQmrxkDf7y3rQHVjO-97tCnpUXu_3?usp=sharing)

### Sentiment Classifier Training
[Sentiment Classification Training Demo](https://colab.research.google.com/drive/1f-EORjO3IpvwRAktuL4EvZPqPr2IZ_g8?usp=sharing)

To train the Binary Sentiment classifier model, you must pass a dataframe with a 'text' column and a 'y' column for the label.

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

### Part Of Speech Tagger Training 
[Part Of Speech Tagger Training demo](https://colab.research.google.com/drive/1CZqHQmrxkDf7y3rQHVjO-97tCnpUXu_3?usp=sharing)

```python
fitted_pipe = nlu.load('train.pos').fit(train_df)
preds = fitted_pipe.predict(train_df)
```



### NLU 1.0.5 Installation changes
Starting from version 1.0.5 NLU will not automatically install pyspark for users anymore.      
This enables easier customizing the Pyspark version which makes it easier to use in various cluster enviroments.

To install NLU from now on, please run
```bash
pip install nlu pyspark==2.4.7 
```
or install any pyspark>=2.4.0 with pyspark<3

### NLU 1.0.5 Improvements
- Improved Databricks path handling for loading and storing models.




## NLU  1.0.4 Release Notes 
##  John Snow Labs NLU 1.0.4 : Trainable Named Entity Recognizer (NER) , achieve SOTA in 1 line of code and easy scaling to 100's of Spark nodes
We are glad to announce NLU 1.0.4 releases the State of the Art breaking Neural Network architecture for NER, Char CNNs - BiLSTM - CRF!

```python
#fit and predict in 1 line!
nlu.load('train.ner').fit(dataset).predict(dataset)


#fit and predict in 1 line with BERT!
nlu.load('bert train.ner').fit(dataset).predict(dataset)


#fit and predict in 1 line with ALBERT!
nlu.load('albert train.ner').fit(dataset).predict(dataset)


#fit and predict in 1 line with ELMO!
nlu.load('elmo train.ner').fit(dataset).predict(dataset)

```



Any NLU pipeline stored can now be loaded as pyspark ML pipeline
```python
# Ready for big Data with Spark distributed computing
import pyspark
nlu_pipe.save(path)
pyspark_pipe = pyspark.ml.PipelineModel.load(stored_model_path)
pyspark_pipe.transform(spark_df)
```


### NLU 1.0.4 New Features
- Trainable  [Named Entity Recognizer](https://nlp.johnsnowlabs.com/docs/en/annotators#ner-dl-named-entity-recognition-deep-learning-annotator)
- NLU pipeline loadable as Spark pipelines

### NLU 1.0.4 New Notebooks,Tutorials and Docs
- [NER training demo](https://colab.research.google.com/drive/1_GwhdXULq45GZkw3157fAOx4Wqo-fmFV?usp=sharing)        
- [Multi Class Text Classifier Training Demo updated to showcase usage of different Embeddings](https://colab.research.google.com/drive/12FA2TVvvRWw4pRhxDnK32WAzl9dbF6Qw?usp=sharing)         
- [New Documentation Page on how to train Models with NLU](https://nlu.johnsnowlabs.com/docs/en/training)
- Databricks Notebook showcasing Scaling with NLU


## NLU 1.0.4 Bug Fixes
- Fixed a bug that NER token confidences do not appear. They now appear when nlu.load('ner').predict(df, meta=True) is called.
- Fixed a bug that caused some Spark NLP models to not be loaded properly in offline mode



## 1.0.3 Release Notes 
We are happy to announce NLU 1.0.3 comes with a lot new features, training classifiers, saving them and loading them offline, enabling running NLU with no internet connection, new notebooks and articles!

### NLU 1.0.3 New Features
- Train a Deep Learning classifier in 1 line! The popular [ClassifierDL](https://nlp.johnsnowlabs.com/docs/en/annotators#classifierdl-multi-class-text-classification)
which can achieve state of the art results on any multi class text classification problem is now trainable!
All it takes is just nlu.load('train.classifier).fit(dataset) . Your dataset can be a Pandas/Spark/Modin/Ray/Dask dataframe and needs to have a column named x for text data and a column named y for labels
- Saving pipelines to HDD is now possible with nlu.save(path)
- Loading pipelines from disk now possible with nlu.load(path=path). 
- NLU offline mode: Loading from disk makes running NLU offline now possible, since you can load pipelines/models from your local hard drive instead of John Snow Labs AWS servers.

### NLU 1.0.3 New Notebooks and Tutorials
- New colab notebook showcasing nlu training, saving and loading from disk
- [Sentence Similarity with BERT, Electra and Universal Sentence Encoder Medium Tutorial](https://medium.com/spark-nlp/easy-sentence-similarity-with-bert-sentence-embeddings-using-john-snow-labs-nlu-ea078deb6ebf)
- [Sentence Similarity with BERT, Electra and Universal Sentence Encoder](https://colab.research.google.com/drive/1LtOdtXtRJ3_N8kYywPd5k2AJMCGcgAdN?usp=sharing)
- [Train a Deep Learning Classifier ](https://colab.research.google.com/drive/12FA2TVvvRWw4pRhxDnK32WAzl9dbF6Qw?usp=sharing)
- [Sentence Detector Notebook Updated](https://colab.research.google.com/drive/1CAXEdRk_q3U5qbMXsxoVyZRwvonKthhF?usp=sharing)
- [New Workshop video](https://events.johnsnowlabs.com/cs/c/?cta_guid=8b2b188b-92a3-48ba-ad7e-073b384425b0&signature=AAH58kFAHrVT-HfvWFxdTg_lm8reKUdTBw&pageId=25538044150&placement_guid=c659363c-2188-4c86-945f-5cfb7b42fcfc&click=8cd42d22-2f03-4358-a9e8-0d8f9aa33139&hsutk=c7a000001cda197314f90175e307161f&canon=https%3A%2F%2Fevents.johnsnowlabs.com%2Fwebinars&utm_referrer=https%3A%2F%2Fwww.johnsnowlabs.com%2F&portal_id=1794529&redirect_url=APefjpGh4Q9Hy0Mg9Ezy0_kJOOLC3l5QYyJsCSfZc1Lf61qrn2Bk6OQIJj65atZ9zzzrNrxuDPk5EHt94G0ZcIJaP_QMuD_E7fnMeJs4bQrEdLl7HE2MC4WNHGB6t1cqABfjZntS_TYSaj02yJNDf6p7Zaj9OYy0qQCmM8bbeuVgxUe6s5946UqHDsVHrpY0Oa2Fs7DJXIahZsB08hGkVj3qSHIM5vpjsA)


### NLU 1.0.3 Bug fixes
- Sentence Detector bugfix 




## NLU 1.0.2 Release Notes 

We are glad to announce nlu 1.0.2 is released!

### NLU 1.0.2  Enhancements
- More semantically concise output levels sentence and document enforced : 
  - If a pipe is set to output_level='document' : 
    -  Every Sentence Embedding will generate 1 Embedding per Document/row in the input Dataframe, instead of 1 embedding per sentence. 
    - Every  Classifier will classify an entire Document/row 
    - Each row in the output DF is a 1 to 1 mapping of the original input DF. 1 to 1 mapping from input to output.
  - If a pipe is set to output_level='sentence' : 
    -  Every Sentence Embedding will generate 1 Embedding per Sentence, 
    - Every  Classifier will classify exactly one sentence
    - Each row in the output DF can is mapped to one row in the input DF, but one row in the input DF can have multiple corresponding rows in the output DF. 1 to N mapping from input to output.
- Improved generation of column names for classifiers. based on input nlu reference
- Improved generation of column names for embeddings, based on input nlu reference
- Improved automatic output level inference
- Various test updates
- Integration of CI pipeline with Github Actions 

### New  Documentation is out!
Check it out here :  http://nlu.johnsnowlabs.com/


## NLU 1.0.1 Release Notes 

### NLU 1.0.1 Bugfixes
- Fixed bug that caused NER pipelines to crash in NLU when input string caused the NER model to predict without additional metadata

## 1.0 Release Notes 
- Automatic to Numpy conversion of embeddings
- Added various testing classes
- [New 6 embeddings at once notebook with t-SNE and Medium article](https://medium.com/spark-nlp/1-line-of-code-for-bert-albert-elmo-electra-xlnet-glove-part-of-speech-with-nlu-and-t-sne-9ebcd5379cd)
 <img src="https://miro.medium.com/max/1296/1*WI4AJ78hwPpT_2SqpRpolA.png" >
- Integration of Spark NLP 2.6.2 enhancements and bugfixes https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.2
- Updated old T-SNE notebooks with more elegant and simpler generation of t-SNE embeddings 

</div><div class="h3-box" markdown="1">

## 0.2.1 Release Notes 
- Various bugfixes
- Improved output column names when using multiple classifirs at once

</div><div class="h3-box" markdown="1">

## 0.2 Release Notes 
-   Improved output column names  classifiers

</div><div class="h3-box" markdown="1">
    
## 0.1 Release Notes



# 1.0 Release Notes 
- Automatic to Numpy conversion of embeddings
- Added various testing classes
- [New 6 embeddings at once notebook with t-SNE and Medium article](https://medium.com/spark-nlp/1-line-of-code-for-bert-albert-elmo-electra-xlnet-glove-part-of-speech-with-nlu-and-t-sne-9ebcd5379cd)
 <img src="https://miro.medium.com/max/1296/1*WI4AJ78hwPpT_2SqpRpolA.png" >
- Integration of Spark NLP 2.6.2 enhancements and bugfixes https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.2
- Updated old T-SNE notebooks with more elegant and simpler generation of t-SNE embeddings 

# 0.2.1 Release Notes 
- Various bugfixes
- Improved output column names when using multiple classifirs at once

# 0.2 Release Notes 
-   Improved output column names  classifiers
    
# 0.1 Release Notes
We are glad to announce that NLU 0.0.1 has been released!
NLU makes the 350+ models and annotators in Spark NLPs arsenal available in just 1 line of python code and it works with Pandas dataframes!
A picture says more than a 1000 words, so here is a demo clip of the 12 coolest features in NLU, all just in 1 line!

</div><div class="h3-box" markdown="1">

## NLU in action 
<img src="http://ckl-it.de/wp-content/uploads/2020/08/My-Video6.gif" width="1800" height="500"/>

</div><div class="h3-box" markdown="1">

## What does NLU 0.1 include?

## NLU in action 
<img src="http://ckl-it.de/wp-content/uploads/2020/08/My-Video6.gif" width="1800" height="500"/>

# What does NLU 0.1 include?
 - NLU provides everything a data scientist might want to wish for in one line of code!
 - 350 + pre-trained models
 - 100+ of the latest NLP word embeddings ( BERT, ELMO, ALBERT, XLNET, GLOVE, BIOBERT, ELECTRA, COVIDBERT) and different variations of them
 - 50+ of the latest NLP sentence embeddings ( BERT, ELECTRA, USE) and different variations of them
 - 50+ Classifiers (NER, POS, Emotion, Sarcasm, Questions, Spam)
 - 40+ Supported Languages
 - Labeled and Unlabeled Dependency parsing
 - Various Text Cleaning and Pre-Processing methods like Stemming, Lemmatizing, Normalizing, Filtering, Cleaning pipelines and more

 </div><div class="h3-box" markdown="1">

## NLU 0.1 Features Google Collab Notebook Demos

- Named Entity Recognition (NER)
    - [NER pretrained on ONTO Notes](https://colab.research.google.com/drive/1_sgbJV3dYPZ_Q7acCgKWgqZkWcKAfg79?usp=sharing)
    - [NER pretrained on CONLL](https://colab.research.google.com/drive/1CYzHfQyFCdvIOVO2Z5aggVI9c0hDEOrw?usp=sharing)
</div><div class="h3-box" markdown="1">

- Part of speech (POS)
    - [POS pretrained on ANC dataset](https://colab.research.google.com/drive/1tW833T3HS8F5Lvn6LgeDd5LW5226syKN?usp=sharing)

</div><div class="h3-box" markdown="1">

# NLU 0.1 Features Google Collab Notebook Demos

- Named Entity Recognition (NER)
    -[NER pretrained on ONTO Notes](https://colab.research.google.com/drive/1_sgbJV3dYPZ_Q7acCgKWgqZkWcKAfg79?usp=sharing)
    -[NER pretrained on CONLL](https://colab.research.google.com/drive/1CYzHfQyFCdvIOVO2Z5aggVI9c0hDEOrw?usp=sharing)
- Part of speech (POS)
    - [POS pretrained on ANC dataset](https://colab.research.google.com/drive/1tW833T3HS8F5Lvn6LgeDd5LW5226syKN?usp=sharing)
- Classifiers
    - [Unsupervised Keyword Extraction with YAKE](https://colab.research.google.com/drive/1BdomIc1nhrGxLFOpK5r82Zc4eFgnIgaO?usp=sharing)
    - [Toxic Text Classifier](https://colab.research.google.com/drive/1QRG5ZtAvoJAMZ8ytFMfXj_W8ogdeRi9m?usp=sharing)
    - [Twitter Sentiment Classifier](https://colab.research.google.com/drive/1H1Gekn2qzXzOf5rrT8LmHmmuoOGsiu8m?usp=sharing)
    - [Movie Review Sentiment Classifier](https://colab.research.google.com/drive/1k5x1zxnG4bBkmYAc-bc63sMA4-oQ6-dP?usp=sharing)
    - [Sarcasm Classifier](https://colab.research.google.com/drive/1XffsjlRp9wxZgxyYvEF9bG2CiX-pjBEw?usp=sharing)
    - [50 Class Questions Classifier](https://colab.research.google.com/drive/1OwlmLzwkcJKhuz__RUH74O9HqFZutxzS?usp=sharing)
    - [20 Class Languages Classifier](https://colab.research.google.com/drive/1CzMfRFJZsj4j1fhormDQdHOIV5IybC57?usp=sharing)
    - [Fake News Classifier](https://colab.research.google.com/drive/1QuoeGLgmtkUnDQQ2oVS1tuZC2qHDj3p9?usp=sharing)
    - [E2E Classifier](https://colab.research.google.com/drive/1OSkiXGEpKlm9HWDoVb42uLNQQgb7nqNZ?usp=sharing)
    - [Cyberbullying Classifier](https://colab.research.google.com/drive/1OSkiXGEpKlm9HWDoVb42uLNQQgb7nqNZ?usp=sharing)
    - [Spam Classifier](https://colab.research.google.com/drive/1u-8Fs3Etz07bFNx0CDV_le3Xz73VbK0z?usp=sharing)

</div><div class="h3-box" markdown="1">

- Word and Sentence Embeddings 
    - [BERT Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1Rg1vdSeq6sURc48RV8lpS47ja0bYwQmt?usp=sharing)
    - [BERT Sentence Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1FmREx0O4BDeogldyN74_7Lur5NeiOVye?usp=sharing)
    - [ALBERT Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/18yd9pDoPkde79boTbAC8Xd03ROKisPsn?usp=sharing)
    - [ELMO Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1TtNYB9z0yH8d1ZjfxkH0TVxQ2O_iOYVV?usp=sharing)
    - [XLNET Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1C9T29QA00yjLuJ1yEMTbjUQMpUv35pHb?usp=sharing)
    - [ELECTRA Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1FueGEaOj2JkbqHzdmxwKrNMHzgVt4baE?usp=sharing)
    - [COVIDBERT Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1Yzc-GuNQyeWewJh5USTN7PbbcJvd-D7s?usp=sharing)
    - [BIOBERT Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1llANd-XGD8vkGNMcqTi_8Dr_Ys6cr83W?usp=sharing)
    - [GLOVE Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1IQxf4pJ_EnrIDyd0fAX-dv6u0YQWae2g?usp=sharing)
    - [USE Sentence Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1gZzOMiCovmrp7z8FIidzDTLS0nt8kPJT?usp=sharing)

</div><div class="h3-box" markdown="1">

- Depenency Parsing 
    - [Untyped Dependency Parsing](https://colab.research.google.com/drive/1PC8ga_NFlOcTNeDVJY4x8Pl5oe0jVmue?usp=sharing)
    - [Typed Dependency Parsing](https://colab.research.google.com/drive/1KXUqcF8e-LU9cXnHE8ni8z758LuFPvY7?usp=sharing)

</div><div class="h3-box" markdown="1">

- Depenency Parsing 
    -[Untyped Dependency Parsing](https://colab.research.google.com/drive/1PC8ga_NFlOcTNeDVJY4x8Pl5oe0jVmue?usp=sharing)
    -[Typed Dependency Parsing](https://colab.research.google.com/drive/1KXUqcF8e-LU9cXnHE8ni8z758LuFPvY7?usp=sharing)

- Text Pre Processing and Cleaning
    - [Tokenization](https://colab.research.google.com/drive/13BC6k6gLj1w5RZ0SyHjKsT2EOwJwbYwb?usp=sharing)
    - [Stopwords removal](https://colab.research.google.com/drive/1nWob4u93t2EJYupcOIanuPBDfShtYjGT?usp=sharing)
    - [Stemming](https://colab.research.google.com/drive/1gKTJJmffR9wz13Ms3pDy64jhUI8ZHZYu?usp=sharing)
    - [Lemmatization](https://colab.research.google.com/drive/1cBtx9cVCjavt-Oq5TG1lO-9JfUfqznnK?usp=sharing)
    - [Normalizing](https://colab.research.google.com/drive/1kfnnwkiQPQa465Jic6va9QXTRssU4mlX?usp=sharing)
    - [Spellchecking](https://colab.research.google.com/drive/1bnRR8FygiiN3zJz3mRdbjPBUvFsx6IVB?usp=sharing)
    - [Sentence Detecting](https://colab.research.google.com/drive/1CAXEdRk_q3U5qbMXsxoVyZRwvonKthhF?usp=sharing)

</div><div class="h3-box" markdown="1">

- Chunkers
    - [N Gram](https://colab.research.google.com/drive/1pgqoRJ6yGWbTLWdLnRvwG5DLSU3rxuMq?usp=sharing)
    - [Entity Chunking](https://colab.research.google.com/drive/1svpqtC3cY6JnRGeJngIPl2raqxdowpyi?usp=sharing)

</div><div class="h3-box" markdown="1">

- Matchers
    - [Date Matcher](https://colab.research.google.com/drive/1JrlfuV2jNGTdOXvaWIoHTSf6BscDMkN7?usp=sharing)

</div><div class="h3-box" markdown="1">


</div></div>
- Chunkers
    -[N Gram](https://colab.research.google.com/drive/1pgqoRJ6yGWbTLWdLnRvwG5DLSU3rxuMq?usp=sharing)
    -[Entity Chunking](https://colab.research.google.com/drive/1svpqtC3cY6JnRGeJngIPl2raqxdowpyi?usp=sharing)
- Matchers
    -[Date Matcher](https://colab.research.google.com/drive/1JrlfuV2jNGTdOXvaWIoHTSf6BscDMkN7?usp=sharing)



