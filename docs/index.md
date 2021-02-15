---
layout: landing
title: 'NLU: <span>State of the Art Text Mining in Python</span>'
excerpt:  <br> The Simplicity of Python, the Power of Spark NLP
permalink: /
header: true
article_header:
 actions:
   - text: Getting Started
     type: active
     url: /docs/en/install   
   - text: '<i class="fab fa-github"></i> GitHub'
     type: trans
     url: https://github.com/JohnSnowLabs/nlu 
   - text: '<i class="fab fa-slack-hash"></i> Slack'
     type: trans
     url: https://app.slack.com/client/T9BRVC9AT/C0196BQCDPY   

 height: 50vh
 theme: dark

data:
 sections:
   - title:
     children:
       - title: Powerful One-Liners
         image: 
            src: /assets/images/powerfull_one.svg
         excerpt: Over a thousand NLP models in hundreds of languages are at your fingertips with just one line of code
       - title: Elegant Python
         image: 
            src: /assets/images/elegant_python.svg
         excerpt: Directly read and write pandas dataframes for frictionless integration with other libraries and existing ML pipelines  
       - title: 100% Open Source
         image: 
            src: /assets/images/open_source.svg
         excerpt: Including pre-trained models & pipelines

   - title: 'Quick and Easy'
     install: yes
     excerpt: NLU is available on <a href="https://pypi.org/project/nlu" target="_blank">PyPI</a>, <a href="https://anaconda.org/JohnSnowLabs/nlu" target="_blank">Conda</a>
     actions:
       - text: Install NLU
         type: big_btn
         url: /docs/en/install
  
  
   - title: Benchmark
     excerpt: NLU is based on the award winning Spark NLP which best performing in peer-reviewed results
     benchmark: yes
     features: false
     theme: dark

   - title: Pandas
     excerpt: NLU ships with many <b>NLP features</b>, pre-trained <b>models</b> and <b>pipelines</b> <div>It takes in Pandas and outputs <b>Pandas Dataframes</b></div><div>All in <b>one line</b></div>
     pandas: yes
     theme: dark

    
---




### Named Entity Recognition (NER) 18 class

{:.contant-descr}
[NER ONTO example](https://colab.research.google.com/drive/1_sgbJV3dYPZ_Q7acCgKWgqZkWcKAfg79?usp=sharing)


```python
nlu.load('ner').predict('Angela Merkel from Germany and the American Donald Trump dont share many opinions')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|embeddings | 	ner_tag | 	entities |
|-----------|------------|------------|
|[[-0.563759982585907, 0.26958999037742615, 0.3...	| PER| 	Angela Merkel |
|[[-0.563759982585907, 0.26958999037742615, 0.3...	| LOC| 	Germany |
|[[-0.563759982585907, 0.26958999037742615, 0.3...	| MISC| 	American |
|[[-0.563759982585907, 0.26958999037742615, 0.3...	| PER| 	Donald Trump |

</div></div>

### Named Entity Recognition (NER) 5 Class

{:.contant-descr}
[NER CONLL example](https://colab.research.google.com/drive/1CYzHfQyFCdvIOVO2Z5aggVI9c0hDEOrw?usp=sharing)


```python
nlu.load('ner.conll').predict('Angela Merkel from Germany and the American Donald Trump dont share many opinions')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|embeddings| 	ner_tag| 	entities| 
|----------|-----------|-------------|
|[[-0.563759982585907, 0.26958999037742615, 0.3...	|PER |	Angela Merkel | 
|[[-0.563759982585907, 0.26958999037742615, 0.3...	|LOC |	Germany | 
|[[-0.563759982585907, 0.26958999037742615, 0.3...	|MISC |	American | 
|[[-0.563759982585907, 0.26958999037742615, 0.3...	|PER |	Donald Trump | 

</div></div>


### Part of speech (POS)

{:.contant-descr}
POS Classifies each token with one of the following tags    
[Part of Speech example](https://colab.research.google.com/drive/1tW833T3HS8F5Lvn6LgeDd5LW5226syKN?usp=sharing)


```python
nlu.load('pos').predict('Part of speech assigns each token in a sentence a grammatical label')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}

|token |pos    |
|------|-----|
|Part|         NN|     
|of|           IN|     
|speech|       NN|     
|assigns|      NNS|    
|each|         DT|     
|token|            NN| 
|in|           IN|     
|a |           DT|     
|sentence|     NN|     
|a |           DT|     
|grammatical|  JJ|     
|label          |NN|   

</div></div>


### Emotion Classifier

{:.contant-descr}
[Emotion Classifier example](https://colab.research.google.com/drive/1eBf3MN_O8uJnimK6GeweksXl6JHYKzOT?usp=sharing)         
Classifies text as one of 4 categories (joy, fear, surprise, sadness)



```python
nlu.load('emotion').predict('I love NLU!')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|sentence_embeddings|  emotion_confidence|   sentence|  emotion|
|--------------------|---------------------|------------|------------|
|[0.027570432052016258, -0.052647676318883896, ...]    |0.976017  |I love NLU!   |joy   |

</div></div>

### Sentiment Classifier

{:.contant-descr}
[Sentiment Classifier Example](https://colab.research.google.com/drive/1k5x1zxnG4bBkmYAc-bc63sMA4-oQ6-dP?usp=sharing)   
Classifies binary sentiment for every sentence, either positive or negative.      

```python
nlu.load('sentiment').predict("I hate this guy Sami")
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|sentiment_confidence  |sentence  |sentiment |checked |
|-----------|----------------------|---------|---------|
|0.5778 |  I hate this guy Sami   | negative |    [I, hate, this, guy, Sami] |

</div></div>

### Question Classifier 50 class

{:.contant-descr}
[50 Class Questions Classifier example](https://colab.research.google.com/drive/1OwlmLzwkcJKhuz__RUH74O9HqFZutxzS?usp=sharing)    
Classify between 50 different types of questions trained on Trec50     
When setting predict(meta=True) nlu will output the probabilities for all other 49 question classes.

```python
nlu.load('en.classify.trec50').predict('How expensive is the Watch?')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|	sentence_embeddings| 	question_confidence| 	sentence| 	question|
|----------------------|----------------------|-------------|----------|
|[0.051809534430503845, 0.03128402680158615, -0...]|	0.919436 | 	How expensive is the watch?| 	NUM_count	|

</div></div>

### Fake News Classifier

{:.contant-descr}
[Fake News Classifier example](https://colab.research.google.com/drive/1k5x1zxnG4bBkmYAc-bc63sMA4-oQ6-dP?usp=sharing)

```python
nlu.load('en.classify.fakenews').predict('Unicorns have been sighted on Mars!')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|sentence_embeddings|  fake_confidence|   sentence|  fake|
|------------------|-----------------------|------------|-----------|
|[-0.01756167598068714, 0.015006818808615208, -...]    | 1.000000 | Unicorns have been sighted on Mars!  |FAKE  |

</div></div>

### Cyberbullying Classifier

{:.contant-descr}
[Cyberbullying Classifier example](https://colab.research.google.com/drive/1OSkiXGEpKlm9HWDoVb42uLNQQgb7nqNZ?usp=sharing)   
Classifies sexism and racism
```python
nlu.load('en.classify.cyberbullying').predict('Women belong in the kitchen.') # sorry we really don't mean it
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|sentence_embeddings| 	cyberbullying_confidence| 	sentence| 	cyberbullying|
|-------------------|----------------------|------------|-----------|
|[-0.054944973438978195, -0.022223370149731636,...]|	0.999998 	| Women belong in the kitchen. | 	sexism|

</div></div>

### Spam Classifier

{:.contant-descr}
[Spam Classifier example](https://colab.research.google.com/drive/1u-8Fs3Etz07bFNx0CDV_le3Xz73VbK0z?usp=sharing)

```python
nlu.load('en.classify.spam').predict('Please sign up for this FREE membership it costs $$NO MONEY$$ just your mobile number!')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|sentence_embeddings|  spam_confidence|   sentence| spam |
|-------------------|----------------------|------------|-----------|
|[0.008322705514729023, 0.009957313537597656, 0...]    | 1.000000 | Please sign up for this FREE membership it cos...    |spam  |

</div></div>

### Sarcasm Classifier

{:.contant-descr}
[Sarcasm Classifier example](https://colab.research.google.com/drive/1XffsjlRp9wxZgxyYvEF9bG2CiX-pjBEw?usp=sharing)

```python
nlu.load('en.classify.sarcasm').predict('gotta love the teachers who give exams on the day after halloween')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
| sentence_embeddings |    sarcasm_confidence  |     sentence |     sarcasm  |
|---------------------|--------------------------|-----------|-----------|
|[-0.03146284446120262, 0.04071342945098877, 0....] | 0.999985 | gotta love the teachers who give exams on the...    | sarcasm  |

</div></div>

### IMDB Movie Sentiment Classifier

{:.contant-descr}
[Movie Review Sentiment Classifier example](https://colab.research.google.com/drive/1k5x1zxnG4bBkmYAc-bc63sMA4-oQ6-dP?usp=sharing)

```python
nlu.load('en.sentiment.imdb').predict('The Matrix was a pretty good movie')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|document |   sentence_embeddings|   sentiment_negative|    sentiment_negative|    sentiment_positive|    sentiment |
|-------|-----|-------------------------|---------------------|----------------------|-------------------------|-------------|
|The Matrix was a pretty good movie    | [[0.04629608988761902, -0.020867452025413513, ...  ]| [2.7235753918830596e-07]     | [2.7235753918830596e-07] |  [0.9999997615814209]|  [positive] |

</div></div>

### Twitter Sentiment Classifier

{:.contant-descr}
[Twitter Sentiment Classifier Example](https://colab.research.google.com/drive/1H1Gekn2qzXzOf5rrT8LmHmmuoOGsiu8m?usp=sharing)

```python
nlu.load('en.sentiment.twitter').predict('@elonmusk Tesla stock price is too high imo')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|document |   sentence_embeddings |  sentiment_negative |   sentiment_negative|    sentiment_positive |   sentiment|
|--------|---------|-----------------------|-----------------------|---------------------|-------------------------|-------------|
| @elonmusk Tesla stock price is too high imo  |  [[0.08604438602924347, 0.04703635722398758, -0...]|  [1.0] |    [1.0]  | [1.692714735043349e-36]  | [negative]|

</div></div>

### Language Classifier

{:.contant-descr}
[Languages Classifier example](https://colab.research.google.com/drive/1CzMfRFJZsj4j1fhormDQdHOIV5IybC57?usp=sharing)         
Classifies the following 20 languages:        
 Bulgarian, Czech, German, Greek, English, Spanish, Finnish, French, Croatian, Hungarian, Italy, Norwegian, Polish, Portuguese, Romanian, Russian, Slovak, Swedish, Turkish, and Ukrainian

```python
nlu.load('lang').predict(['NLU is an open-source text processing library for advanced natural language processing for the Python.','NLU est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python.'])
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|language_confidence|  document|  language|
|------------------|-----------|-------------|
|0.985407  |NLU is an open-source text processing library ...]|   en|    
|0.999822  |NLU est une bibliothèque de traitement de text...]|   fr|    

</div></div>

### E2E Classifier

{:.contant-descr}
[E2E Classifier example](https://colab.research.google.com/drive/1OSkiXGEpKlm9HWDoVb42uLNQQgb7nqNZ?usp=sharing)   
This is a multi class classifier trained on the E2E [dataset for Natural language generation](http://www.macs.hw.ac.uk/InteractionLab/E2E/#)

```python
nlu.load('e2e').predict('E2E is a dataset for training generative models')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|sentence_embeddings | 	e2e | 	e2e_confidence| 	sentence|
|--------------------|------|-----------------|--------------|
|[0.021445205435156822, -0.039284929633140564, ...,]|	customer rating[high]| 	0.703248 | 	E2E is a dataset for training generative models | 
|None|	name[The Waterman]	| 0.703248	|None|
|None|	eatType[restaurant]	| 0.703248	|None|
|None|	priceRange[£20-25]	| 0.703248	|None|
|None|	familyFriendly[no]	| 0.703248	|None|
|None|	familyFriendly[yes]	| 0.703248	|None|

</div></div>

###  Toxic Classifier

{:.contant-descr}
[Toxic Text Classifier example](https://colab.research.google.com/drive/1QRG5ZtAvoJAMZ8ytFMfXj_W8ogdeRi9m?usp=sharing)

```python
nlu.load('en.classify.toxic').predict('You are to stupid')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|	toxic_confidence | 	toxic | 	sentence_embeddings| 	document| 
|-------------------|---------|------------------------|------------|
| 0.978273 | 	[toxic,insult]	| [[-0.03398505970835686, 0.0007853527786210179,...,]	You are to stupid|

</div></div>

### Word Embeddings Bert

{:.contant-descr}
[BERT Word Embeddings example](https://colab.research.google.com/drive/1Rg1vdSeq6sURc48RV8lpS47ja0bYwQmt?usp=sharing)

```python
nlu.load('bert').predict('NLU offers the latest embeddings in one line ')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token         |bert_embeddings                                         |
|--------------|--------------------------------------------------------|
|  NLU        |   [0.3253086805343628, -0.574441134929657, -0.08...] |
|  offers     |   [-0.6660361886024475, -0.1494743824005127, -0...]  |
|  the        |   [-0.6587662696838379, 0.3323703110218048, 0.16...] |
|  latest     |   [0.7552685737609863, 0.17207926511764526, 1.35...] |
|  embeddings |   [-0.09838500618934631, -1.1448147296905518, -1...] |
|  in         |   [-0.4635896384716034, 0.38369956612586975, 0.0...] |
|  one        |   [0.26821616291999817, 0.7025910019874573, 0.15...] |
|  line       |   [-0.31930840015411377, -0.48271292448043823, 0...] |

</div></div>

### Word Embeddings Biobert 

{:.contant-descr}
[BIOBERT Word Embeddings example](https://colab.research.google.com/drive/1llANd-XGD8vkGNMcqTi_8Dr_Ys6cr83W?usp=sharing)         
Bert model pretrained on Bio dataset

```python
nlu.load('biobert').predict('Biobert was pretrained on a medical dataset')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token         |biobert_embeddings                                         |
|--------------|--------------------------------------------------------|
|  NLU        |   [0.3253086805343628, -0.574441134929657, -0.08...] |
|  offers     |   [-0.6660361886024475, -0.1494743824005127, -0...]  |
|  the        |   [-0.6587662696838379, 0.3323703110218048, 0.16...] |
|  latest     |   [0.7552685737609863, 0.17207926511764526, 1.35...] |
|  embeddings |   [-0.09838500618934631, -1.1448147296905518, -1...] |
|  in         |   [-0.4635896384716034, 0.38369956612586975, 0.0...] |
|  one        |   [0.26821616291999817, 0.7025910019874573, 0.15...] |
|  line       |   [-0.31930840015411377, -0.48271292448043823, 0...] |

</div></div>

### Word Embeddings Covidbert

{:.contant-descr}
[COVIDBERT Word Embeddings](https://colab.research.google.com/drive/1Yzc-GuNQyeWewJh5USTN7PbbcJvd-D7s?usp=sharing)    
Bert model pretrained on COVID dataset

```python
nlu.load('covidbert').predict('Albert uses a collection of many berts to generate embeddings')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|	token | 	covid_embeddings| 
|---------|--------------------|
|He| 	[-1.0551927089691162, -1.534174919128418, 1.29...,] | 
|was| 	[-0.14796507358551025, -1.3928604125976562, 0....,] | 
|suprised| 	[1.0647121667861938, -0.3664901852607727, 0.54...,] | 
|by| 	[-0.15271103382110596, -0.6812090277671814, -0...,] | 
|the| 	[-0.45744237303733826, -1.4266574382781982, -0...,] | 
|diversity| 	[-0.05339818447828293, -0.5118572115898132, 0....,] | 
|of| 	[-0.2971905767917633, -1.0936176776885986, -0....,] | 
|NLU| 	[-0.9573594331741333, -0.18001675605773926, -1...,] | 

</div></div>

### Word Embeddings Albert

{:.contant-descr}
[ALBERT Word Embeddings examle](https://colab.research.google.com/drive/18yd9pDoPkde79boTbAC8Xd03ROKisPsn?usp=sharing)

```python
nlu.load('albert').predict('Albert uses a collection of many berts to generate embeddings')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token |   albert_embeddings |
|-----|----------------------|
| Albert |         [-0.08257609605789185, -0.8017427325248718, 1...]   |
| uses |       [0.8256351947784424, -1.5144840478897095, 0.90...]  |
| a |          [-0.22089454531669617, -0.24295514822006226, 3...]  |
| collection |     [-0.2136894017457962, -0.8225528597831726, -0...]   |
| of |         [1.7623294591903687, -1.113651156425476, 0.800...]  |
| many |       [0.6415284872055054, -0.04533941298723221, 1.9...]  |
| berts |      [-0.5591965317726135, -1.1773797273635864, -0...]   |
| to |             [1.0956681966781616, -1.4180747270584106, -0.2...]  |
| generate |   [-0.6759272813796997, -1.3546931743621826, 1.6...]  |
| embeddings |     [-0.0035803020000457764, -0.35928264260292053,...]  |

</div></div>

### Electra Embeddings

{:.contant-descr}
[ELECTRA Word Embeddings example](https://colab.research.google.com/drive/1FueGEaOj2JkbqHzdmxwKrNMHzgVt4baE?usp=sharing)

```python
nlu.load('electra').predict('He was suprised by the diversity of NLU')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token | 	electra_embeddings | 
|------|---------------|
|He | 	[0.29674115777015686, -0.21371933817863464, -0...,]|
|was | 	[-0.4278327524662018, -0.5352768898010254, -0....,]|
|suprised | 	[-0.3090559244155884, 0.8737565279006958, -1.0...,]|
|by | 	[-0.07821277529001236, 0.13081523776054382, 0....,]|
|the | 	[0.5462881922721863, 0.0683358758687973, -0.41...,]|
|diversity | 	[0.1381239891052246, 0.2956242859363556, 0.250...,]|
|of | 	[-0.5667567253112793, -0.3955455720424652, -0....,]|
|NLU | 	[0.5597224831581116, -0.703249454498291, -1.08...,]|

</div></div>

### Word Embeddings Elmo

{:.contant-descr}
[ELMO Word Embeddings example](https://colab.research.google.com/drive/1TtNYB9z0yH8d1ZjfxkH0TVxQ2O_iOYVV?usp=sharing)

```python
nlu.load('elmo').predict('Elmo was trained on Left to right masked to learn its embeddings')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token|    elmo_embeddings     |
|------|-----------------|
|Elmo |    [0.6083735227584839, 0.20089012384414673, 0.42...]  |
|was | [0.2980785369873047, -0.07382500916719437, -0...]   |
|trained | [-0.39923471212387085, 0.17155063152313232, 0...]   |
|on |  [0.04337821900844574, 0.1392083466053009, -0.4...]  |
|Left |    [0.4468783736228943, -0.623046875, 0.771505534...]  |
|to |  [-0.18209676444530487, 0.03812692314386368, 0...]   |
|right |   [0.23305709660053253, -0.6459438800811768, 0.5...]  |
|masked |  [-0.7243442535400391, 0.10247116535902023, 0.1...]  |
|to |  [-0.18209676444530487, 0.03812692314386368, 0...]   |
|learn |   [1.2942464351654053, 0.7376189231872559, -0.58...]  |
|its | [0.055951207876205444, 0.19218483567237854, -0...]  |
|embeddings |  [-1.31377112865448, 0.7727609872817993, 0.6748...]  |

</div></div>

### Word Embeddings Xlnet

{:.contant-descr}
[XLNET Word Embeddings example](https://colab.research.google.com/drive/1C9T29QA00yjLuJ1yEMTbjUQMpUv35pHb?usp=sharing)

```python
nlu.load('xlnet').predict('XLNET computes contextualized word representations using combination of Autoregressive Language Model and Permutation Language Model')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
| token    | xlnet_embeddings |
|------|--------------------|
|XLNET |   [-0.02719488926231861, -1.7693557739257812, -0...] |
|computes |    [-1.8262947797775269, 0.8455266356468201, 0.57...] |
|contextualized |  [2.8446314334869385, -0.3564329445362091, -2.1...] |
|word |    [-0.6143839359283447, -1.7368144989013672, -0...]  |
|representations |     [-0.30445945262908936, -1.2129613161087036, 0...]  |
|using |   [0.07423821836709976, -0.02561005763709545, -0...] |
|combination |     [-0.5387097597122192, -1.1827564239501953, 0.5...] |
|of |  [-1.403516411781311, 0.3108177185058594, -0.32...] |
|Autoregressive |  [-1.0869172811508179, 0.7135171890258789, -0.2...] |
|Language |    [-0.33215752243995667, -1.4108021259307861, -0...] |
|Model |   [-1.6097160577774048, -0.2548254430294037, 0.0...] |
|and |     [0.7884324789047241, -1.507911205291748, 0.677...] |
|Permutation |     [0.6049966812133789, -0.157279372215271, -0.06...] |
|Language |    [-0.33215752243995667, -1.4108021259307861, -0...] |
|Model |   [-1.6097160577774048, -0.2548254430294037, 0.0...] |

</div></div>

### Word Embeddings Glove 

{:.contant-descr}
[GLOVE Word Embeddings example](https://colab.research.google.com/drive/1IQxf4pJ_EnrIDyd0fAX-dv6u0YQWae2g?usp=sharing)

```python
nlu.load('glove').predict('Glove embeddings are generated by aggregating global word-word co-occurrence matrix from a corpus')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|  token    |glove_embeddings    |
|---------|------------------|
|Glove |   [0.3677999973297119, 0.37073999643325806, 0.32...] |
|embeddings |  [0.732479989528656, 0.3734700083732605, 0.0188...] |
|are |     [-0.5153300166130066, 0.8318600058555603, 0.22...] |
|generated |   [-0.35510000586509705, 0.6115900278091431, 0.4...] |
|by |  [-0.20874999463558197, -0.11739999800920486, 0...] |
|aggregating |     [-0.5133699774742126, 0.04489300027489662, 0.1...] |
|global |  [0.24281999468803406, 0.6170300245285034, 0.66...] |
|word-word |   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...] |
|co-occurrence  | [0.16384999454021454, -0.3178800046443939, 0.1...]  |
|matrix |  [-0.2663800120353699, 0.4449099898338318, 0.32...] |
|  from |     [0.30730998516082764, 0.24737000465393066, 0.6...] |
|  a |    [-0.2708599865436554, 0.04400600120425224, -0...]  |
|  corpus |   [0.39937999844551086, 0.15894000232219696, -0...]  |

</div></div>

### Multiple Token Embeddings at once 

{:.contant-descr}
[Compare 6 Embeddings at once with NLU and T-SNE example](https://colab.research.google.com/drive/1DBk55f9iERI9BDA4kmZ8yO6J65jGmcEA?usp=sharing)

```python
#This takes around 10GB RAM, watch out!
nlu.load('bert albert electra elmo xlnet use glove').predict('Get all of them at once! Watch your RAM tough!')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|xlnet_embeddings	| use_embeddings | 	elmo_embeddings 	| electra_embeddings | 	glove_embeddings | 	sentence| 	albert_embeddings| 	biobert_embeddings| 	bert_embeddings| 
|-------------------|---------------|------------------------|-----------------|--------------------|----------|--------------------|---------------------|-------------------|
[[-0.003953204490244389, -1.5821468830108643, ...,]|	[-0.019299551844596863, -0.04762779921293259, ...,]|	[[0.04002974182367325, -0.43536433577537537, -...,]|	[[0.19559216499328613, -0.46693214774131775, -...,]|	[[0.1443299949169159, 0.4395099878311157, 0.58...,]|	Get all of them at once, watch your RAM tough!| 	[[-0.4743960201740265, -0.581386387348175, 0.7...,]|	[[-0.00012563914060592651, -1.372296929359436,...,]|	[[-0.7687976360321045, 0.8489367961883545, -0....,]|  

</div></div>

###  Bert Sentence Embeddings

{:.contant-descr}
[BERT Sentence Embeddings example](https://colab.research.google.com/drive/1FmREx0O4BDeogldyN74_7Lur5NeiOVye?usp=sharing)

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
| 	sentence | 	bert_sentence_embeddings| 
|------------|-----------------------------|
|He was  suprised by the diversity of NLU	| [-1.0726687908172607, 0.4481312036514282, -0.0...,] |

</div></div>

### Electra Sentence Embeddings

{:.contant-descr}
[ELECTRA Sentence Embeddings example](https://colab.research.google.com/drive/1VXHH0ltHF_hXdiRqRlrV_lymAO4ws5PO?usp=sharing)

```python
nlu.load('embed_sentence.electra').predict('He was suprised by the diversity of NLU')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|	sentence | 	electra_sentence_embeddings | 
|-----------|--------------------------------|
|He was suprised by the diversity of NLU	| [0.005376118700951338, 0.18036000430583954, -0...,] |

</div></div>

### Sentence Embeddings Use

{:.contant-descr}
[USE Sentence Embeddings example](https://colab.research.google.com/drive/1gZzOMiCovmrp7z8FIidzDTLS0nt8kPJT?usp=sharing)

```python
nlu.load('use').predict('USE is designed to encode whole sentences and documents into vectors that can be used for text classification, semantic similarity, clustering or oder NLP tasks')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|sentence |    use_embeddings |
|---------|--------------------|
|USE  is designed to encode whole sentences and ...]   | [0.03302069380879402, -0.004255455918610096, -...]    |

</div></div>

### Spell Checking

{:.contant-descr}
[Spell checking example](https://colab.research.google.com/drive/1bnRR8FygiiN3zJz3mRdbjPBUvFsx6IVB?usp=sharing)

```python
nlu.load('spell').predict('I liek pentut buttr ant jely')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|  token | checked    |
|--------|--------|
|I|    I |    
|liek| like | 
|peantut|  pentut |   
|buttr|    buttr |   
|and|  and |  
|jelli|    jely |    

</div></div>

### Dependency Parsing Unlabeled

{:.contant-descr}
[Untyped Dependency Parsing example](https://colab.research.google.com/drive/1PC8ga_NFlOcTNeDVJY4x8Pl5oe0jVmue?usp=sharing)


```python
nlu.load('dep.untyped').predict('Untyped Dependencies represent a grammatical tree structure')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|  token |    pos |  dependency |   
|--------|---------|---------------|
|Untyped|  NNP|   ROOT| 
|Dependencies| NNP|   represent| 
|represent|    VBD|   Untyped| 
|a|    DT|    structure| 
|grammatical|  JJ|    structure| 
|tree| NN|    structure| 
|structure|    NN|    represent| 

</div></div>

### Dependency Parsing Labeled

{:.contant-descr}
[Typed Dependency Parsing example](https://colab.research.google.com/drive/1KXUqcF8e-LU9cXnHE8ni8z758LuFPvY7?usp=sharing)

```python
nlu.load('dep').predict('Typed Dependencies represent a grammatical tree structure where every edge has a label')
```
<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token|    pos|   dependency|    labled_dependency | 
|----|-----|--------------|------------------|
|Typed|    NNP |  ROOT|  root|  
|Dependencies| NNP |  represent|     nsubj|     
|represent|    VBD |  Typed|     parataxis|     
|a|    DT |   structure|     nsubj|     
|grammatical|  JJ |   structure|     amod|  
|tree| NN |   structure|     flat|  
|structure|    NN |   represent|     nsubj|     
|where|    WRB |  structure|     mark|  
|every|    DT |   edge|  nsubj|     
|edge| NN |   where|     nsubj|     
|has|  VBZ |  ROOT|  root|  
|a|    DT |   label|     nsubj|     
|label|    NN |   has|   nsubj|     

</div></div>

### Tokenization

{:.contant-descr}
[Tokenization example](https://colab.research.google.com/drive/13BC6k6gLj1w5RZ0SyHjKsT2EOwJwbYwb?usp=sharing)

```python
nlu.load('tokenize').predict('Each word and symbol in a sentence will generate token.')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token         |
|--------------|
|  Each        |
|  word     |
|     and     |
|  symbol     |
|   will |
|     generate      |
|         a  |
|       token  |
|       \.  |

</div></div>

### Stemmer

{:.contant-descr}
[Stemmer example](https://colab.research.google.com/drive/1gKTJJmffR9wz13Ms3pDy64jhUI8ZHZYu?usp=sharing)

```python
nlu.load('stemm').predict('NLU can get you the stem of a word')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token|    stem|  
|----|------|
|NLU | nlu |  
|can | can |  
|get | get |  
|you | you |  
|the | the |  
|stem |    stem | 
|of |  of |   
|a |   a |    
|word |    word | 

</div></div>

### Stopwords Removal

{:.contant-descr}
[Stopwords Removal example](https://colab.research.google.com/drive/1nWob4u93t2EJYupcOIanuPBDfShtYjGT?usp=sharing)

```python
nlu.load('stopwords').predict('I want you to remove stopwords from this sentence please')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token|    cleanTokens|   
|-----|------------|
|I|    remove |   
|want| stopewords |   
|you|  sentence | 
|to|   None | 
|remove|   None | 
|stopwords|    None | 
|from| None | 
|this| None | 
|sentence| None | 
|please|   None | 

</div></div>

### Lemmatization

{:.contant-descr}
[Lemmatization example](https://colab.research.google.com/drive/1cBtx9cVCjavt-Oq5TG1lO-9JfUfqznnK?usp=sharing)

```python
nlu.load('lemma').predict('Lemmatizing generates a less noisy version of the inputted tokens')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|token|    lemma| 
|------|------|
|Lemmatizing|  Lemmatizing|   
|generates|    generate|  
|a|    a| 
|less| less|  
|noisy|    noisy| 
|version|  version|   
|of|   of|    
|the|  the|   
|inputted| input| 
|tokens|   token| 

</div></div>

### Normalizers

{:.contant-descr}
[Normalizing example](https://colab.research.google.com/drive/1kfnnwkiQPQa465Jic6va9QXTRssU4mlX?usp=sharing)

```python
nlu.load('norm').predict('@CKL_IT says that #normalizers are pretty useful to clean #structured_strings in #NLU like tweets')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|normalized     | token   | 
|------------|----------|
|CKLIT|    @CKL_IT    | 
|says| says| 
|that| that| 
|normalizers|  #normalizers| 
|are|  are|  
|pretty|   pretty|   
|useful|   useful|   
|to|   to|   
|clean|    clean|    
|structuredstrings|    #structured_strings|  
|in|   in|   
|NLU|  #NLU| 
|like| like| 
|tweets|   tweets|   

</div></div>

### NGrams

{:.contant-descr}
[NGrams example](https://colab.research.google.com/drive/1pgqoRJ6yGWbTLWdLnRvwG5DLSU3rxuMq?usp=sharing)

```python
nlu.load('ngram').predict('Wht a wondful day!')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|document |    ngrams|    pos|
|---------|-----|---------|
|To be or not to be|    [To, be, or, not, to, be, To be, be or, or not...] |   [TO, VB, CC, RB, TO, VB] |

</div></div>

### Date Matching

{:.contant-descr}
[Date Matching example](https://colab.research.google.com/drive/1JrlfuV2jNGTdOXvaWIoHTSf6BscDMkN7?usp=sharing)

```python
nlu.load('match.datetime').predict('In the years 2000/01/01 to 2010/01/01 a lot of things happened')
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|document |  date|
|---------|--------|
|In the years 2000/01/01 to 2010/01/01 a lot of things happened |  [2000/01/01, 2001/01/01] |

</div></div>

### Entity Chunking    

{:.contant-descr}
Checkout [see here](http://localhost:4000/docs/en/examples#part-of-speech--pos) for all possible POS labels or        
Splits text into rows based on matched grammatical entities.    
[Entity Chunking Example](https://colab.research.google.com/drive/1svpqtC3cY6JnRGeJngIPl2raqxdowpyi?usp=sharing)

```python
# First we load the pipeline
pipe = nlu.load('match.chunks')
# Now we print the info to see at which index which com,ponent is and what parameters we can configure on them 
pipe.generate_class_metadata_table()
# Lets set our Chunker to only match NN
pipe['default_chunker'].setRegexParsers(['<NN>+', '<JJ>+'])
# Now we can predict with the configured pipeline
pipe.predict("Jim and Joe went to the big blue market next to the town hall")
```

```bash
# the outputs of pipe.print_info()
The following parameters are configurable for this NLU pipeline (You can copy paste the examples) :
>>> pipe['document_assembler'] has settable params:
pipe['document_assembler'].setCleanupMode('disabled')         | Info: possible values: disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full | Currently set to : disabled
>>> pipe['sentence_detector'] has settable params:
pipe['sentence_detector'].setCustomBounds([])                 | Info: characters used to explicitly mark sentence bounds | Currently set to : []
pipe['sentence_detector'].setDetectLists(True)                | Info: whether detect lists during sentence detection | Currently set to : True
pipe['sentence_detector'].setExplodeSentences(False)          | Info: whether to explode each sentence into a different row, for better parallelization. Defaults to false. | Currently set to : False
pipe['sentence_detector'].setMaxLength(99999)                 | Info: Set the maximum allowed length for each sentence | Currently set to : 99999
pipe['sentence_detector'].setMinLength(0)                     | Info: Set the minimum allowed length for each sentence. | Currently set to : 0
pipe['sentence_detector'].setUseAbbreviations(True)           | Info: whether to apply abbreviations at sentence detection | Currently set to : True
pipe['sentence_detector'].setUseCustomBoundsOnly(False)       | Info: Only utilize custom bounds in sentence detection | Currently set to : False
>>> pipe['regex_matcher'] has settable params:
pipe['regex_matcher'].setCaseSensitiveExceptions(True)        | Info: Whether to care for case sensitiveness in exceptions | Currently set to : True
pipe['regex_matcher'].setTargetPattern('\S+')                 | Info: pattern to grab from text as token candidates. Defaults \S+ | Currently set to : \S+
pipe['regex_matcher'].setMaxLength(99999)                     | Info: Set the maximum allowed length for each token | Currently set to : 99999
pipe['regex_matcher'].setMinLength(0)                         | Info: Set the minimum allowed length for each token | Currently set to : 0
>>> pipe['sentiment_dl'] has settable params:
>>> pipe['default_chunker'] has settable params:
pipe['default_chunker'].setRegexParsers(['<DT>?<JJ>*<NN>+'])  | Info: an array of grammar based chunk parsers | Currently set to : ['<DT>?<JJ>*<NN>+']```
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|	chunk | 	pos | 
|-------|-----------|
|market	    |[NNP, CC, NNP, VBD, TO, DT, JJ, JJ, NN, JJ, TO...|
|town hall	|[NNP, CC, NNP, VBD, TO, DT, JJ, JJ, NN, JJ, TO...|
|big blue	|[NNP, CC, NNP, VBD, TO, DT, JJ, JJ, NN, JJ, TO...|
|next	    |[NNP, CC, NNP, VBD, TO, DT, JJ, JJ, NN, JJ, TO...|

</div></div>

### Sentence Detection

{:.contant-descr}
[Sentence Detection example](https://colab.research.google.com/drive/1CAXEdRk_q3U5qbMXsxoVyZRwvonKthhF?usp=sharing)

```python 
nlu.load('sentence_detector').predict('NLU can detect things. Like beginning and endings of sentences. It can also do much more!', output_level ='sentence')  
```

<div class="table-wrapper"><div class="table-inner" markdown="1">

{:.table2}
|sentence|  word_embeddings|   pos|   ner|
|--------|---------------------|-------|------|
|NLU can detect things.    |  [[0.4970400035381317, -0.013454999774694443, 0...]|  [NNP, MD, VB, NNS, ., IN, VBG, CC, NNS, IN, NN...  ]|[O, O, O, O, O, B-sent, O, O, O, O, O, O, B-se...] |
|Like beginning and endings of sentences.  |   [[0.4970400035381317, -0.013454999774694443, 0...]|    [NNP, MD, VB, NNS, ., IN, VBG, CC, NNS, IN, NN...]|    [O, O, O, O, O, B-sent, O, O, O, O, O, O, B-se...] |
|It can also do much more! | [[0.4970400035381317, -0.013454999774694443, 0...]|   [NNP, MD, VB, NNS, ., IN, VBG, CC, NNS, IN, NN...]|    [O, O, O, O, O, B-sent, O, O, O, O, O, O, B-se...] |

</div></div>