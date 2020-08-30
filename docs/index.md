---
layout: landing
title: 'NLU: State of the Art  <br /> Text Mining in Python'
excerpt:  <br> The Simplicity of Python, the Power of Spark NLP
permalink: /
header: true
article_header:
 actions:
   - text: Getting Started
     type: error
     url: /docs/en/install   
   - text: '<i class="fab fa-github"></i> GitHub'
     type: outline-theme-dark
     url: https://github.com/JohnSnowLabs/nlu 
   - text: '<i class="fab fa-slack-hash"></i> Slack'
     type: outline-theme-dark
     url: https://app.slack.com/client/T9BRVC9AT/C0196BQCDPY   

 height: 50vh
 theme: dark
 background_color: "#0296D8"

data:
 sections:
   - title:
     children:
       - title: Powerful One-Liners
         excerpt: Hundreds of NLP models in tens of languages are at your fingertips with just one line of code
       - title: Elegant Python
         excerpt: Directly read and write pandas dataframes for frictionless integration with other libraries and existing ML pipelines  
       - title: 100% Open Source
         excerpt: Including pre-trained models & pipelines
   - title: '<h2> Quick and Easy </h2>'
     install: yes
     excerpt: NLU is available on <a href="https://pypi.org/project/nlu" target="_blank">PyPI</a>, <a href="https://anaconda.org/JohnSnowLabs/nlu" target="_blank">Conda</a>
     background_color: "#ecf0f1"
     actions:
       - text: Install NLU
         url: /docs/en/install
  
  
   - title: Benchmark
     excerpt: NLU is based on the award winning Spark NLP which best performing in peer-reviewed results
     benchmark: yes
     features: false
     theme: dark
     background_color: "#123"

    
---




## Tokenization
```python
nlu.load('tokenize').predict('Each word and symbol in a sentence will generate token.')
```

{:.steelBlueCols}
|token         |id|
|--------------|----|
|  Each        |1|
|  word     |1|
|     and     |1|
|  symbol     |1|
|   will |1|
|     generate      |1|
|         a  |1|
|       token  |1|
|       \.  |1|

## Stemmer
```python
nlu.load('stemm').predict('NLU can get you the stem of a word')
```

{:.steelBlueCols}
|token|    stem|  id|
|----|------|-----|
|NLU | nlu |  1 |
|can | can |  1 |
|get | get |  1 |
|you | you |  1 |
|the | the |  1 |
|stem |    stem | 1 |
|of |  of |   1 |
|a |   a |    1 |
|word |    word | 1 |


## Stopwords Removal
```python
nlu.load('stopwords').predict('I want you to remove stopwords from this sentence please')
```


{:.steelBlueCols}
|token|    cleanTokens|   id|
|-----|------------|-------|
|I|    remove |   1 |
|want| stopewords |   1 |
|you|  sentence | 1 |
|to|   None | 1 |
|remove|   None | 1 |
|stopwords|    None | 1 |
|from| None | 1 |
|this| None | 1 |
|sentence| None | 1 |
|please|   None | 1 |


## Part of speech  (POS)
```python
nlu.load('pos').predict('Part of speech assigns each token in a sentence a grammatical label')
```

{:.steelBlueCols}

|token |pos    | id|
|------|-----|-----|
|Part|         NN|     1|
|of|           IN|     1|
|speech|       NN|     1|
|assigns|      NNS| 1|
|each|         DT|     1|
|token|            NN|     1|
|in|           IN|     1|
|a |           DT|     1|
|sentence|     NN|     1|
|a |           DT|     1|
|grammatical|  JJ|     1|
|label          |NN| 1|

## Multi Class Emotion detection
```python
nlu.load('emotion').predict('I love NLU!')
```

{:.steelBlueCols}


|sentence_embeddings|  category_confidence|   sentence|  category|  id|
|--------------------|---------------------|------------|------------|-----|
|[0.027570432052016258, -0.052647676318883896, ...]    |0.976017  |I love NLU!   |joy   |1|

## Sentiment
```python
nlu.load('sentiment').predict("I hate this guy Sami")
```

{:.steelBlueCols}
|sentiment_confidence  |SENTENCE  |sentiment |id    |checked |
|-----------|----------------------|------------|---|---------|
|0.5778 |  I hate this guy Sami   | negative |   1 |    [I, hate, this, guy, Sami] |

## Spell Checking
```python
nlu.load('spell').predict('I liek pentut butr and jelli')
```

{:.steelBlueCols}
|  token | checked    |id|
|--------|--------|-----|
|I|    I |    1|
|liek| like | 1|
|peantut|  peanut |   1|
|buttr|    butter |   1|
|and|  and |  1|
|jelli|    jelly |    1|

## Lemmatization
```python
nlu.load('lemma').predict('Lemmatizing generates a less noisy version of the inputted tokens')
```


{:.steelBlueCols}
|token|    lemma| id|
|------|------|-----|
|Lemmatizing|  Lemmatizing|   1 |
|generates|    generate|  1 |
|a|    a| 1 |
|less| less|  1 |
|noisy|    noisy| 1 |
|version|  version|   1 |
|of|   of|    1 |
|the|  the|   1 |
|inputted| input| 1 |
|tokens|   token| 1 |

## Normalizers
```python
nlu.load('norm').predict('@CKL_IT says that #normalizers are pretty useful to clean #structured_strings in #NLU like tweets')
```

{:.steelBlueCols}
|normalized     | token   | id|
|------------|----------|----|
|CKLIT|    @CKL_IT    |  1|
|says| says|  1|
|that| that|  1|
|normalizers|  #normalizers|  1|
|are|  are|   1|
|pretty|   pretty|    1|
|useful|   useful|    1|
|to|   to|    1|
|clean|    clean|     1|
|structuredstrings|    #structured_strings|   1|
|in|   in|    1|
|NLU|  #NLU|  1|
|like| like|  1|
|tweets|   tweets|    1|


## Language Classification
```python
nlu.load('lang').predict(['NLU is an open-source text processing library for advanced natural language processing for the Python.','NLU est une bibliothèque de traitement de texte open source pour le traitement avancé du langage naturel pour les langages de programmation Python.'])
```

{:.steelBlueCols}

|language_confidence|  document|  language|  id|
|------------------|-----------|-------------|------|
|0.985407  |NLU is an open-source text processing library ...]|   en|    0|
|0.999822  |NLU est une bibliothèque de traitement de text...]|   fr|    1|


## Named Entity Recognition (NER)
```python
nlu.load('ner').predict('Donald Trump had a meeting with Tim Apple in new York !')
```


{:.steelBlueCols}
|token | embeddings_embeddings    |ner   |id    | entities |
|-------|-----------------------|-------|----|---------|
|Donald|   [-0.5496799945831299, -0.488319993019104, 0.59...] |B-PER |1|    [Donald Trump, Tim Apple, York!] |
|Trump|    [-0.15730999410152435, -0.7550299763679504, 0...]      |I-PER |1|    [Donald Trump, Tim Apple, York!] |
|had|  [0.6325600147247314, -0.12717999517917633, -0...]      |O     |1|    [Donald Trump, Tim Apple, York!] |
|a|    [-0.2708599865436554, 0.04400600120425224, -0...]          |O     |1|    [Donald Trump, Tim Apple, York!] |
|meeting|  [-0.4541400074958801, -0.5592300295829773, 0.7...] |O     |1|    [Donald Trump, Tim Apple, York!] |
|with| [-0.4360800087451935, 0.39103999733924866, 0.5...]     |O     |1|    [Donald Trump, Tim Apple, York!] |
|Tim|  [0.07363300025463104, -0.037108998745679855, 0...]     |B-PER |1|    [Donald Trump, Tim Apple, York!] |
|Apple|    [0.2628200054168701, -0.19162000715732574, 0.1...]     |I-PER |1|    [Donald Trump, Tim Apple, York!] |
|in|   [0.08570300042629242, -0.22201000154018402, 0...]      |O     |1|    [Donald Trump, Tim Apple, York!] |
|new|  [-0.04395899921655655, 0.18935999274253845, 0...]      |O     |1|    [Donald Trump, Tim Apple, York!] |
|York| [0.7569400072097778, 0.03385699912905693, 0.37...]     |B-LOC |1|    [Donald Trump, Tim Apple, York!] |
|!|    [0.38471999764442444, 0.49351000785827637, 0.4...]         |I-LOC |1|    [Donald Trump, Tim Apple, York!] |


## NGrams TODO FIX


```python
nlu.load('ngram').predict('What a wondful day!')
```

{:.steelBlueCols}
|document |    id|    ngrams|    pos|
|---------|-----|------|---------|
|To be or not to be| 1 |   [To, be, or, not, to, be, To be, be or, or not...] |   [TO, VB, CC, RB, TO, VB] |



## Regex Matching (Work in Progres)
```python
regex_pipe = nlu.load('match.regex')
regex_pipe.predict('Wht a wondful day!')
```

## Text Matching (Work in Progres)
```python
regex_pipe = nlu.load('match.text')
regex_pipe.pipe['regex_model']setParam()
regex_pipe.predict('Wht a wondful day!')
```


## Date Matching
```python
nlu.load('match.datetime').predict('In the years 2000/01/01 to 2010/01/01 a lot of things happened')
```

{:.steelBlueCols}
|document |    id |   date|
|---------|--------|--------|
|In the years 2000/01/01 to 2010/01/01 a lot of things happened | 1 | [2000/01/01, 2001/01/01] |

## Chunking (Work in Progres)
```python
nlu.load('match.chunks').predict('What a wondful day!') 
```

{:.steelBlueCols}
|chunk|    id|    pos|
|-----|----|----|
|a wondful day | 1 |  [WP, DT, JJ, NN, .] |


## Sentence Detector
```python 
nlu.load('sentence_detector').predict('NLU can detect things. Like beginning and endings of sentences. It can also do much more!', output_level ='sentence')  
```

{:.steelBlueCols}
|sentence|     id|    word_embeddings|   pos|   ner|
|--------|----|---------------------|-------|------|
|NLU can detect things.    | 1    | [[0.4970400035381317, -0.013454999774694443, 0...]|  [NNP, MD, VB, NNS, ., IN, VBG, CC, NNS, IN, NN...  ]|[O, O, O, O, O, B-sent, O, O, O, O, O, O, B-se...] |
|Like beginning and endings of sentences.  |   1 | [[0.4970400035381317, -0.013454999774694443, 0...]|    [NNP, MD, VB, NNS, ., IN, VBG, CC, NNS, IN, NN...]|    [O, O, O, O, O, B-sent, O, O, O, O, O, O, B-se...] |
|It can also do much more! | 1    |[[0.4970400035381317, -0.013454999774694443, 0...]|   [NNP, MD, VB, NNS, ., IN, VBG, CC, NNS, IN, NN...]|    [O, O, O, O, O, B-sent, O, O, O, O, O, O, B-se...] |


## Word Embeddings Bert
```python
nlu.load('bert').predict('NLU offers the latest embeddings in one line ')
```

{:.steelBlueCols}
|token         |bert_embeddings                                         |id|
|--------------|--------------------------------------------------------|----|
|  NLU        |   [0.3253086805343628, -0.574441134929657, -0.08...] |1|
|  offers     |   [-0.6660361886024475, -0.1494743824005127, -0...]  |1|
|  the        |   [-0.6587662696838379, 0.3323703110218048, 0.16...] |1|
|  latest     |   [0.7552685737609863, 0.17207926511764526, 1.35...] |1|
|  embeddings |   [-0.09838500618934631, -1.1448147296905518, -1...] |1|
|  in         |   [-0.4635896384716034, 0.38369956612586975, 0.0...] |1|
|  one        |   [0.26821616291999817, 0.7025910019874573, 0.15...] |1|
|  line       |   [-0.31930840015411377, -0.48271292448043823, 0...] |1|

## Word Embeddings Biobert
```python
nlu.load('biobert').predict('Biobert was pretrained on a medical dataset')
```

{:.steelBlueCols}
|token         |bert_embeddings                                         |id|
|--------------|--------------------------------------------------------|----|
|  NLU        |   [0.3253086805343628, -0.574441134929657, -0.08...] |1|
|  offers     |   [-0.6660361886024475, -0.1494743824005127, -0...]  |1|
|  the        |   [-0.6587662696838379, 0.3323703110218048, 0.16...] |1|
|  latest     |   [0.7552685737609863, 0.17207926511764526, 1.35...] |1|
|  embeddings |   [-0.09838500618934631, -1.1448147296905518, -1...] |1|
|  in         |   [-0.4635896384716034, 0.38369956612586975, 0.0...] |1|
|  one        |   [0.26821616291999817, 0.7025910019874573, 0.15...] |1|
|  line       |   [-0.31930840015411377, -0.48271292448043823, 0...] |1|



## Word Embeddings Albert
```python
nlu.load('albert').predict('Albert uses a collection of many berts to generate embeddings')
```

{:.steelBlueCols}
|token |   albert_embeddings |    id|
|-----|----------------------|---------|
| Albert |         [-0.08257609605789185, -0.8017427325248718, 1...]   |1|
| uses |       [0.8256351947784424, -1.5144840478897095, 0.90...]  |1|
| a |          [-0.22089454531669617, -0.24295514822006226, 3...]  |1|
| collection |     [-0.2136894017457962, -0.8225528597831726, -0...]   |1|
| of |         [1.7623294591903687, -1.113651156425476, 0.800...]  |1|
| many |       [0.6415284872055054, -0.04533941298723221, 1.9...]  |1|
| berts |      [-0.5591965317726135, -1.1773797273635864, -0...]   |1|
| to |             [1.0956681966781616, -1.4180747270584106, -0.2...]  |1|
| generate |   [-0.6759272813796997, -1.3546931743621826, 1.6...]  |1|
| embeddings |     [-0.0035803020000457764, -0.35928264260292053,...]  |1|

## Word Embeddings Elmo
```python
nlu.load('elmo').predict('Elmo was trained on Left to right masked to learn its embeddings')
```

{:.steelBlueCols}
|token|    elmo_embeddings     | id|
|------|-----------------|----|
|Elmo |    [0.6083735227584839, 0.20089012384414673, 0.42...]  |1|
|was | [0.2980785369873047, -0.07382500916719437, -0...]   |1|
|trained | [-0.39923471212387085, 0.17155063152313232, 0...]   |1|
|on |  [0.04337821900844574, 0.1392083466053009, -0.4...]  |1|
|Left |    [0.4468783736228943, -0.623046875, 0.771505534...]  |1|
|to |  [-0.18209676444530487, 0.03812692314386368, 0...]   |1|
|right |   [0.23305709660053253, -0.6459438800811768, 0.5...]  |1|
|masked |  [-0.7243442535400391, 0.10247116535902023, 0.1...]  |1|
|to |  [-0.18209676444530487, 0.03812692314386368, 0...]   |1|
|learn |   [1.2942464351654053, 0.7376189231872559, -0.58...]  |1|
|its | [0.055951207876205444, 0.19218483567237854, -0...]  |1|
|embeddings |  [-1.31377112865448, 0.7727609872817993, 0.6748...]  |1|


## Word Embeddings Xlnet
```python
nlu.load('xlnet').predict('XLNET computes contextualized word representations using combination of Autoregressive Language Model and Permutation Language Model')
```

{:.steelBlueCols}
| token    | xlnet_embeddings |id|
|------|--------------------|---|
|XLNET |   [-0.02719488926231861, -1.7693557739257812, -0...] |1|
|computes |    [-1.8262947797775269, 0.8455266356468201, 0.57...] |1|
|contextualized |  [2.8446314334869385, -0.3564329445362091, -2.1...] |1|
|word |    [-0.6143839359283447, -1.7368144989013672, -0...]  |1|
|representations |     [-0.30445945262908936, -1.2129613161087036, 0...]  |1|
|using |   [0.07423821836709976, -0.02561005763709545, -0...] |1|
|combination |     [-0.5387097597122192, -1.1827564239501953, 0.5...] |1|
|of |  [-1.403516411781311, 0.3108177185058594, -0.32...] |1|
|Autoregressive |  [-1.0869172811508179, 0.7135171890258789, -0.2...] |1|
|Language |    [-0.33215752243995667, -1.4108021259307861, -0...] |1|
|Model |   [-1.6097160577774048, -0.2548254430294037, 0.0...] |1|
|and |     [0.7884324789047241, -1.507911205291748, 0.677...] |1|
|Permutation |     [0.6049966812133789, -0.157279372215271, -0.06...] |1|
|Language |    [-0.33215752243995667, -1.4108021259307861, -0...] |1|
|Model |   [-1.6097160577774048, -0.2548254430294037, 0.0...] |1|


## Word Embeddings Glove
```python
nlu.load('glove').predict('Glove embeddings are generated by aggregating global word-word co-occurrence matrix from a corpus')
```

{:.steelBlueCols}
|  token    |glove_embeddings    |id|
|---------|------------------|-----|  
|Glove |   [0.3677999973297119, 0.37073999643325806, 0.32...] |1|
|embeddings |  [0.732479989528656, 0.3734700083732605, 0.0188...] |1|
|are |     [-0.5153300166130066, 0.8318600058555603, 0.22...] |1|
|generated |   [-0.35510000586509705, 0.6115900278091431, 0.4...] |1|
|by |  [-0.20874999463558197, -0.11739999800920486, 0...] |1|
|aggregating |     [-0.5133699774742126, 0.04489300027489662, 0.1...] |1|
|global |  [0.24281999468803406, 0.6170300245285034, 0.66...] |1|
|word-word |   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...] |1|
|co-occurrence  | [0.16384999454021454, -0.3178800046443939, 0.1...]  |1|
|matrix |  [-0.2663800120353699, 0.4449099898338318, 0.32...] |1|
|  from |     [0.30730998516082764, 0.24737000465393066, 0.6...] |1|
|  a |    [-0.2708599865436554, 0.04400600120425224, -0...]  |1|
|  corpus |   [0.39937999844551086, 0.15894000232219696, -0...]  |1|

## Multiple Token Embeddings at once
```python
#watch out for your RAM, this could kill your machine
nlu.load('bert elmo albert xlnet use glove').predict('Get all of them at once! Watch your RAM tough!')
```


{:.steelBlueCols}
|token |   glove_embeddings   |albert_embeddings |xlnet_embeddings   |  bert_embeddings    |elmo_embeddings   |use_embeddings    | id
|------|-----------------------|-------------------|------------------|-------------------------------------|---|----------------------------------------------------------------------
|Get       | [0.1443299949169159, 0.4395099878311157, 0.583...]       | [-0.41224443912506104, -0.4611411392688751, 1...]        | [-0.003953204490244389, -1.5821468830108643, -...]       | [-0.7420049905776978, -0.8647691011428833, 0.1...]       | [0.04002974182367325, -0.43536433577537537, -0...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|all       | [-0.2182299941778183, 0.6919900178909302, 0.70...]       | [1.1014549732208252, -0.43204769492149353, -0...]        | [0.31148090958595276, -1.0986182689666748, 0.3...]       | [-0.8933112025260925, 0.44822725653648376, -0...]        | [0.17885173857212067, 0.045830272138118744, -0...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|of            | [-0.15289999544620514, -0.24278999865055084, 0...]       | [1.1535910367965698, 0.28440719842910767, 0.60...]       | [-1.403516411781311, 0.3108177185058594, -0.32...]       | [-0.5550722479820251, 0.2702311873435974, 0.04...]       | [0.24783466756343842, -0.248960942029953, 0.02...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|them          | [-0.10130999982357025, 0.10941000282764435, 0...]    | [0.5475010871887207, 0.8660883903503418, 2.817...]   | [-0.7559828758239746, -0.4712887704372406, -1...]    | [-0.2922026813030243, -0.1301671266555786, -0...]    | [-0.24157099425792694, -0.8055092692375183, -0...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|at            | [0.17659999430179596, 0.0938510000705719, 0.24...]       | [-0.5005946159362793, -0.4600788354873657, 0.5...]       | [0.04092511534690857, -1.0951932668685913, -1...]        | [-0.5613634586334229, -0.00903533399105072, -0...]       | [-0.11999595910310745, 0.012994140386581421, -...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|once      | [-0.23837999999523163, 0.221670001745224, 0.35...]   | [-0.39100387692451477, -0.8297092914581299, 2...]    | [-0.46001458168029785, -1.2062749862670898, 0...]    | [0.2988640069961548, 0.3360409140586853, -0.37...]   | [0.6701997518539429, 1.1368376016616821, 0.244...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|!         | [0.38471999764442444, 0.49351000785827637, 0.4...]       | [0.007945209741592407, -0.27733859419822693, 0...]       | [-1.5816600322723389, -0.992130696773529, -0.1...]       | [0.7550013065338135, -0.5257778167724609, -0.4...]       | [-1.3351283073425293, 0.6296550035476685, -1.4...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|Watch     | [-0.38264000415802, -0.08968199789524078, 0.02...]   | [-0.10218311846256256, -0.4334276020526886, 0...]    | [-1.3921688795089722, 0.6997514963150024, -0.8...]   | [-0.24852752685546875, 1.222611427307129, -0.1...]   | [0.04002974182367325, -0.43536433577537537, -0...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|your      | [-0.5718399882316589, 0.046348001807928085, 0...]    | [-0.4086211323738098, 1.0755341053009033, 1.78...]   | [-0.8588163256645203, -2.3702170848846436, 0.0...]   | [-0.035358428955078125, 0.7711482048034668, 0...]    | [0.17885173857212067, 0.045830272138118744, -0...]   | [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|RAM       | [-1.875599980354309, -0.40814998745918274, -0...]        | [-0.09772858023643494, 0.3632940351963043, -0...]        | [1.1277621984481812, -1.689896583557129, -0.19...]       | [0.4528151750564575, -0.36768051981925964, -0...]        | [0.24783466756343842, -0.248960942029953, 0.02...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|tough     | [-0.5099300146102905, -0.1428000032901764, 0.5...]   | [-0.22261293232440948, 0.21325691044330597, 0...]    | [-1.3547197580337524, 0.43423181772232056, -1...]    | [0.46073707938194275, 0.05694812536239624, 0.5...]   | [-0.24157099425792694, -0.8055092692375183, -0...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |
|!         | [0.38471999764442444, 0.49351000785827637, 0.4...]       | [0.21658605337142944, -0.04937351495027542, 0...]        | [-1.5816600322723389, -0.992130696773529, -0.1...]       | [0.6830563545227051, -0.5751053094863892, -0.6...]       | [-0.11999595910310745, 0.012994140386581421, -...]   |  [[-0.0019260947592556477, 0.009215019643306732...]| 1 |





## Sentence Embeddings Use
```python
nlu.load('use').predict('USE is designed to encode whole sentences and documents into vectors that can be used for text classification, semantic similarity, clustering or oder NLP tasks')
```

{:.steelBlueCols}
|sentence |    use_embeddings |   id |
|---------|--------------------|--------|
|USE  is designed to encode whole sentences and ...]   | [0.03302069380879402, -0.004255455918610096, -...]    | 1 |

## General Sentence Embeddings  (Work in Progress)
Via the sentence Embeddings converter provided by Spark NLP any token Embeddings can be
converted to a sentence or document embedding.      

```python
#Any works!
nlu.load('embeddings_sentence.elmo').predict('Wht a wondful day!')
nlu.load('embeddings_sentence.xlnet').predict('Wht a wondful day!')
nlu.load('embeddings_sentence.bert').predict('Wht a wondful day!')
```

## Chunk Embeddings (Work in Progres)
```python
#Any works!
nlu.load('embeddings_chunk.elmo').predict('Wht a wondful day!')
nlu.load('embeddings_chunk.bert').predict('Wht a wondful day!')
nlu.load('embeddings_chunk.xlnet').predict('Wht a wondful day!')
```


## Dependency Parsing Unlabelled
```python
nlu.load('dep.untyped').predict('Untyped Dependencies represent a grammatical tree structure')
```

{:.steelBlueCols}
|  token |    pos |  dependency |   id|
|--------|---------|---------------|------|
|Untyped|  NNP|   ROOT| 1 |
|Dependencies| NNP|   represent| 1 |
|represent|    VBD|   Untyped| 1 |
|a|    DT|    structure| 1 |
|grammatical|  JJ|    structure| 1 |
|tree| NN|    structure| 1 |
|structure|    NN|    represent| 1 |

## Dependency Parsing Labeled
```python
nlu.load('dep').predict('Typed Dependencies represent a grammatical tree structure where every edge has a label')
```

{:.steelBlueCols}
|token|    pos|   dependency|    labled_dependency | id|
|----|-----|--------------|------------------|----|
|Typed|    NNP |  ROOT|  root|  1 |
|Dependencies| NNP |  represent|     nsubj|     1 |
|represent|    VBD |  Typed|     parataxis|     1 |
|a|    DT |   structure|     nsubj|     1 |
|grammatical|  JJ |   structure|     amod|  1 |
|tree| NN |   structure|     flat|  1 |
|structure|    NN |   represent|     nsubj|     1 |
|where|    WRB |  structure|     mark|  1 |
|every|    DT |   edge|  nsubj|     1 |
|edge| NN |   where|     nsubj|     1 |
|has|  VBZ |  ROOT|  root|  1 |
|a|    DT |   label|     nsubj|     1 |
|label|    NN |   has|   nsubj|     1 |


## Classify Question (6 class)
Classify between 6 different types of questions trainec on Trec6

```python
nlu.load('en.classify.trec6').predict('Where is the next food store?')
```

{:.steelBlueCols}
| sentence_embeddings |    category_ ABBR |   category_sentence  | category_ DESC |     category_ NUM |    category_ ENTY |   category_ LOC |    category_ HUM|     sentence   | category|    id|
|---------------------|--------------------|------------------------|-----------------|---------------|--------------------|------------------|-------------------|--------------|--------|-------|
|[-0.05699703469872475, 0.039651867002248764, -...]|    2.2486939E-13  | 0   | 1.2741682E-9 | 2.3458482E-9 | 1.6385917E-8 | 1.0  | 7.901978E-9  | Where is the next food store?    | LOC  |1 |

## Classify Question (50 class)
Classify between 50 different types of questions trained on Trec50

```python
nlu.load('en.classify.trec50').predict('How expensive is the Watch?')
```

{:.steelBlueCols}
| sentence_embeddings  | category_ ENTY_letter    | category_ DESC_reason    | category_ ENTY_word  | category_ LOC_country    | category_ ENTY_other | category_ ENTY_instru    | category_ HUM_desc   | category_ LOC_mount  | category_ NUM_dist   | category_ ENTY_symbol |  category_ DESC_manner  | category_ ABBR_abb   | category_ NUM_count  | category_ DESC_def   | category_ ENTY_color |   category_sentence  | category_ ENTY_event | category_ ENTY_veh |     category_ NUM_ord |     category_ NUM_other|  category_ NUM_code | category_ DESC_desc|     category_ HUM_title    | category_ NUM_weight | category_ ENTY_substance | category_ HUM_gr | category_ ENTY_body|     category_ HUM_ind  | category_ LOC_other  | category_ NUM_speed  | category_ NUM_volsize    | category_ ENTY_religion  | category_ ENTY_dismed    | category_ ABBR_exp   | category_ ENTY_product   | category_ ENTY_currency  | category_ ENTY_sport | category_ NUM_perc   | category_ ENTY_cremat    | category_ ENTY_plant | category_ NUM_money   | category_ NUM_temp |    category_ ENTY_animal  | category_ NUM_period | category_ ENTY_lang  | category_ NUM_date   | category_ ENTY_techmeth  | category_ ENTY_termeq    | category_ LOC_state  | category_ LOC_city   | category_ ENTY_food  | sentence | category |id|
|------------------------|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|---------------------------|----------------------|-------------------------|----------------------|----------------------|--------------------------|----------------------|---------------------------|----------------------|----------------------|-------------------------|----------------------|------------------|--------------------------|----------------------|--------------------|----------------------|-----------------------|---------------------------|-------------------|---------------------|----------------------|----------------------|-----------------------|-----------------------|---------------------------|----------------------|-------------------------|---------------------------|--------------------------|-----------------------|-----------------------|----------------------|----------------------|----------------------|----------------------|-------------------------------|-----------------------|----------------------|-----------------------|----------------------------|----------------------|------------------------|----------------------|------------------------|-----------------------|---|
|[0.051809534430503845, 0.03128402680158615, -0...]|   8.459624E-23 | 3.4999828E-14  | 3.5003475E-23     | 5.841585E-20 |  4.958728E-9     | 1.2962921E-22   | 3.265098E-23 | 2.7989845E-22    | 1.1879483E-22    | 3.3417754E-23    | 1.1275173E-10    | 2.0653923E-23    | 0.91943645   | 1.1442368E-20    | 6.967735E-21 0  | 6.008502E-24 | 4.201739E-23  |  1.9875016E-25  | 7.991396E-25  | 5.3842046E-22   | 3.6065938E-8 | 9.143437E-24 | 5.219503E-23 | 4.7433556E-23    | 4.337756E-19 | 2.7589757E-23    | 5.1413796E-18    | 7.611624E-14 | 4.302599E-21 |4.125377E-23  |5.2841376E-21 |1.519717E-21  | 7.546624E-23 | 8.906359E-22 | 2.617869E-25 | 1.8343639E-24    | 3.1659472E-22  |     2.537563E-21   | 4.2162198E-24    | 2.3034208E-20 |  1.740357E-22   | 4.4444944E-23|   7.8172316E-20  | 8.119662E-24 | 0.0805635    | 4.5887844E-21    | 3.4650898E-22    | 2.3069434E-23    | 4.631075E-22|    1.0575064E-22  |  1.2962221E-22  |     How expensive is the Watch? |  NUM_count  | 1 |

## Fakenews Classifier

```python
nlu.load('en.classify.fakenews').predict('Unicorns have been sighted on Mars!')
```

{:.steelBlueCols}
|sentence_embeddings|  category_confidence|   sentence|  category|  id|
|------------------|-----------------------|------------|-----------|------|
|[-0.01756167598068714, 0.015006818808615208, -...]    | 1.000000 | Unicorns have been sighted on Mars!  |FAKE  |1|


## Cyberbullying Classifier (sexism and Racism Classifier)

```python
nlu.load('en.classify.cyberbullying').predict('Women belong in the kitchen.') # sorry we really don't mean it
```

{:.steelBlueCols}
| sentence_embeddings |    category_sentence |    category_sexism |  category_neutral |     category_racism |  sentence |     category |     id |
|---------------------|-----------------------|---------------------|--------------------|------------------|------------|-----------|-----|
|[-0.054944973438978195, -0.022223370149731636,...] |  0 |    0.999998 |     1.87933E-6 |   7.737535E-8     | Women belong in the kitchen. |  sexism | 1 |

## Spam Classifier

```python
nlu.load('en.classify.spam').predict('Please sign up for this FREE membership it costs $$NO MONEY$$ just your mobile number!')
```

{:.steelBlueCols}
|sentence_embeddings|  category_confidence|   sentence|  category |     id |
|-------------------|----------------------|------------|-----------|-------|
|[0.008322705514729023, 0.009957313537597656, 0...]    | 1.000000 | Please sign up for this FREE membership it cos...    |spam  |1 |

## Sarcasm Classifier

```python
nlu.load('en.classify.sarcasm').predict('gotta love the teachers who give exams on the day after halloween')
```


{:.steelBlueCols}
| sentence_embeddings |    category_confidence  |     sentence |     category  |    id|
|---------------------|--------------------------|-----------|-----------|---------|
|[-0.03146284446120262, 0.04071342945098877, 0....] | 0.999985 | gotta love the teachers who give exams on the...    | sarcasm  | 1 |


## IMDB Movie Sentiment
```python
nlu.load('en.sentiment.imdb').predict('The Matrix was a pretty good movie')
```

{:.steelBlueCols}
|document |    id |   sentence_embeddings|   sentiment_negative|    sentiment_negative|    sentiment_positive|    sentiment |
|-------|-----|-------------------------|---------------------|----------------------|-------------------------|-------------|
|The Matrix was a pretty good movie    |1 |   [[0.04629608988761902, -0.020867452025413513, ...  ]| [2.7235753918830596e-07]     | [2.7235753918830596e-07] |  [0.9999997615814209]|  [positive] |

## Twitter Sentiment
```python
nlu.load('en.sentiment.twitter').predict('@elonmusk Tesla stock price is too high imo')
```

{:.steelBlueCols}
|document |    id |   sentence_embeddings |  sentiment_negative |   sentiment_negative|    sentiment_positive |   sentiment|
|--------|---------|-----------------------|-----------------------|---------------------|-------------------------|-------------|
| @elonmusk Tesla stock price is too high imo  | 1    | [[0.08604438602924347, 0.04703635722398758, -0...]|  [1.0] |    [1.0]  | [1.692714735043349e-36]  | [negative]|



