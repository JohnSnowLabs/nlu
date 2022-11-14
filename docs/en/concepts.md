---
layout: docs
header: true
seotitle: NLU | John Snow Labs
title: Quick Start
permalink: /docs/en/concepts
key: docs-concepts
modify_date: "2020-05-08"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

## Load & Predict 1 liner

The `johnsnowlabs` library provides 2 simple methods with which most NLP tasks can be solved while achieving state-of-the-art
results.   
The **load** and **predict** method.

when building a `load&predict` based model you will follow these steps:

1. Pick a model/pipeline/component you want to create from the [Namespace](/docs/en/namespace)
2. Call the `model = nlp.load(component)` method which will return an auto-completed pipeline 
3. Call `model.predict('that was easy')` on some String input

These 3 steps can be boiled down to **just 1 line**

```python
from johnsnowlabs import nlp
nlp.load('sentiment').predict('How does this witchcraft work?')
```

</div><div class="h3-box" markdown="1">

`jsl.load()` defines **18 components types** usable in 1-liners, some can be prefixed with `.train` for [training models](/docs/en/training)


Any of the actions for the component types can be passed as a string to nlp.load() and will return you the default model
for that component type for the English language.
You can further specify your model selection by placing a '.' behind your component selection.        
After the '.' you can specify the model you want via specifying a dataset or model version.   
See [the Models Hub](https://nlp.johnsnowlabs.com/models),  [the Components Namespace](https://nlp.johnsnowlabs.com/docs/en/namespace)
and [The load function](https://nlp.johnsnowlabs.com/docs/en/load_api) for more infos.

</div><div class="h3-box" markdown="1">
{:.table-model-big}
| Component type                | nlp.load() base   |
|-------------------------------|-------------------------------|
| Named Entity Recognition(NER) | `nlp.load('ner')`               |
| Part of Speech (POS)          | `nlp.load('pos')`               |
| Classifiers                   | `nlp.load('classify')`          |
| Word embeddings               | `nlp.load('embed')`             |
| Sentence embeddings           | `nlp.load('embed_sentence')`    |
| Chunk embeddings              | `nlp.load('embed_chunk')`       |
| Labeled dependency parsers    | `nlp.load('dep')`               |
| Unlabeled dependency parsers  | `nlp.load('dep.untyped')`       |
| Legitimatizes                 | `nlp.load('lemma')`             |
| Matchers                      | `nlp.load('match')`             |
| Normalizers                   | `nlp.load('norm')`              |
| Sentence detectors            | `nlp.load('sentence_detector')` |
| Chunkers                      | `nlp.load('chunk')`             |
| Spell checkers                | `nlp.load('spell')`             |
| Stemmers                      | `nlp.load('stem')`              |
| Stopwords cleaners            | `nlp.load('stopwords')`         |
| Cleaner                       | `nlp.load('clean')`             |
| N-Grams                       | `nlp.load('ngram')`             |
| Tokenizers                    | `nlp.load('tokenize')`          |

## Annotator & PretrainedPipeline based pipelines
You can create [Annotator & PretrainedPipeline based pipelines](https://nlp.johnsnowlabs.com/docs/en/concepts) using all the classes 
attached to the `nlp` module.


`nlp.PretrainedPipeline('pipe_name')` gives access to [Pretrained Pipelines](https://nlp.johnsnowlabs.com/models?type=pipeline)

```python
from johnsnowlabs import nlp
from pprint import pprint

nlp.start()
explain_document_pipeline = nlp.PretrainedPipeline("explain_document_ml")
annotations = explain_document_pipeline.annotate("We are very happy about SparkNLP")
pprint(annotations)

OUTPUT:
{
  'stem': ['we', 'ar', 'veri', 'happi', 'about', 'sparknlp'],
  'checked': ['We', 'are', 'very', 'happy', 'about', 'SparkNLP'],
  'lemma': ['We', 'be', 'very', 'happy', 'about', 'SparkNLP'],
  'document': ['We are very happy about SparkNLP'],
  'pos': ['PRP', 'VBP', 'RB', 'JJ', 'IN', 'NNP'],
  'token': ['We', 'are', 'very', 'happy', 'about', 'SparkNLP'],
  'sentence': ['We are very happy about SparkNLP']
}

```


### Custom Pipes
Alternatively you can compose [Annotators](https://nlp.johnsnowlabs.com/docs/en/annotators) into a pipeline which offers the highest degree of customization 
```python
from johnsnowlabs import nlp
spark = nlp.start(nlp=False)
pipe = nlp.Pipeline(stages=
[
    nlp.DocumentAssembler().setInputCol('text').setOutputCol('doc'),
    nlp.Tokenizer().setInputCols('doc').setOutputCol('tok')
])
spark_df = spark.createDataFrame([['Hello NLP World']]).toDF("text")
pipe.fit(spark_df).transform(spark_df).show()
```



[//]: # (</div><div class="h3-box" markdown="1">)



[//]: # ()
[//]: # ()
[//]: # (## Specify language for an action)

[//]: # ()
[//]: # ()
[//]: # (### Print all supported languages)

[//]: # ()
[//]: # ()
[//]: # (Any of these are partial NLU references which can be prefixed to a request to specify a language)

[//]: # ()
[//]: # ()
[//]: # (```python)

[//]: # ()
[//]: # (nlp.languages&#40;&#41;)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (</div><div class="h3-box" markdown="1">)

[//]: # ()
[//]: # ()
[//]: # (### Print every component for one specific language)

[//]: # ()
[//]: # ()
[//]: # (These are complete NLU references and can be passed to the nlp.load&#40;&#41; method right away)

[//]: # ()
[//]: # ()
[//]: # (```python)

[//]: # ()
[//]: # (# Print every German NLU component)

[//]: # ()
[//]: # (nlp.print_components&#40;lang='de'&#41;)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (</div><div class="h3-box" markdown="1">)

[//]: # ()
[//]: # ()
[//]: # (### Print every model for an action)

[//]: # ()
[//]: # ()
[//]: # (These are complete NLU references and can be passed to the nlp.load&#40;&#41; method right away)

[//]: # ()
[//]: # ()
[//]: # (```python)

[//]: # ()
[//]: # (# Print every lemmatizer for every language)

[//]: # ()
[//]: # (nlp.print_components&#40;action='lemma'&#41;)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (</div><div class="h3-box" markdown="1">)

[//]: # ()
[//]: # ()
[//]: # (### Print every model kind for an action and a language)

[//]: # ()
[//]: # ()
[//]: # (These are complete NLU references and can be passed to the nlp.load&#40;&#41; method right away)

[//]: # ()
[//]: # ()
[//]: # (```python)

[//]: # ()
[//]: # (# Print all english classifiers)

[//]: # ()
[//]: # (nlp.print_components&#40;lang='en', action='classify'&#41;)

[//]: # ()
[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (</div><div class="h3-box" markdown="1">)

[//]: # ()
[//]: # ()
[//]: # (### Print the entire NLU spellbook offering)

[//]: # ()
[//]: # ()
[//]: # (These are complete NLU references and can be passed to the nlp.load&#40;&#41; method right away)

[//]: # ()
[//]: # ()
[//]: # (```python)

[//]: # ()
[//]: # (nlp.print_components&#40;&#41;)

[//]: # ()
[//]: # (```)

</div></div>