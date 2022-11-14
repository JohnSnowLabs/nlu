---
layout: docs
header: true
seotitle: Medical NLP | John Snow Labs
title: Quick Start
permalink: /docs/en/quickstart_medical
key: docs-concepts
modify_date: "2020-05-08"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
You can create [Medical Annotator & PretrainedPipeline based pipelines](https://nlp.johnsnowlabs.com/docs/en/concepts) using all the classes 
attached to the `Medical` & `nlp` module after [installing the licensed libraries](/docs/en/install_licensed_quick).

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
nlp.start()
medical_text = ''' The patient is a 5-month-old infant who presented initially on Monday with
a cold, cough, and runny nose for 2 days'''
nlp.load('med_ner.jsl.wip.clinical').predict(medical_text)

```

| entity      | entity_class | entity_confidence |
|:------------|:-------------|------------------:|
| 5-month-old | Age          |            0.9982 |
| infant      | Age          |            0.9999 |
| Monday      | RelativeDate |            0.9983 |
| cold        | Symptom      |            0.7517 |
| cough       | Symptom      |            0.9969 |
| runny nose  | Symptom      |            0.7796 |
| for 2 days  | Duration     |            0.5479 |



</div><div class="h3-box" markdown="1">


`nlp.load()` defines **additional components types** usable in 1-liners which are only avaiable if a medical license is provided.     

</div><div class="h3-box" markdown="1">

Licensed Component Types :

{:.table-model-big}
| Component type                                                     | nlp.load() base                                    |
|--------------------------------------------------------------------|----------------------------------------------------|
| Medical Named Entity Recognition(NER)                              | `nlp.load('med.ner')`                              |
| Entity Resolution                                                  | `nlp.load('resolve')`                              |
| Entity Assertion                                                   | `nlp.load('assert')`                               |
| Entity Relation Classification                                     | `nlp.load('relation')`                             |
| Entity De-Identification                                           | `nlp.load('de_identify')`                          |
| Map Entities into Terminologies                                    | `nlp.load('map_entity')`                           |
| Translate Entities from One Terminologies into Another Terminology | `nlp.load('<Terminilogy>_to_<other_terminology>')` |
| Drug Normalizers                                                   | `nlp.load('norm_drugs')`                           |
| Rule based NER with Context Matcher                                | `nlp.load('match.context')`                        |


## Annotator & PretrainedPipeline based pipelines
You can create [Annotator & PretrainedPipeline based pipelines](https://nlp.johnsnowlabs.com/docs/en/concepts) using all the classes 
attached to the `nlp` module.


`nlp.PretrainedPipeline('pipe_name')` gives access to [Pretrained Pipelines](https://nlp.johnsnowlabs.com/models?type=pipeline)

```python
from johnsnowlabs import nlp
nlp.start()

deid_pipeline = nlp.PretrainedPipeline("clinical_deidentification", "en", "clinical/models")
sample = """Name : Hendrickson, Ora, Record date: 2093-01-13, # 719435.
Dr. John Green, ID: 1231511863, IP 203.120.223.13.
He is a 60-year-old male was admitted to the Day Hospital for cystectomy on 01/13/93.
Patient's VIN : 1HGBH41JXMN109286, SSN #333-44-6666, Driver's license no:A334455B.
Phone (302) 786-5227, 0295 Keats Street, San Francisco, E-MAIL: smith@gmail.com."""

result = deid_pipeline.annotate(sample)
print("\n".join(result['masked']))
print("\n".join(result['masked_with_chars']))
print("\n".join(result['masked_fixed_length_chars']))
print("\n".join(result['obfuscated']))

```
OUTPUT:
```shell
Masked with entity labels
------------------------------
Name : <PATIENT>, Record date: <DATE>, # <MEDICALRECORD>.
Dr. <DOCTOR>, ID<IDNUM>, IP <IPADDR>.
He is a <AGE> male was admitted to the <HOSPITAL> for cystectomy on <DATE>.
Patient's VIN : <VIN>, SSN <SSN>, Driver's license <DLN>.
Phone <PHONE>, <STREET>, <CITY>, E-MAIL: <EMAIL>.


Masked with chars
------------------------------
Name : [**************], Record date: [********], # [****].
Dr. [********], ID[**********], IP [************].
He is a [*********] male was admitted to the [**********] for cystectomy on [******].
Patient's VIN : [***************], SSN [**********], Driver's license [*********].
Phone [************], [***************], [***********], E-MAIL: [*************].


Masked with fixed length chars
------------------------------
Name : ****, Record date: ****, # ****.
Dr. ****, ID****, IP ****.
He is a **** male was admitted to the **** for cystectomy on ****.
Patient's VIN : ****, SSN ****, Driver's license ****.
Phone ****, ****, ****, E-MAIL: ****.


Obfuscated
------------------------------
Name : Berneta Phenes, Record date: 2093-03-14, # Y5003067.
Dr. Dr Gaston Margo, IDOX:8976967, IP 001.001.001.001.
He is a 91 male was admitted to the MADONNA REHABILITATION HOSPITAL for cystectomy on 07-22-1994.
Patient's VIN : 5eeee44ffff555666, SSN 999-84-3686, Driver's license S99956482.
Phone 74 617 042, 1407 west stassney lane, Edmonton, E-MAIL: Carliss@hotmail.com.

```



### Custom Pipes
Alternatively you can compose [Annotators](https://nlp.johnsnowlabs.com/docs/en/annotators) into a pipeline which offers the highest degree of customization 
```python
from johnsnowlabs import nlp,medical
spark = nlp.start()
documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentenceDetector = nlp.SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

zero_shot_ner = medical.ZeroShotNerModel.pretrained("zero_shot_ner_roberta", "en", "clincial/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("zero_shot_ner")\
    .setEntityDefinitions(
    {
        "NAME": ["What is his name?", "What is my name?", "What is her name?"],
        "CITY": ["Which city?", "Which is the city?"]
    })

ner_converter = medical.NerConverterInternal()\
    .setInputCols(["sentence", "token", "zero_shot_ner"])\
    .setOutputCol("ner_chunk")

pipeline = nlp.Pipeline(stages = [
    documentAssembler,
    sentenceDetector,
    tokenizer,
    zero_shot_ner,
    ner_converter])

zero_shot_ner_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame(["Hellen works in London, Paris and Berlin. My name is Clara, I live in New York and Hellen lives in Paris.",
                              "John is a man who works in London, London and London."], nlp.StringType()).toDF("text")

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