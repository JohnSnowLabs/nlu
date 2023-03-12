---
layout: docs
header: true
seotitle: Finance NLP | John Snow Labs
title: Quick Start
permalink: /docs/en/quickstart_finance
key: docs-concepts
modify_date: "2020-05-08"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
Installing 

## Annotator & PretrainedPipeline based pipelines
You can create [Finance Annotator & PretrainedPipeline based pipelines](https://nlp.johnsnowlabs.com/docs/en/concepts) using all the classes
attached to the `finance` & `nlp` module after [installing the licensed libraries](/docs/en/install_licensed_quick).


`nlp.PretrainedPipeline('pipe_name')` gives access to [Pretrained Pipelines](https://nlp.johnsnowlabs.com/models?type=pipeline)

```python
from johnsnowlabs import nlp
from sparknlp.pretrained import PretrainedPipeline
nlp.start()

deid_pipeline = nlp.PretrainedPipeline("finpipe_deid", "en", "finance/models")

sample = """CARGILL, INCORPORATED

By:     Pirkko Suominen



Name: Pirkko Suominen Title: Director, Bio Technology Development,  Date:   10/19/2011

BIOAMBER, SAS

By:     Jean-François Huc



Name: Jean-François Huc  Title: President Date:   October 15, 2011

email : jeanfran@gmail.com
phone : 1808733909 

"""

result = deid_pipeline.annotate(sample)
print("\nMasked with entity labels")
print("-"*30)
print("\n".join(result['deidentified']))
print("\nMasked with chars")
print("-"*30)
print("\n".join(result['masked_with_chars']))
print("\nMasked with fixed length chars")
print("-"*30)
print("\n".join(result['masked_fixed_length_chars']))
print("\nObfuscated")
print("-"*30)
print("\n".join(result['obfuscated']))
```
Output:

```shell
Masked with entity labels
------------------------------
<PARTY>, <PARTY>
By:     <SIGNING_PERSON>
Name: <PARTY>: <SIGNING_TITLE>,  Date:   <EFFDATE>
<PARTY>, <PARTY>
By:     <SIGNING_PERSON>
Name: <PARTY>: <SIGNING_TITLE>Date:   <EFFDATE>

email : <EMAIL>
phone : <PHONE>

Masked with chars
------------------------------
[*****], [**********]
By:     [*************]
Name: [*******************]: [**********************************]  Center,  Date:   [********]
[******], [*]
By:     [***************]
Name: [**********************]: [*******]Date:   [**************]

email : [****************]
phone : [********]

Masked with fixed length chars
------------------------------
****, ****
By:     ****
Name: ****: ****,  Date:   ****
****, ****
By:     ****
Name: ****: ****Date:   ****

email : ****
phone : ****

Obfuscated
------------------------------
MGT Trust Company, LLC., Clarus llc.
By:     Benjamin Dean
Name: John Snow Labs Inc: Sales Manager,  Date:   03/08/2025
Clarus llc., SESA CO.
By:     JAMES TURNER
Name: MGT Trust Company, LLC.: Business ManagerDate:   11/7/2016

email : Tyrus@google.com
phone : 78 834 854
```



### Custom Pipes
Alternatively you can compose [Legal Annotators](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators) & [Open Source Annotators](https://nlp.johnsnowlabs.com/docs/en/annotators) into a pipeline which offers the highest degree of customization.

```python
from johnsnowlabs import nlp,finance
spark = nlp.start()

documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sparktokenizer = nlp.Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")

zero_shot_ner = finance.ZeroShotNerModel.pretrained("finner_roberta_zeroshot", "en", "finance/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("zero_shot_ner")\
    .setEntityDefinitions(
    {
        "DATE": ['When was the company acquisition?', 'When was the company purchase agreement?'],
        "ORG": ["Which company was acquired?"],
        "PRODUCT": ["Which product?"],
        "PROFIT_INCREASE": ["How much has the gross profit increased?"],
        "REVENUES_DECLINED": ["How much has the revenues declined?"],
        "OPERATING_LOSS_2020": ["Which was the operating loss in 2020"],
        "OPERATING_LOSS_2019": ["Which was the operating loss in 2019"]
    })

nerconverter = nlp.NerConverter()\
    .setInputCols(["document", "token", "zero_shot_ner"])\
    .setOutputCol("ner_chunk")

pipeline =  nlp.Pipeline(stages=[
    documentAssembler,
    sparktokenizer,
    zero_shot_ner,
    nerconverter,
]
)

sample_text = ["In March 2012, as part of a longer-term strategy, the Company acquired Vertro, Inc., which owned and operated the ALOT product portfolio.",
               "In February 2017, the Company entered into an asset purchase agreement with NetSeer, Inc.",
               "While our gross profit margin increased to 81.4% in 2020 from 63.1% in 2019, our revenues declined approximately 27% in 2020 as compared to 2019."
               "We reported an operating loss of approximately $8,048,581 million in 2020 as compared to an operating loss of approximately $7,738,193 million in 2019."]

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

res = p_model.transform(spark.createDataFrame(sample_text, nlp.StringType()).toDF("text"))

res.select(nlp.F.explode(nlp.F.arrays_zip(res.ner_chunk.result, res.ner_chunk.begin, res.ner_chunk.end, res.ner_chunk.metadata)).alias("cols"))\
    .select(nlp.F.expr("cols['0']").alias("chunk"),\
            nlp.F.expr("cols['3']['entity']").alias("ner_label"))\
    .filter("ner_label!='O'")\
    .show(truncate=False)

```
Output:

```shell
+---------------------------------------+-----------------+
|chunk                                  |ner_label        |
+---------------------------------------+-----------------+
|March 2012                             |DATE             |
|Vertro, Inc                            |ORG              |
|February 2017                          |DATE             |
|asset purchase agreement               |AGREEMENT        |
|NetSeer                                |ORG              |
|INTELLECTUAL PROPERTY AGREEMENT        |AGREEMENT        |
|December 31, 2018                      |DATE             |
|Armstrong Flooring                     |ORG              |
|Delaware                               |STATE            |
|AFI Licensing LLC                      |ORG              |
|Delaware                               |ORG              |
|Seller                                 |LICENSE_RECIPIENT|
|perpetual, non- exclusive, royalty-free|LICENSE          |
+---------------------------------------+-----------------+
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