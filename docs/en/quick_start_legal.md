---
layout: docs
header: true
seotitle: Legal NLP | John Snow Labs
title: Quick Start
permalink: /docs/en/quickstart_legal
key: docs-concepts
modify_date: "2020-05-08"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">
Installing 

## Annotator & PretrainedPipeline based pipelines
You can create [Legal Annotator & PretrainedPipeline based pipelines](https://nlp.johnsnowlabs.com/docs/en/concepts) using all the classes 
attached to the `legal` & `nlp` module after [installing the licensed libraries](/docs/en/install_licensed_quick).


`nlp.PretrainedPipeline('pipe_name')` gives access to [Pretrained Pipelines](https://nlp.johnsnowlabs.com/models?type=pipeline)

```python
from johnsnowlabs import nlp
nlp.start()
deid_pipeline = nlp.PretrainedPipeline("legpipe_deid", "en", "legal/models")

sample_2 = """Pizza Fusion Holdings, Inc. Franchise Agreement This Franchise Agreement (the "Agreement") is entered into as of the Agreement Date shown on the cover page between Pizza Fusion Holding, Inc., a Florida corporation, and the individual or legal entity identified on the cover page.

Source: PF HOSPITALITY GROUP INC., 9/23/2015


1. RIGHTS GRANTED 1.1. Grant of Franchise. 1.1.1 We grant you the right, and you accept the obligation, to use the Proprietary Marks and the System to operate one Restaurant (the "Franchised Business") at the Premises, in accordance with the terms of this Agreement. 

Source: PF HOSPITALITY GROUP INC., 9/23/2015


1.3. Our Limitations and Our Reserved Rights. The rights granted to you under this Agreement are not exclusive.sed Business.

Source: PF HOSPITALITY GROUP INC., 9/23/2015

"""

result = deid_pipeline.annotate(sample_2)
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
<PARTY>. <DOC> This <DOC> (the <ALIAS>) is entered into as of the Agreement Date shown on the cover page between <PARTY> a Florida corporation, and the individual or legal entity identified on the cover page.
Source: <PARTY>., <EFFDATE>


1.
<PARTY> 1.1.
<PARTY>.
1.1.1 We grant you the right, and you accept the obligation, to use the <PARTY> and the System to operate one Restaurant (the <ALIAS>) at the Premises, in accordance with the terms of this Agreement.
Source: <PARTY>., <EFFDATE>


1.3.
Our <PARTY> and <PARTY>.
The rights granted to you under this Agreement are not exclusive.sed Business.
Source: <PARTY>., <EFFDATE>

Masked with chars
------------------------------
[************************]. [*****************] This [*****************] (the [*********]) is entered into as of the Agreement Date shown on the cover page between [*************************] a Florida corporation, and the individual or legal entity identified on the cover page.
Source: [**********************]., [*******]


1.
[************] 1.1.
[****************].
1.1.1 We grant you the right, and you accept the obligation, to use the [***************] and the System to operate one Restaurant (the [*******************]) at the Premises, in accordance with the terms of this Agreement.
Source: [**********************]., [*******]


1.3.
Our [*********] and [*****************].
The rights granted to you under this Agreement are not exclusive.sed Business.
Source: [**********************]., [*******]

Masked with fixed length chars
------------------------------
****. **** This **** (the ****) is entered into as of the Agreement Date shown on the cover page between **** a Florida corporation, and the individual or legal entity identified on the cover page.
Source: ****., ****


1.
**** 1.1.
****.
1.1.1 We grant you the right, and you accept the obligation, to use the **** and the System to operate one Restaurant (the ****) at the Premises, in accordance with the terms of this Agreement.
Source: ****., ****


1.3.
Our **** and ****.
The rights granted to you under this Agreement are not exclusive.sed Business.
Source: ****., ****

Obfuscated
------------------------------
SESA CO.. Estate Document This Estate Document (the (the "Contract")) is entered into as of the Agreement Date shown on the cover page between Clarus llc. a Florida corporation, and the individual or legal entity identified on the cover page.
Source: SESA CO.., 11/7/2016


1.
SESA CO. 1.1.
Clarus llc..
1.1.1 We grant you the right, and you accept the obligation, to use the John Snow Labs Inc and the System to operate one Restaurant (the (the" Agreement")) at the Premises, in accordance with the terms of this Agreement.
Source: SESA CO.., 11/7/2016


1.3.
Our MGT Trust Company, LLC. and John Snow Labs Inc.
The rights granted to you under this Agreement are not exclusive.sed Business.
Source: SESA CO.., 11/7/2016
```



### Custom Pipes
Alternatively you can compose [Legal Annotators](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators) & [Open Source Annotators](https://nlp.johnsnowlabs.com/docs/en/annotators) into a pipeline which offers the highest degree of customization.

```python
from johnsnowlabs import nlp,legal
spark = nlp.start() 

documentAssembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sparktokenizer = nlp.Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")

zero_shot_ner = legal.ZeroShotNerModel.pretrained("legner_roberta_zeroshot", "en", "legal/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("zero_shot_ner")\
    .setEntityDefinitions(
    {
        "DATE": ['When was the company acquisition?', 'When was the company purchase agreement?', "When was the agreement?"],
        "ORG": ["Which company?"],
        "STATE": ["Which state?"],
        "AGREEMENT": ["What kind of agreement?"],
        "LICENSE": ["What kind of license?"],
        "LICENSE_RECIPIENT": ["To whom the license is granted?"]
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
               "This INTELLECTUAL PROPERTY AGREEMENT, dated as of December 31, 2018 (the 'Effective Date') is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ('Seller') and AFI Licensing LLC, a Delaware company('Licensing')"
               "The Company hereby grants to Seller a perpetual, non- exclusive, royalty-free license"]

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

res = p_model.transform(spark.createDataFrame(sample_text, StringType()).toDF("text"))

res.select(nlp.F.explode(nlp.F.arrays_zip(res.ner_chunk.result, res.ner_chunk.begin, res.ner_chunk.end, res.ner_chunk.metadata)).alias("cols"))\
    .select(nlp.F.expr("cols['0']").alias("chunk"),
            nlp.F.expr("cols['3']['entity']").alias("ner_label"))\
    .filter("ner_label!='O'")\
    .show(truncate=False)\



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