---
layout: docs
seotitle: NLU | John Snow Labs
title: John Snow labs Usage & Overview
permalink: /docs/en/import-structure
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1">

The John Snow Labs Python library gives you a clean and easy way to structure your Python projects.
The very first line of a project should be:
```python
from johnsnowlabs import *
```
This imports all licensed and open source Python modules installed from other John Snow Labs Products, as well as
many handy utility imports.


The following Functions, Classes and Modules will available in the global namespace

## The **nlp** Module
-------------------
`nlp` module with classes and methods from [Spark NLP](https://nlp.johnsnowlabs.com/docs/en/quickstart)  like `nlp.BertForSequenceClassification`  and `nlp.map_annotations()`
- `nlp.AnnotatorName` via Spark NLP [Annotators](https://nlp.johnsnowlabs.com/docs/en/annotators) and [Transformers](https://nlp.johnsnowlabs.com/docs/en/transformers) i.e. `nlp.BertForSequenceClassification`
- Spark NLP [Helper Functions](https://nlp.johnsnowlabs.com/docs/en/auxiliary) i.e. `nlp.map_annotations()`
- `nlp.F` via `import pyspark.sql.functions as F` under the hood
- `nlp.T` via `import pyspark.sql.types as T` under the hood
- `nlp.SQL` via `import pyspark.sql as SQL` under the hood
- `nlp.ML` via  `from pyspark import ml as ML` under the hood
- To see all the imports see [the source](https://github.com/JohnSnowLabs/johnsnowlabs/blob/main/johnsnowlabs/nlp.py)


## The **jsl** Module

`jsl` module with the following methods
- `jsl.install()` for installing John Snow Labs libraries and managing your licenses, [more info here](https://nlu.johnsnowlabs.com/docs/en/install)
- `jsl.load()` for predicting with any the 10k+ pretrained models in 1 line of code or training new ones, using the [nlu.load() method](https://nlu.johnsnowlabs.com/) under the hood
- `jsl.start()` for starting a Spark Session with access to features, [more info here](https://nlu.johnsnowlabs.com/docs/en/start-a-sparksession)
- `jsl.viz()` for visualizing predictions with any of the 10k+ pretrained models using [nlu.viz()](https://nlu.johnsnowlabs.com/docs/en/viz_examples) under the hood
- `jsl.viz_streamlit()` and other `jsl.viz_streamlit_xyz for using any of the 10k+ pretrained models in 0 lines of code with an [interactive Streamlit GUI and re-usable and stackable Streamlit Components](https://nlu.johnsnowlabs.com/docs/en/streamlit_viz_examples)
- `jsl.to_pretty_df()` for predicting on raw strings getting a nicely structures Pandas DF from a Spark Pipeline using [nlu.to_pretty_df()](https://nlu.johnsnowlabs.com/docs/en/utils_for_spark_nlp) under the hood


## The **viz** Module

`viz` module with classes from [Spark NLP Display](https://nlp.johnsnowlabs.com/docs/en/display)
- `viz.NerVisualizer` for visualizing prediction outputs of Ner based Spark Pipelines
- `viz.DependencyParserVisualizer` for visualizing prediction outputs of DependencyParser based Spark Pipelines
- `viz.RelationExtractionVisualizer` for visualizing prediction outputs of RelationExtraction based Spark Pipelines
- `viz.EntityResolverVisualizer` for visualizing prediction outputs of EntityResolver based Spark Pipelines
- `viz.AssertionVisualizer` for visualizing prediction outputs of Assertion based Spark Pipelines


## The **ocr** Module

`ocr` module with annotator classes and methods from [Spark OCR](https://nlp.johnsnowlabs.com/docs/en/ocr) like `ocr.VisualDocumentClassifier`  and `ocr.helpful_method()
- [Pipeline Components](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components) i.e. `ocr.ImageToPdf`
- [Table Recognizers](https://nlp.johnsnowlabs.com/docs/en/ocr_table_recognition) i.e. `ocr.ImageTableDetector`
- [Visual Document Understanding](https://nlp.johnsnowlabs.com/docs/en/ocr_visual_document_understanding) i.e. `ocr.VisualDocumentClassifier`
- [Object detectors](https://nlp.johnsnowlabs.com/docs/en/ocr_object_detection) i.e. `ocr.ImageHandwrittenDetector`
- [Enums, Structures and helpers](https://nlp.johnsnowlabs.com/docs/en/ocr_structures) i.e. `ocr.Color`
- To see all the imports see [the source](https://github.com/JohnSnowLabs/johnsnowlabs/blob/main/johnsnowlabs/ocr.py)

## The **medical** Module


`medical` module with annotator classes and methods from [Spark NLP for Medicine](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators)  like `medical.RelationExtractionDL`  and `medical.profile()`
- [Medical Annotators](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators) , i.e. `medical.DeIdentification`
- [Training Methods](https://nlp.johnsnowlabs.com/docs/en/licensed_training)  i.e. `medical.AnnotationToolJsonReader`
- [Evaluation Methods](https://nlp.johnsnowlabs.com/docs/en/evaluation), i.e. `medical.NerDLEvaluation`
- **NOTE:** Any class which has `Medical` in its name is available, but the `Medical` prefix has been omitted. I.e. `medical.NerModel` maps to `sparknlp_jsl.annotator.MedicalNerModel`
  - This is achieved via `from sparknlp_jsl.annotator import MedicalNerModel as NerModel` under the hood.
- To see all the imports see [the source](https://github.com/JohnSnowLabs/johnsnowlabs/blob/main/johnsnowlabs/medical.py)

## The **legal** Module

`legal` module with annotator classes and methods from [Spark NLP for Legal](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators)  like `legal.RelationExtractionDL`  and `legal.profile()`
- [Legal Annotators](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators) , i.e. `legal.DeIdentification`
- [Training Methods](https://nlp.johnsnowlabs.com/docs/en/licensed_training)  i.e. `legal.AnnotationToolJsonReader`
- [Evaluation Methods](https://nlp.johnsnowlabs.com/docs/en/evaluation), i.e. `legal.NerDLEvaluation`
- **NOTE:** Any class which has `Legal` in its name is available, but the `Legal` prefix has been omitted. I.e. `legal.NerModel` maps to `sparknlp_jsl.annotator.LegalNerModel`
  - This is achieved via `from sparknlp_jsl.annotator import LegalNerModel as NerModel` under the hood.
- To see all the imports see [the source](https://github.com/JohnSnowLabs/johnsnowlabs/blob/main/johnsnowlabs/legal.py)


## The **finance** Module


`finance` module with annotator classes and methods from [Spark NLP for Finance](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators)  like `finance.RelationExtractionDL`  and `finance.profile()`
- [Finance Annotators](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators) , i.e. `finance.DeIdentification`
- [Training Methods](https://nlp.johnsnowlabs.com/docs/en/licensed_training)  i.e. `finance.AnnotationToolJsonReader`
- [Evaluation Methods](https://nlp.johnsnowlabs.com/docs/en/evaluation), i.e. `finance.NerDLEvaluation`
- **NOTE:** Any class which has `Finance` in its name is available, but the `Finance` prefix has been omitted. I.e. `finance.NerModel` maps to `sparknlp_jsl.annotator.FinanceNerModel`
  - This is achieved via `from sparknlp_jsl.annotator import FinanceNerModel as NerModel` under the hood.
- To see all the imports see [the source](https://github.com/JohnSnowLabs/johnsnowlabs/blob/main/johnsnowlabs/finance.py)
- 
</div>