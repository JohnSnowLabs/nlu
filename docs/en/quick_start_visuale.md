---
layout: docs
header: true
seotitle: NLU | John Snow Labs
title: Quick Start
permalink: /docs/en/quickstart_visual
key: docs-concepts
modify_date: "2020-05-08"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


## Load & Predict 1 liner

The `johnsnowlabs` library provides 2 simple methods with which most visual NLP tasks can be solved while achieving state-of-the-art results.   
The **load** and **predict** method.        
When building a `load&predict` based model you will follow these steps:

1. Pick a visual model/pipeline/component you want to create from the [Namespace](/docs/en/namespace)
2. Call the `model = ocr.load('visual_component')` method which will return an auto-completed pipeline
3. Call `model.predict('path/to/image.png')` with a path to a file or an array of paths

These 3 steps can be boiled down to **just 1 line**

```python
from johnsnowlabs import nlp
nlp.load('img2text').predict('path/to/haiku.png')
```


`jsl.load()` defines **6 visual components types** usable in 1-liners

</div><div class="h3-box" markdown="1">
{:.table-model-big}
| 1-liner                                                                              | Transformer Class                                                                             |
|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `nlp.load('img2text').predict('path/to/cat.png')`                                    | [ImageToText](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#imagetotext)       |
| `nlp.load('pdf2text').predict('path/to/taxes.pdf')`                                  | [PdfToText](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#pdftotext)           |
| `nlp.load('doc2text').predict('path/to/my_homework.docx')`                           | [DocToText](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#doctotext)           |
| `nlp.load('pdf2table').predict('path/to/data_tables.pdf')`                           | [PdfToTextTable](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#pdftotexttable) |              
| `nlp.load('ppt2table').predict('path/to/great_presentation_with_tabular_data.pptx')` | [PptToTextTable](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#ppttotexttable) |              
| `nlp.load('doc2table').predict('path/to/tabular_income_data.docx')`                  | [DocToTextTable](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#doctotexttable) |              



## Custom Pipelines
You can create [Visual Annotator & PretrainedPipeline based pipelines](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components) using all the classes 
attached to the `visual` module which gives you the highest degree of freedom

```python
from johnsnowlabs import nlp,visual
spark = nlp.start(visual=True)
# Load a PDF File and convert it into Spark DF format
doc_example = visual.pkg_resources.resource_filename('sparkocr', 'resources/ocr/docs/doc2.docx')
doc_example_df = spark.read.format("binaryFile").load(doc_example).cache()

# Run the visual DocToText Annotator inside a pipe, recognize text and show the result
pipe = nlp.PipelineModel(stages=[visual.DocToText().setInputCol("content").setOutputCol("text")])
result = pipe.transform(doc_example_df)

print(result.take(1)[0].text)
```
output:
![ocrresult.png](/assets/images/jsl_lib/ocr/doc2text.png)



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