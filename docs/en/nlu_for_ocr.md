---
layout: docs
header: true
title: OCR models overview
key: docs-nlu-for-ocr
permalink: /docs/en/nlu_for_ocr
modify_date: "2019-05-16"
---
<div class="main-docs" markdown="1">


This page gives you an overview of every OCR model in NLU which are provided by [Spark
OCR](https://nlp.johnsnowlabs.com/docs/en/ocr).

Additionally this [this tutorial NLU OCR notebook](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/colab/ocr/ocr_for_img_pdf_docx_files.ipynb) gives you an overview of all OCR features
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/colab/ocr/ocr_for_img_pdf_docx_files.ipynb)


<div class="h3-box" markdown="1">


| NLU Spell | Transformer Class |
|----------------------|-----------------------------------------------------------------------------------------|
| nlu.load(`img2text`) | [ImageToText](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#imagetotext) |
| nlu.load(`pdf2text`) | [PdfToText](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#pdftotext) |
| nlu.load(`doc2text`) | [DocToText](https://nlp.johnsnowlabs.com/docs/en/ocr_pipeline_components#doctotext) |


When your nlu pipeline contains a `ocr spell` the predict method will accept the following inputs :

- a `string` pointing to a folder or to a file
- a `list`, `numpy array` or `Pandas Series` containing paths pointing to folders or files
- a `Pandas Dataframe` or `Spark Dataframe` containing a column named `path` which has one path entry per row
pointing to folders or files

For every path in the input passed to the `predict()` method, nlu will distinguish between two cases:
1. If the path points to a `file`, nlu will apply OCR transformers to it, if the file type is applicable with
the currently loaded OCR pipeline.
2. If the path points to a `folder`, nlu will recuirsively search for files in the folder and subfolders which
have file types wich are applicable with the loaded OCR pipeline.

NLU checks the file endings to determine wether the OCR models can be applied or not, i.e. `.pdf`, `.img` etc..
If your files lack these endings, NLU will not process them.


## Image to Text
Sample image:
![MarineGEO circle logo](/assets/images/ocr/nlu_ocr/haiku.png )

```python
nlu.load('img2text').predict('path/to/haiku.png')
```

**Output of IMG OCR:**

| text                           |
|:-------------------------------|
| “The Old Pond” by Matsuo Basho |
| An old silent pond             |
| A frog jumps into the pond—    |
| Splash! Silence again.         |

## PDF to Text
Sample PDF:
![MarineGEO circle logo](/assets/images/ocr/nlu_ocr/haiku_pdf.png )

```python
nlu.load('pdf2text').predict('path/to/haiku.pdf')
```
**Output of PDF OCR:**

| text                                |
|:------------------------------------|
| “Lighting One Candle” by Yosa Buson |
| The light of a candle               |
| Is transferred to another candle—   |
| Spring twilight                     |

## DOCX to text
Sample DOCX:
![MarineGEO circle logo](/assets/images/ocr/nlu_ocr/haiku_docx.png )

```python
nlu.load('doc2text').predict('path/to/haiku.docx')
```

**Output of DOCX OCR:**

| text                                        |
|:--------------------------------------------|
| “In a Station of the Metro” by Ezra Pound   |
| The apparition of these faces in the crowd; |
| Petals on a wet, black bough.               |

## Combine OCR and NLP models
## DOCX to text

Sample image containing named entities [from U.S. Presidents Wikipedia](https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States):

![MarineGEO circle logo](/assets/images/ocr/nlu_ocr/presidents.png )
```python
nlu.load('img2text ner').predict('path/to/presidents.png')
```
**Output of image OCR and NER NLP :**

| entities_ner                                 | entities_ner_class   |   entities_ner_confidence |
|:---------------------------------------------|:---------------------|--------------------------:|
| Four                                         | CARDINAL             |                  0.9986   |
|  Abraham Lincoln                             | PERSON               |                  0.705514 |
| John F. Kennedy),                            | PERSON               |                  0.966533 |
| one                                          | CARDINAL             |                  0.9457   |
|  Richard Nixon,                              | PERSON               |                  0.71895  |
| John Tyler                                   | PERSON               |                  0.9929   |
| first                                        | ORDINAL              |                  0.9811   |
| The Twenty-fifth Amendment                   | LAW                  |                  0.548033 |
| Constitution                                 | LAW                  |                  0.9762   |
| Tyler's                                      | CARDINAL             |                  0.5329   |
| 1967                                         | DATE                 |                  0.8926   |
| Richard Nixon                                | PERSON               |                  0.99515  |
| first                                        | ORDINAL              |                  0.9588   |
| Gerald Ford                                  | PERSON               |                  0.996    |
| Spiro Agnew’s                                | PERSON               |                  0.99165  |
| 1973                                         | DATE                 |                  0.9438   |
| Ford                                         | PERSON               |                  0.8337   |
| second                                       | ORDINAL              |                  0.9119   |
| Nelson Rockefeller                           | PERSON               |                  0.98615  |
| 1967                                         | DATE                 |                  0.589    |

## Authorize NLU for OCR
You need a set of **credentials** to access the licensed OCR features.
[You can grab one here](https://www.johnsnowlabs.com/spark-nlp-try-free/)


### Authorize anywhere via providing via JSON file
If you provide a JSON file with credentials, nlu will check whether there are only OCR or also Healthcare secrets.
If both are contained in the JSON file, nlu will give you access to healthcare and OCR features, if only one of them
is present you will be accordingly only authorized for one set of the features.
You can specify the location of your `secrets.json` like this :
```python
path = '/path/to/secrets.json'
nlu.auth(path).load('licensed_model').predict(data)
```

### Authorize via providing String parameters
You can manually enter your secrets and authorize nlu for OCR and Healthcare features
```python
import nlu
AWS_ACCESS_KEY_ID = 'YOUR_SECRETS'
AWS_SECRET_ACCESS_KEY = 'cgsHeZR+YOUR_SECRETS'
OCR_SECRET = 'YOUR_SECRETS'
JSL_SECRET = 'YOUR_SECRETS'
OCR_LICENSE = "YOUR_SECRETS"
SPARK_NLP_LICENSE = 'YOUR_SECRETS'
# this will automatically install the OCR library and NLP Healthcare library when credentials are provided
nlu.auth(SPARK_NLP_LICENSE,AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,JSL_SECRET, OCR_LICENSE, OCR_SECRET)
```



</div>
</div>