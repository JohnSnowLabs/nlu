---
layout: docs
seotitle: NLU | John Snow Labs
title: Installation
permalink: /docs/en/install
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

To install the **johnsnowlabs Python library** and all of John Snow Labs open **source libraries**, just run

```shell 
pip install johnsnowlabs
```

To quickly test the installation, you can run in your **Shell**:

```shell
python -c "from johnsnowlabs import nlp;print(nlp.load('emotion').predict('Wow that easy!'))"
```
or in **Python**:
```python
from  johnsnowlabs import nlp
nlp.load('emotion').predict('Wow that easy!')
```

when using **Annotator based pipelines**, use `nlp.start()` to start up your session 
```python
from johnsnowlabs import nlp
nlp.start()
pipe = nlp.Pipeline(stages=
[
    nlp.DocumentAssembler().setInputCol('text').setOutputCol('doc'),
    nlp.Tokenizer().setInputCols('doc').setOutputCol('tok')
])
nlp.to_nlu_pipe(pipe).predict('That was easy')
```


for alternative installation options see [Custom Installation](/docs/en/install_advanced)