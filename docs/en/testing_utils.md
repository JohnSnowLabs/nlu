---
layout: docs
seotitle: NLU | John Snow Labs
title: Release Testing Utilities
permalink: /docs/en/testing-utils
key: docs-install
modify_date: "2020-05-26"
header: true
---

## Utilities for Testing Models & Modelshub Code Snippets

<div class="main-docs" markdown="1">

You can use the John Snow Labs library to automatically test 10000+ models and 100+ Notebooks in 1 line of code within
a small machine like a **single Google Colab Instance** and generate very handy error reports of potentially broken Models, Notebooks or Models hub Markdown Snippets.

You can test the following things with the `test_markdown()` function :

- A `local` Models Hub Markdown snippet via path.
- a `remote` Models Hub Markdown snippet via URL.
- a `local` folder of Models Hub Markdown files. Generates report
- a `list`  of local paths or urls to .md files. Generates a report

Test-Report Pandas Dataframe has the columns:

| Report Column | Description                                                   | 
|---------------|---------------------------------------------------------------|
| `test_script` | is the generated script for testing                           |
| `stderr`      | Error logs of process ran. Print this to easily read          |
| `stdout`      | Standard Print logs of process ran. Print this to easily read |
| `success`     | True if script ran successfully from top to bottom            |
| `notebook`    | The Source notebook for testing                               |







### Test a Local Models Hub Markdown Snippet

```python
from johnsnowlabs.utils.modelhub_markdown import test_markdown
test_markdown('path/to/my/file.md')
```

### Test a Remote Models Hub Markdown Snippet

```python
from johnsnowlabs.utils.modelhub_markdown import test_markdown
test_markdown('https://nlp.johnsnowlabs.com/2022/08/31/legpipe_deid_en.html')
```

### Test a Folder with Models Hub Markdown Snippets
This will scan the folder for all files ending with `.md` , test them and generate a report
```python
from johnsnowlabs.utils.modelhub_markdown import test_markdown
test_markdown('my/markdown/folder')
```

### Test a List of Markdown References
Can be mixed with Urls and paths, will generate a report
```python
from johnsnowlabs.utils.notebooks import test_ipynb
md_to_test = ['legpipe_deid_en.html',
  'path/to/local/markdown_snippet.md',]
test_markdown(md_to_test)
```


</div>



<div class="main-docs" markdown="1">

## Utilities for Testing Notebooks

You can use the John Snow Labs library to automatically test 10000+ models and 100+ Notebooks in 1 line of code within
a small machine like a **single Google Colab Instance** and generate very handy error reports of potentially broken Models, Notebooks or Models hub Markdown Snippets.

You can test the following things with the `test_ipynb()` function :


- A `local` .ipynb file
- a `remote` .ipynb URL, point to RAW githubuser content URL of the file when using git.
- a `local` folder of ipynb files, generates report
- a `list` of local paths or urls to .ipynb files. Generates a Report
- The entire [John Snow Labs Workshop Certification Folder](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings) Generates a Report
- A sub-folder of the [John Snow Labs Workshop Certification Folder](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings) , i.e. only OCR or only Legal. Generates a Report




The generated Test-Report Pandas Dataframe has the columns:

| Report Column | Description                                                                                                                         | 
|---------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `test_script` | is the generated script for testing. If you think the notebook should not crash, check the file, there could be a generation error. |
| `stderr`      | Error logs of process ran. Print this to easily read                                                                                |
| `stdout`      | Standard Print logs of process ran. Print this to easily read                                                                       |
| `success`     | True if script ran successfully from top to bottom                                                                                  |
| `notebook`    | The Source notebook for testing                                                                                                     |







### Test a Local Notebook

```python
from johnsnowlabs.utils.notebooks import test_ipynb
test_ipynb('path/to/local/notebook.ipynb')
```

### Test a Remote Notebook

```python
from johnsnowlabs.utils.notebooks import test_ipynb
test_ipynb('https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/5.Spark_OCR.ipynb',)
```

### Test a Folder with Notebooks
This will scan the folder for all files ending with `.ipynb` , test them and generate a report
```python
from johnsnowlabs.utils.notebooks import test_ipynb
test_ipynb('my/notebook/folder')
```



### Test a List of Notebook References
Can be mixed with Urls and paths, will generate a report
```python
from johnsnowlabs.utils.notebooks import test_ipynb
nb_to_test = [
  'https://raw.githubusercontent.com/JohnSnowLabs/spark-nlp-workshop/master/tutorials/Certification_Trainings/Healthcare/5.Spark_OCR.ipynb',
  'path/to/local/notebook.ipynb',]
test_ipynb(nb_to_test)
```


### Run All Certification Notebooks
Will generate a report
```python
from johnsnowlabs.utils.notebooks import test_ipynb
test_result = test_ipynb('WORKSHOP')
```




### Run Finance Certification Notebooks only
Will generate a report
```python
from johnsnowlabs.utils.notebooks import test_ipynb
test_result = test_ipynb('WORKSHOP-FIN')
```

### Run Legal notebooks only
Will generate a report
```python
from johnsnowlabs.utils.notebooks import test_ipynb
test_result = test_ipynb('WORKSHOP-LEG')
```

### Run Medical notebooks only
Will generate a report
```python
from johnsnowlabs.utils.notebooks import test_ipynb
test_result = test_ipynb('WORKSHOP-MED')
```

### Run Open Source notebooks only
Will generate a report
```python
from johnsnowlabs.utils.notebooks import test_ipynb
test_result = test_ipynb('WORKSHOP-OS')
```



</div>