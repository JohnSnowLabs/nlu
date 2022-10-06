---
layout: docs
seotitle: NLU | John Snow Labs
title: Utilities for Testing Models & Modelshub Code Snippets
permalink: /docs/en/testing-utils-modelshub
key: docs-install
modify_date: "2020-05-26"
header: true
---

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
test_ipynb('my/markdown/folder')
```

### Test a List of Markdown References
Can be mixed with Urls and paths, will generate a report
```python
from johnsnowlabs.utils.notebooks import test_ipynb
md_to_test = ['legpipe_deid_en.html',
  'path/to/local/markdown_snippet.md',]
test_ipynb(md_to_test)
```




</div>