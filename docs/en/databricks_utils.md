---
layout: docs
seotitle: NLU | John Snow Labs
title: Utilities for Databricks
permalink: /docs/en/databricks-utils
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1">

## Submit a Task with jsl.run_in_databricks
Easily run Python code in a Databricks cluster, using the John Snow Labs library. 
The fastest way to test this out, is to create a cluster with `jsl.install()` and then use `jsl.run_in_databricks` to start a task.
```python

# Execute a Raw Python string as script on Databricks
from johnsnowlabs import *
script = """
import nlu
print(nlu.load('sentiment').predict('That was easy!'))"""

cluster_id = jsl.install(json_license_path=my_license, databricks_host=my_host,databricks_token=my_token)
jsl.run_in_databricks(script,
                      databricks_cluster_id=cluster_id,
                      databricks_host=my_host,
                      databricks_token=my_token,
                      run_name='Python Code String Example')

```
This will start a **Job Run** which you can view in the **Workflows tab**
![databricks_cluster_submit_raw.png](/assets/images/jsl_lib/databricks_utils/submit_raw_str.png)

And after a while you can see the results 
![databricks_cluster_submit_raw.png](/assets/images/jsl_lib/databricks_utils/submit_raw_str_result.png)


### Run a Python Function in Databricks

Define a function, which will be written to a local file, copied to HDFS and executed by the Databricks cluster.

```python
def my_function():
    import nlu
    medical_text = """A 28-year-old female with a history of gestational 
    diabetes presented with a one-week history of polyuria ,
     polydipsia , poor appetite , and vomiting ."""
    df = nlu.load('en.med_ner.diseases').predict(medical_text)
    for c in df.columns: print(df[c])

# my_function will run on databricks
jsl.run_in_databricks(my_function,
                      databricks_cluster_id=cluster_id,
                      databricks_host=my_host,
                      databricks_token=my_token,
                      run_name='Function test')

```
This example will print all columns of the resulting dataframe which contains emdical NER predictions.
![databricks_cluster_submit_raw.png](/assets/images/jsl_lib/databricks_utils/submit_func.png)


### Run a Raw Python Code String in Databricks
Provide a string which must be valid Python Syntax.    
It will be written to string, copied to HDFS and executed by the Databricks Cluster.

```python
script = """
import nlu
print(nlu.load('sentiment').predict('That was easy!'))"""

jsl.run_in_databricks(script,
                      databricks_cluster_id=cluster_id,
                      databricks_host=my_host,
                      databricks_token=my_token,
                      run_name='Python Code String Example')

```


### Run a Python Script in Databricks
Provide the path to a script on your machine. It will be copied to the Databricks HDFS and executed as task.
```python
jsl.run_in_databricks('path/to/my/script.py',
                      databricks_cluster_id=cluster_id,
                      databricks_host=my_host,
                      databricks_token=my_token,
                      run_name='Script test ')
```

### Run a Python Module in Databricks

Provide a module accessible to the john snow labs library.
It's content's will be written to a local file, copied to HDFS and executed by the databricks cluster.

```python
import johnsnowlabs.auto_install.health_checks.nlp_test as nlp_test
jsl.run_in_databricks(nlp_test,
                      databricks_cluster_id=cluster_id,
                      databricks_host=my_host,
                      databricks_token=my_token,
                      run_name='nlp_test')
```


</div>