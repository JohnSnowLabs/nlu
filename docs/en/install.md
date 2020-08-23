---
layout: article
title: Installation
permalink: /docs/en/install
key: docs-install
modify_date: "2020-05-26"
---

# NLU Installation

```bash
# Install Spark NLU from PyPI
pip install nlu==2.5rc1
```
## Requirements & Setup

Spark NLU is built on top the latest Spark NLP which is built on top of **Apache Spark 2.4.4**.     
In order to use Spark NLU you need the following requirements:

* Java 8
* Apache Spark 2.4.x
* spark-nlp


```bash
# should be Java 8 (Oracle or OpenJDK)
java -version
```

## Verify that NLU is working properly

```python
import nlu
nlu.load('sentiment').predict('Why is NLU is awesome? Its because of the sauce!')
```


##  Supported data types
NLU supports currently the following data formats : 
- Pandas Dataframes  (one column ***must be named text*** and be of object/string type
- 1-D Numpy arrays of Strings
- Strings
- Arrays of Strings
- Spark Dataframes  (one column ***must be named text*** and be of string type

NLU plans to support the following data formats in the future
- Modin


## Troubleshoot

On manjaro and Arch you might encounter an error because of missing libffi.so.6 . 
With *yay libffi6* you can resolve this error.



## Join our Slack channel

Join our channel, to ask for help and share your feedback. Developers and users can help each other getting started here.

[Spark NLP Slack](https://join.slack.com/t/spark-nlp/shared_invite/enQtNjA4MTE2MDI1MDkxLWVjNWUzOGNlODg1Y2FkNGEzNDQ1NDJjMjc3Y2FkOGFmN2Q3ODIyZGVhMzU0NGM3NzRjNDkyZjZlZTQ0YzY1N2I){:.button.button--info.button--rounded.button--md}


## Where to go next

If you want to get your hands dirty with some NLU work, check out the [Examples page](examples)
Detailed information about NLU APIs, concepts, componnents and more can be found on the following pages : 

- [The NLU load function](load_api)
- [The NLU predict function](predict_api)
- [The NLU components namespace](model_namespace)
- [NLU starter Collab notebook](https://colab.research.google.com/drive/1hJ6BiYXxfeDfDjsZu0ZI2TnOa9nrIxfI?usp=sharing)