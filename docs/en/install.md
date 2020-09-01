---
layout: article
title: Installation
permalink: /docs/en/install
key: docs-install
modify_date: "2020-05-26"
---



# 1. Get Prerequisites

You only need to configure Java 8 on your machine and are good to go!

- [Setup Java 8 on Windows](https://access.redhat.com/documentation/en-us/openjdk/8/html/openjdk_8_for_windows_getting_started_guide/getting_started_with_openjdk_for_windows)
- [Setup Java 8 on Linux](https://openjdk.java.net/install/)
- [Setup Java 8 on Mac](https://docs.oracle.com/javase/8/docs/technotes/guides/install/mac_jdk.html)

## Setup Java in Google Collab or Kaggle
If you work in a Kaggle or Collab Notebook you can simply configure Java by running the following code in a cell

```bash
import os
! apt-get update -qq > /dev/null   
# Install java
! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
```



## Verify Java 8 works
```bash
# should be Java 8 (Oracle or OpenJDK)
java -version
```


# 2. Install NLU package

```bash
# Install Spark NLU from PyPI
pip install nlu
```



# 3. Verify that NLU is working properly
```python
import nlu
nlu.load('sentiment').predict('Why is NLU is awesome? It's because of the sauce!')
```


##  Supported data types
NLU supports currently the following data formats :
- Pandas Dataframes  (one column ***must be named text*** and be of object/string type
- Spark Dataframes  (one column ***must be named text*** and be of string type
- Modin with Dask backend
- Modin with Ray backend
- 1-D Numpy arrays of Strings
- Strings
- Arrays of Strings


## Troubleshoot

On Arch based distributions like Manjaro you might encounter an error because of missing libffi.so.6.      
With *yay libffi6* you can resolve this error.



## Join our Slack channel

Join our channel, to ask for help and share your feedback. Developers and users can help each other get started here.

[NLU Slack](https://spark-nlp.slack.com/archives/C0196BQCDPY){:.button.button--info.button--rounded.button--md}


## Where to go next

If you want to get your hands dirty with some NLU work, check out the [Examples page](examples)
Detailed information about NLU APIs, concepts, components and more can be found on the following pages :

- [The NLU load function](load_api)
- [The NLU predict function](predict_api)
- [The NLU components namespace](https://nlu.johnsnowlabs.com/docs/en/namespace)
- [NLU Notebooks](notebooks)
- [NLU starter Collab notebook](https://colab.research.google.com/drive/1hJ6BiYXxfeDfDjsZu0ZI2TnOa9nrIxfI?usp=sharing)


