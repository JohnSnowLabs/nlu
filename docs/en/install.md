---
layout: docs
title: Installation
permalink: /docs/en/install
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1">

<div class="h3-box" markdown="1">

## 0. Super Quickstart on Google Colab or Kaggle

If you work on a fresh Notebook on Kaggle or Google colab, you can just copy paste the following commands into your first cell which 
will automatically setup Java, nlu and import nlu, so you are good to go right away!

```bash
import os
! apt-get update -qq > /dev/null   
# Install java
! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
! pip install nlu
import nlu
```
You can test it out right away with :
```python
nlu.load('emotion').predict('wow that was easy')
```

</div><div class="h3-box" markdown="1">

## 1. Get Prerequisites (Java 8)

You only need to configure Java 8 on your machine and are good to go! 
Unless you are on Windows, which requires 1 additional step.

- [Setup Java 8 on Windows](https://access.redhat.com/documentation/en-us/openjdk/8/html/openjdk_8_for_windows_getting_started_guide/getting_started_with_openjdk_for_windows)
- [Setup Java 8 on Linux](https://openjdk.java.net/install/)
- [Setup Java 8 on Mac](https://docs.oracle.com/javase/8/docs/technotes/guides/install/mac_jdk.html)

</div><div class="h3-box" markdown="1">

### Setup Java in Google Collab or Kaggle
If you work in a Kaggle or Collab Notebook you can simply configure Java by running the following code in a cell

```bash
import os
! apt-get update -qq > /dev/null   
# Install java
! apt-get install -y openjdk-8-jdk-headless -qq > /dev/null
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
```

</div><div class="h3-box" markdown="1">

## Verify Java 8 works
```bash
# should be Java 8 (Oracle or OpenJDK)
java -version
```

</div><div class="h3-box" markdown="1">

## 1.1 Get Windows Specific Prerequisites (winutils.exe)
This is only required for Windows usesr.
1. Download [winutils.exe](https://github.com/steveloughran/winutils/blob/master/hadoop-2.7.1/bin/winutils.exe)
2. Create folder C:\winutils\bin
3. Copy winutils.exe inside C:\winutils\bin
4. Set environment variable HADOOP_HOME to C:\winutils

</div><div class="h3-box" markdown="1">

## 2. Install NLU package

```bash
# Install Spark NLU from PyPI
pip install nlu
```

</div><div class="h3-box" markdown="1">

## 3. Verify that NLU is working properly
Launch a Python shell an run a simple script.         
**On Windows you need to launch your shell as admim**

```python
import nlu
nlu.load('sentiment').predict('Why is NLU so awesome? Because of the sauce!')
```

</div><div class="h3-box" markdown="1">

## Supported data types
NLU supports currently the following data formats :
- Pandas Dataframes 
- Spark Dataframes 
- Modin with Dask backend
- Modin with Ray backend
- 1-D Numpy arrays of Strings
- Strings
- Arrays of Strings

</div><div class="block-wrapper"><div class="block-box" markdown="1">

### Troubleshoot

On Arch based distributions like Manjaro you might encounter an error because of missing libffi.so.6.      
With *yay libffi6* you can resolve this error.

</div><div class="block-box" markdown="1">

### Join our Slack channel

Join our channel, to ask for help and share your feedback. Developers and users can help each other get started here.

[NLU Slack](https://spark-nlp.slack.com/archives/C0196BQCDPY){:.button.button--info.button--rounded.button--md}

</div></div><div class="h3-box" markdown="1">

### Where to go next

If you want to get your hands dirty with some NLU work, check out the [Examples page](examples)
Detailed information about NLU APIs, concepts, components and more can be found on the following pages :

{:.list4}
- [The NLU load function](load_api)
- [The NLU predict function](predict_api)
- [The NLU components namespace](https://nlu.johnsnowlabs.com/docs/en/namespace)
- [NLU Notebooks](notebooks)
- [NLU starter Collab notebook](https://colab.research.google.com/drive/1hJ6BiYXxfeDfDjsZu0ZI2TnOa9nrIxfI?usp=sharing)

</div></div>