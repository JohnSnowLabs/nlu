---
layout: docs
seotitle: NLU | John Snow Labs
title: John Snow Labs Configurations
permalink: /docs/en/john-snow-labs-home
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1">



## Installed Library Version Settings
Each version of the John Snow Labs library comes with a **hardcoded set of versions** for very of product of the John Snow Labs company.       
It will not accept **library secrets** which correspond to **versions do not match the settings**.
This essentially prevents you from installing **outdated** or **new but not deeply tested** libraries, or from shooting yourself in the foot you might say.


You can work around this protection mechanism, by configuring which version of licensed and open source libraries should be installed and accepted by updating the 
`jsl.settings` module. 

```python
#Example of all library versions to configure
from johnsnowlabs import *
jsl.settings.raw_version_jsl_lib='1.2.3'
jsl.settings.raw_version_nlp='1.2.3.rc1'
jsl.settings.raw_version_medical='1.2.3rc2'
jsl.settings.raw_version_secret_medical='1.2.3.a3'
jsl.settings.raw_version_secret_ocr='1.2.3.abc'
jsl.settings.raw_version_ocr='1.2.3.abc'
jsl.settings.raw_version_nlu='1.2.3.abc'
jsl.settings.raw_version_pyspark='1.2.3.abc'
jsl.settings.raw_version_nlp_display='1.2.3.abc'
```


## John Snow Labs Home Cache Folder
The John Snow Labs library maintains a home folder in `~/.johnsnowlabs` which contains all your Licenses, Jars for Java and Wheels for Python to install and run any feature.
Additionally, each directory has an `info.json` file, telling you more about Spark compatibility, Hardware Targets and versions of the files.


```shell
~/.johnsnowlabs/
   ├─ licenses/
   │  ├─ info.json
   │  ├─ license1.json
   │  ├─ license2.json
   ├─ java_installs/
   │  ├─ info.json
   │  ├─ app1.jar
   │  ├─ app2.jar
   ├─ py_installs/
   │  ├─ info.json
   │  ├─ app1.tar.gz
   │  ├─ app2.tar.gz
   ├─ info.json

```
</div>