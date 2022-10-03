---
layout: docs
seotitle: NLU | John Snow Labs
title: Starting a Spark Session
permalink: /docs/en/start-a-sparksession
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1">

To use most features you must start a Spark Session with `jsl.start()`first.
This will launch a Java [Virtual Machine(JVM)](https://en.wikipedia.org/wiki/Java_virtual_machine) process on your machine
which has all of John Snow Labs and Sparks [Scala/Java Libraries(JARs)](https://de.wikipedia.org/wiki/Java_Archive) loaded into memory. 

The `jsl.start()` method loads all jars for which credentials are provided if they are missing.
If you have installed via `jsl.install()` you can most likely skip the rest of this page, since your secrets have been`~/.jsl_home` and re-use will be re-used.
If you disabled license caching while installing, installed manually or if you want to  tweak settings about your spark session continue reading this section further.


## Authorization Flow Parameters 
Most of the authorization Flows and Parameters of `jsl.install()` 
Review detailed [docs here](TODO)  

| Parameter           | Description                                                                                                                                                    | Example                                          | Default |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|---------|
| `None`              | Load license automatically from via one of the **Auto-Detection Mechanisms*                                                                                    | `jsl.start()`                                    | `False` |
| `browser_login`     | Browser based authorization, Button to click on Notebooks and Browser Pop-Up otherwise.                                                                        | `jsl.start(browser_login=True)`                  | `False` |
| `access_token`      | Vist [my.johnsnowlabs.com](https://my.johnsnowlabs.com/) to extract a token which you can provide to enable license access. See [Accesses Token Example](TODO) | `jsl.start(access_token='myToken')`              | `None`  |
| `secrets_file`      | Define JSON license file with keys  defined by [License Variable Overview](TODO) and provide file path                                                         | `jsl.start(secrets_file='path/to/license.json')` | `None`  |
| `store_in_jsl_home` | Disable caching of new licenses to `~./jsl_home`                                                                                                               | `jsl.start(store_in_jsl_home=False)`             | `True`  |
| `license_number`    | Specify which license to use, if you have access to multiple locally cached or are loading one from  [my.jsl.com](https://my.johnsnowlabs.com/)                | `jsl.start(license_number=5)`                    | `0`     |


### Manually specify License Parameters 
These can be omitted according to the [License Variable Overview](TODO)

| Parameter        | Description                            |
|------------------|----------------------------------------|
| `aws_access_key` | Corresponds to `AWS_ACCESS_KEY_ID`     |
| `aws_key_id`     | Corresponds to `AWS_SECRET_ACCESS_KEY` |
| `hc_secret`      | Corresponds to `HC_SECRET`             |
| `ocr_secret`     | Corresponds to `OCR_SECRET`            |
| `hc_license`     | Corresponds to `HC_LICENSE`            |
| `ocr_license`    | Corresponds to `OCR_LICENSE`           |
| `fin_license`    | Corresponds to `JSL_LEGAL_LICENSE`     |
| `leg_license`    | Corresponds to `JSL_FINANCE_LICENSE`   |

## Sparksession Parameters
These parameters configure how your spark Session is started up.
See [Spark Configuration](https://spark.apache.org/docs/latest/configuration.html) for a comprehensive overview of all spark settings 

| Parameter            | Default    | Description                                                                                                                                                        | Example                                                                    |
|----------------------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `spark_conf`         | `None`     | Dictionary Key/Value pairs of [Spark Configurations](https://spark.apache.org/docs/latest/configuration.html) for the Spark Session                                | `jsl.start(spark_conf={'spark.executor.memory':'6g')`                      |
| `master_url`         | `local[*]` | URL to Spark Cluster master                                                                                                                                        | `jsl.start(master_url=spark://my.master)`                                  |
| `jar_paths`          | `None`     | List of paths to jars which should be loaded into the Spark Session                                                                                                | `jsl.start(jar_paths=['my/jar_folder/jar1.zip','my/jar_folder/jar2.zip' )` |
| `exclude_nlp`        | `False`    | Whether to include Spark NLP jar in Session or not. This will always load the jar if available, unless set to `True`.                                              | `jsl.start(exclude_nlp=True)`                                              |
| `exclude_healthcare` | `False`    | Whether to include licensed NLP Jar for Legal,Finance or Healthcare. This will always load the jar if available using your provided license, unless set to `True`. | `jsl.start(exclude_healthcare=True)`                                       |
| `exclude_ocr`        | `False`    | Whether to include licensed OCR Jar for Legal,Finance or Healthcare. This will always load the jar if available using your provided license, unless set to `True`. | `jsl.start(exclude_ocr=True)`                                              |
| `hardware_target`    | `cpu`      | Specify for which hardware Jar should be optimized. Valid values are `gpu`,`cpu`,`m1`,`aarch`                                                                      | `jsl.start(hardware_target=True)`                                          |
| `model_cache_folder` | `None`     | Specify where models should be downloaded to when using `model.pretrained()`                                                                                       | `jsl.start(model_cache_folder=True)`                                       |





</div>