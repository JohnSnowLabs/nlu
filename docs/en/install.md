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

To install the johnsnowlabs Python library and all of John Snow Labs open **source libraries**, just run

```shell 
pip install johnsnowlabs
```

This installs [Spark-NLP](https://nlp.johnsnowlabs.com/docs/en/quickstart), [NLU](https://nlu.johnsnowlabs.com/)
, [Spark-NLP-Display](https://nlp.johnsnowlabs.com/docs/en/display)
, [Pyspark](https://spark.apache.org/docs/latest/api/python/) and other open
source [sub-dependencies](https://github.com/JohnSnowLabs/johnsnowlabs/blob/main/setup.py).

To quickly test the installation, you can run in your Shell:

```shell
python -c "from johnsnowlabs import *;print(nlu.load('emotion').predict('Wow that easy!'))"
```

or in Python:

```python
from johnsnowlabs import *
jsl.load('emotion').predict('Wow that easy!')
```

The quickest way to get access to **licensed libraries** like 
[Finance NLP, Legal NLP, Healthcare NLP](https://nlp.johnsnowlabs.com/docs/en/licensed_install) or
[Visual NLP](https://nlp.johnsnowlabs.com/docs/en/ocr) is to run the following in python:

```python
from johnsnowlabs import *
jsl.install()
```

This will display a **Browser Window Pop Up**  or show a **Clickable Button with Pop Up**.        
Click on the **Authorize** button to allow the library to connect to your account on my.JohnSnowLabs.com and access you licenses.          
This will enable the installation and use of all licensed products for which you have a valid license. 

Make sure to **Restart your Notebook** after installation.

Colab Button
![install_button_colab.png](/assets/images/jsl_lib/install/install_button_colab.png)

Where the Pop-Up leads you to:
![install_pop_up.png](/assets/images/jsl_lib/install/install_pop_up.png)

After clicking **Authorize**:
![install_logs_colab.png](/assets/images/jsl_lib/install/install_logs_colab.png)

**Additional Requirements**

- Make sure you have `Java 8` installed, for setup instructions
  see [How to install Java 8 for Windows/Linux/Mac?](https://nlu.johnsnowlabs.com/docs/en/install#get-prerequisites-java-8)
- Windows Users must additionally follow every step precisely defined
  in [How to correctly install Spark NLP for Windows?](https://nlp.johnsnowlabs.com/docs/en/install#windows-support)

</div><div class="h3-box" markdown="1">

{:.h2-select}
## Install Licensed Libraries

The following is a more detailed overview of the alternative installation methods and parameters you can use.
The parameters of `jsl.install()`parameters fall into 3 categories:

- **Authorization Flow Choice & Auth Flow Tweaks**
- **Installation Target** such as `Airgap Offline`, `Databricks`, `new Pytho Venv`
  ,  `Currently running Python Enviroment`,
  or `target Python Environment`
- **Installation process tweaks**

</div><div class="h3-box" markdown="1">

### List all of your accessible Licenses

You can use `jsl.list_remote_licenses()` to list all available licenses in your [my.johnsnowlabs.com/](https://my.johnsnowlabs.com/) account
and `jsl.list_local_licenses()` to list all locally cached licenses. 

</div><div class="h3-box" markdown="1">

### Authorization Flows overview

The `johnsnowlabs` library gives you multiple methods to authorize and provide your license when installing licensed
libraries.
Once access to your license is provided, it is cached locally `~/.johnsnowlabs/licenses` and re-used when
calling `jst.start()` and `jsl.install()`, so you don't need to authorize again.          
Only 1 licenses can be provided and will be cached during authorization flows.           
If you have multiple licenses you can re-run an authorization method and use the `license_number` parameter choose
between licenses you have access to.      
Licenses are locally numbered in order they have been provided, for more info see [License Caching](https://nlu.johnsnowlabs.com/docs/en/install#storage-of-license-data-and-license-search-behaviour).

| Auth Flow Method                                             | Description                                                                                                                                                                                                 | Python `jsl.install()` usage                                                                                                                               |
|--------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Browser Based Login (OAuth) Localhost                        | Browser window will pop up, where you can give access to your license                                                                                                                                       | `jsl.install()`                                                                                                                                            |
| Browser Based Login (OAuth) on Google Colab                  | A button is displayed in your notebook, click it and visit new page to give access to your license                                                                                                         | `jsl.install()`                                                                                                                                            |
| Access Token                                                 | Vist [my.johnsnowlabs.com](https://my.johnsnowlabs.com/) to extract a token which you can provide to enable license access. See [Access Token Example](https://nlu.johnsnowlabs.com/docs/en/install#via-access-token) for more details                             | `jsl.install(access_token=my_token)`                                                                                                                       |
| License JSON file path                                       | Define JSON license file with keys  defined by [License Variable Overview](https://nlu.johnsnowlabs.com/docs/en/install#license-variables-names-for-json-and-os-variables) and provide file path                                                                                                      | `jsl.install(json_license_path=path)`                                                                                                                      |
| **Auto-Detect** License JSON file from `os.getcwd()`         | `os.getcwd()` directory is scanned for a `.json` file containing license keys defined by [License Variable Overview](https://nlu.johnsnowlabs.com/docs/en/install#license-variables-names-for-json-and-os-variables)                                                                                  | `jsl.install()`                                                                                                                                            |
| **Auto-Detect** OS Environment Variables                     | Environment Variables are scanned for license variables defined by [License Variable Overview](https://nlu.johnsnowlabs.com/docs/en/install#license-variables-names-for-json-and-os-variables)                                                                                                        | `jsl.install()`                                                                                                                                            |
| **Auto-Detect** Cached License in `~/.johnsnowlabs/licenses` | If you already have provided a license previously, it is cached in `~/.johnsnowlabs/licenses` and automatically loaded.<br/> Use `license_number` parameter to choose between licenses if you have multiple | `jsl.install()`                                                                                                                                            |
| Manually specify license data                                | Set each license value as python parameter, defined by  [License Variable Overview](https://nlu.johnsnowlabs.com/docs/en/install#license-variables-names-for-json-and-os-variables)                                                                                                                   | `jsl.install(hc_license=hc_license enterprise_nlp_secret=enterprise_nlp_secret ocr_secret=ocr_secret ocr_license=ocr_license aws_access_key=aws_access_key aws_key_id=aws_key_id)` |

</div><div class="h3-box" markdown="1">

### Optional Auth Flow Parameters

Use these parameters to configure **the preferred authorization flow**.

| Parameter                  | description                                                                                                                                                                                                                                                                      |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `browser_login`            | Enable or disable browser based login and pop up if no license is provided or automatically detected. Defaults to `True`.                                                                                                                                                         |
| `force_browser_login`      | If a cached license if found, no browser pop up occurs. Set `True` to force the browser pop up, so that you can download different license, if you have several ones.                                                                                                                |
| `license_number`           | Specify the license number to use with OAuth based approaches or when loading a cached license from jsl home and multiple licenses have been cached. Defaults to `0` which will use your 0th license from my.johnsnowlabs.com.                                                       |
| `store_in_jsl_home`        | By default license data and Jars/Wheels are stored in JSL home directory. <br/> This enables `jsl.start()` and `jsl.install()` to re-use your information and you don't have to specify on every run.<br/> Set to `False` to disable this caching behaviour.<br/> |
| `only_refresh_credentials` | Set to `True` if you don't want to install anything and just need to refresh or index a new license. Defaults to `False`.                                                                                                                                                         |

</div><div class="h3-box" markdown="1">

### Optional Installation Target Parameters

Use these parameters to configure **where** to install the libraries.

| Parameter                                 | description                                                                                                                                                                                                                    |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `python_exec_path`                        | Specify path to a python executable into whose environment the libraries will be installed. Defaults to the current executing Python process, i.e. `sys.executable` and it's pip module is used for setup.                     |
| `venv_creation_path`                      | Specify path to a folder, in which a fresh venv will be created with all libraries. Using this parameter ignores the `python_exec_path` parameter, since the newly created venv's python executable is used for setup.         |
| `offline_zip_dir`                         | Specify path to a folder in which 3 sub-folders are created,  `py_installs`, `java_installs` with corrosponding Wheels/Jars/Tars and  `licenses`. It will additionallly be zipped.                                            |
| `Install to Databricks` with access Token | See [Databricks Documentation](https://docs.databricks.com/dev-tools/api/latest/authentication.html) for  extracting a token which you can provide to databricks access, see [Databricks Install Section](https://nlu.johnsnowlabs.com/docs/en/install#automatic-databricks-installation) for more details. |

</div><div class="h3-box" markdown="1">

### Optional Installation Process Parameters

Use the following parameters to configure **what should** be installed. 

| Parameter              | description                                                                                                                                                                                                          |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `install_optional`     | By default install all open source libraries if missing. Set the `False` to disable.                                                                                                                                 |
| `install_licensed`     | By default installs all licensed libraries you have access to if they are missing. Set to `False` to disable.                                                                                                        |
| `include_dependencies` | Defaults to `True` which installs all depeendencies. If set to `False` pip will be executed with the `--no-deps` argument under the hood.                                                                            |
| `product`              | Specify product to install. By default installs everything you have access to.                                                                                                                                       |
| `only_download_jars`   | By default all libraries are installed to the current environment via pip. Set to False to disable installing Python dependencies and **only download jars** to the John Snow Labs home directory.                     |
| `jvm_install_type`     | Specify hardware install type, either `cpu`, `gpu`, `m1`, or `aarch` . Defaults to `cpu`. If you have a GPU and want to leverage CUDA, set `gpu`. If you are an Apple M1 or Arch user choose the corresponding types. |
| `py_install_type`      | Specify Python installation type to use, either `tar.gz` or `whl`, defaults to whl.                                                                                                                                   |
| `refresh_install`      | Delete any cached files before installing by removing John Snow Labs home folder. **This will delete your locally cached licenses**.                                                                                  |

</div><div class="h3-box" markdown="1">

### Automatic Databricks Installation

Use any of the databricks auth flows to enable the `johnsnowlabs` library to automatically install   
all open source and licensed features into a Databricks cluster.   
You additionally must use one of the [John Snow Labs License Authorization Flows](https://nlu.johnsnowlabs.com/docs/en/install#authorization-flows-overview) to give access to your John Snow
Labs license,which will be installed to your Databricks cluster.        
A John Snow Labs Home directory is constructed in the distributed Databricks File System`/dbfs/johnsnowlabs` which has
all Jars, Wheels and License Information to run all features in a Databricks cluster.

| Databricks Auth Flow Method | Description                                                                                                                                                                                                                                                                                           | Python `jsl.install()` usage                                                                                                        | 
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `Access Token`              | See [Databricks Documentation](https://docs.databricks.com/dev-tools/api/latest/authentication.html) for  extracting a token which you can provide to databricks access, see [Databricks Install Section](https://nlu.johnsnowlabs.com/docs/en/install#automatic-databricks-installation) for details | `jsl.install(databricks_cluster_id=my_cluster_id, databricks_host=my_databricks_host, databricks_token=my_access_databricks_token)` |

Where to find your Databricks Access Token:
![databricks_access_token.png](/assets/images/jsl_lib/install/databricks_access_token.png)

</div><div class="h3-box" markdown="1">


#### Databricks Cluster Creation Parameters

You can set the following parameters on the `jsl.install()` function to define properties of the cluster which will be created.  
See [Databricks Cluster Creation](https://docs.databricks.com/dev-tools/api/latest/clusters.html#create) for a detailed description of all parameters.

| Cluster creation Parameter | Default Value                              | 
|----------------------------|--------------------------------------------|
| block_till_cluster_ready   | `True`                                     | 
| num_workers                | `1`                                        | 
| cluster_name               | `John-Snow-Labs-Databricks-Auto-Cluster🚀` | 
| node_type_id               | `i3.xlarge`                                | 
| driver_node_type_id        | `i3.xlarge`                                | 
| spark_env_vars             | `None`                                     | 
| autotermination_minutes    | `60`                                       | 
| spark_version              | `10.5.x-scala2.12`                         | 
| spark_conf                 | `None`                                     | 
| auto_scale                 | `None`                                     | 
| aws_attributes             | `None`                                     | 
| ssh_public_keys            | `None`                                     | 
| custom_tags                | `None`                                     | 
| cluster_log_conf           | `None`                                     | 
| enable_elastic_disk        | `None`                                     | 
| cluster_source             | `None`                                     | 
| instance_pool_id           | `None`                                     | 
| headers                    | `None`                                     | 


The created cluster
![databricks_cluster.png](/assets/images/jsl_lib/install/databricks_cluster.png)


</div><div class="h3-box" markdown="1">

### License Variables Names for JSON and OS variables

The following variable names are checked when using a JSON or environment variables based approach for installing
licensed features or when using `jsl.start()` .         
You can find all of your license information on [https://my.johnsnowlabs.com/subscriptions](https://my.johnsnowlabs.com/subscriptions)

- `AWS_ACCESS_KEY_ID` : Assigned to you by John Snow Labs. Must be defined.
- `AWS_SECRET_ACCESS_KEY` : Assigned to you by John Snow Labs. Must be defined.
- `HC_SECRET` : The secret for a version of the enterprise NLP engine library. Changes between releases. Can be omitted if you don't
  have access to enterprise nlp.
- `HC_LICENSE` : Your license for the medical features. Can be omitted if you don't have a medical license.
- `OCR_SECRET` : The secret for a version of the Visual NLP (Spark OCR) library. Changes between releases. Can be omitted if you don't have a Visual NLP (Spark OCR) license.
- `OCR_LICENSE` : Your license for the Visual NLP (Spark OCR) features. Can be omitted if you don't have a Visual NLP (Spark OCR) license.
- `JSL_LEGAL_LICENSE`: Your license for Legal NLP Features
- `JSL_FINANCE_LICENSE` Your license for Finance NLP Features

NOTE: Instead of `JSL_LEGAL_LICENSE`, `HC_LICENSE` and `JSL_FINANCE_LICENSE` you may have 1 generic `SPARK_NLP_LICENSE`.

</div><div class="h3-box" markdown="1">

{:.h2-select}
## Installation Examples

### Auth Flow Examples

#### Via Auto Detection & Browser Login

All [default search locations ]() are searched, if any credentials are found they will be used.
If no credentials are auto-detected, a Browser Window will pop up, asking to authorize access to https://my.johnsnowlabs.com/
In Google Colab, a clickable button will appear, which will make a window pop up where you can authorize access to https://my.johnsnowlabs.com/.

```python
jsl.install()
``` 

</div><div class="h3-box" markdown="1">

#### Via Access Token

Get your License Token from [My John Snow Labs](https://my.johnsnowlabs.com/)

```python
jsl.install(access_token='secret')
```

Where you find the license
![access_token1.png](/assets/images/jsl_lib/install/access_token1.png)

</div><div class="h3-box" markdown="1">

#### Via Json Secrets file

Path to a JSON containing secrets, see [License Variable Names](https://nlu.johnsnowlabs.com/docs/en/install#license-variables-names-for-json-and-os-variables) for more details.

```python
jsl.install(json_file_path='my/secret.json')
``` 

</div><div class="h3-box" markdown="1">

#### Via Manually defining Secrets

Manually specify all secrets. Some of these can be omitted, see [License Variable Names](https://nlu.johnsnowlabs.com/docs/en/install#license-variables-names-for-json-and-os-variables) for more details.

```python
jsl.install(
    hc_license='Your HC License',
    fin_license='Your FIN License',
    leg_license='Your LEG License',
    enterprise_nlp_secret='Your NLP Secret',
    ocr_secret='Your OCR Secret',
    ocr_license='Your OCR License',
    aws_access_key='Your Access Key',
    aws_key_id='Your Key ID',
)
```

</div><div class="h3-box" markdown="1">

## Installation Target Examples

### Into Current Python Process

Uses sys.executable by default, i.e. the Python that is currently running the program.

```python
jsl.install() 
``` 

</div><div class="h3-box" markdown="1">

### Into Custom Python Env

Using specific python executable, which is not the currently running python.
Will use the provided python's executable pip module to install libraries.

```python
jsl.install(python_exec_path='my/python.exe')
``` 

</div><div class="h3-box" markdown="1">


### Into freshly created venv

Create a new Venv using the currently executing Pythons Venv Module.

```python
jsl.install(venv_creation_path='path/to/where/my/new/venv/will/be')
``` 

</div><div class="h3-box" markdown="1">

### Into Airgap/Offline Installation (Automatic)

Create a Zip with all Jars/Wheels/Licenses you need to run all libraries in an offline environment.
**Step1:**

```python
jsl.install(offline_zip_dir='path/to/where/my/zip/will/be')
``` 

**Step2:**
Transfer the zip file securely to your offline environment and unzip it. One option is the unix `scp`  command.

```shell
scp /to/where/my/zip/will/be/john_snow_labs.zip 123.145.231.001:443/remote/directroy
```

**Step3:**
Then from the **remote machine shell** unzip with:

```shell
# Unzip all files to ~/johnsowlabs
unzip remote/directory/jsl.zip -d ~/johnsowlabs
```

**Step4 (option1):**
Install the wheels via jsl:

```python 
# If you unzipped to ~/johnsowlabs, then just update this setting before running and jsl.install() handles the rest for you!
from johnsnowlabs import * 
jsl.settings.jsl_root = '~/johnsowlabs'
# Make sure you have Java 8 installed!
jsl.install()

```

**Step4 (option2):**
Install the wheels via pip: 

```shell
# Assuming you unzipped to ~/johnsnowlabs, you can install all wheels like this
pip install ~/johnsnowlabs/py_installs/*.whl
```

**Step5:**
Test your installation Via shell:

```shell
python -c "from johnsnowlabs import *;print(nlu.load('emotion').predict('Wow that easy!'))"
```

or in Python:

```python
from johnsnowlabs import *
jsl.load('emotion').predict('Wow that easy!')
```

</div><div class="h3-box" markdown="1">

### Into Airgap/Offline Manual

Download all files yourself from the URLs printed by jsl.install().
You will have to folly the Automatic Instructions starting from step (2) of the automatic installation.
I.e. provide the files somehow on your offline machine.

```python
# Print all URLs to files you need to provide on your host machine 
jsl.install(offline=True)
```
</div><div class="h3-box" markdown="1">
### Into a freshly created Databricks cluster automatically

To install in databricks you must provide your `accessToken` and `hostUrl`.
You can provide the secrets to the install function with any of the methods listed above, i.e. using `access_token`
, `browser`, `json_file`, or `manually defining secrets`
Your can get it from:

``` python
# Create a new Cluster with Spark NLP and all licensed libraries ready to go:
jsl.install(databricks_host='https://your_host.cloud.databricks.com', databricks_token = 'dbapi_token123',)
```
</div><div class="h3-box" markdown="1">

### Into Existing Databricks cluster Manual
If you do not wish to use the recommended **automatic installation** but instead want to install manually
you must install the `johnsnowlabs_for_databricks` pypi package instead of `johnsnowlabs` via the UI or any method of your choice.
![databricks_manual.png](/assets/images/jsl_lib/install/databricks_manual.png)


{:.h2-select}
## License Management & Caching

## Storage of License Data and License Search behaviour

The John Snow Labs library caches license data in `~/.johnsnowlabs/licenses` whenever a new one is provided .
After having provided license data once, you don't need to specify it again since the cached licensed will be used.
Use the `license_number` parameter to switch between multiple licenses.  
`Note: Locally cached licenses are numbered in the order they have been provided, starting at 0.`

</div><div class="h3-box" markdown="1">

## List all available licenses

This shows you all licenses for your account in https://my.johnsnowlabs.com/.         
Use this to decide which license number to install when installing via browser or access token.

```python
jsl.list_remote_licenses()
```

</div><div class="h3-box" markdown="1">

## List all locally cached licenses

Use this to decide which license number to use when using jsl.start() or jsl.install() to specify which local license
you want to load.

```python
jsl.list_local_licenses()
```

</div><div class="h3-box" markdown="1">

## License Search precedence

If there are multiples possible sources for licenses, the following order takes precedence:

1. Manually provided license data by defining all license parameters.
2. Browser/ Access Token.
3. `Os environment Variables` for any var names that match up with secret names.
4. `/content/*.json` for any json file smaller than 1 MB.
5. `current_working_dir/*.json` for any json smaller than 1 MB.
6. `~/.johnsnowlabs/licenses` for any licenses.

JSON files are scanned if they have any keys that match up with names of secrets.         
Name of the json file does not matter, file just needs to end with .json.

</div><div class="h3-box" markdown="1">

## Upgrade Flow

**Step 1:** Upgrade the johnsnowlabs library. 
```shell
pip install johnsnowlabs --upgrade
```
**Step 2:** Run install again, while using one [Authorization Flows](https://nlu.johnsnowlabs.com/docs/en/install#authorization-flows-overview). 
```python
jsl.install()
```



The John Snow Labs Teams are working early to push out new Releases and Features each week!       
Simply run `pip install johnsnowlabs --upgrade` to get the latest open **source libraries** updated.
Once the johnsnowlabs library is upgraded, it will detect any out-dated libraries any inform you
that you can upgrade them by running `jsl.install()` again.
You must run one of the [Authorization Flows](https://nlu.johnsnowlabs.com/docs/en/install#authorization-flows-overview) again,
to gian access to the latest enterprise libraries.

{:.h2-select}
## Next Steps & Frequently Asked Questions

</div><div class="h3-box" markdown="1">

## How to setup Java 8

- [Setup Java 8 on Windows](https://access.redhat.com/documentation/en-us/openjdk/8/html/openjdk_8_for_windows_getting_started_guide/getting_started_with_openjdk_for_windows)
- [Setup Java 8 on Linux](https://openjdk.java.net/install/)
- [Setup Java 8 on Mac](https://docs.oracle.com/javase/8/docs/technotes/guides/install/mac_jdk.html)

</div><div class="h3-box" markdown="1">

## Join our Slack channel

Join our channel, to ask for help and share your feedback. Developers and users can help each other get started here.

[NLU Slack](https://spark-nlp.slack.com/archives/C0196BQCDPY){:.button.button--info.button--rounded.button--md}

</div><div class="h3-box" markdown="1">

## Where to go next

If you want to get your hands dirty with any of the features check out the [NLU examples page](examples),
or [Licensed Annotators Overview](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators)
Detailed information about Johnsnowlabs Libraries APIs, concepts, components and more can be found on the following pages :

{:.list4}

- [Starting a Spark Session](start-a-sparksession)
- [John Snow Labs Library usage and import Overview](import-structure)
- [The NLU load function](load_api)
- [The NLU predict function](predict_api)
- [The NLU components spellbook](https://nlu.johnsnowlabs.com/docs/en/spellbook)
- [NLU Notebooks](notebooks)

</div></div>