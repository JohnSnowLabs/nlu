---
layout: docs
seotitle: NLU | John Snow Labs
title: Installation
permalink: /docs/en/install_licensed_quick
key: docs-install
modify_date: "2020-05-26"
header: true
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

To install the **johnsnowlabs Python library** and all of John Snow Labs **licensed libraries**, just run

1. run in your `shell`
```shell 
pip install johnsnowlabs
```
2. run in a `Python Shell`
```python
from johnsnowlabs import *
jsl.install()
```
This will display a **Browser Window Pop Up**  or show a **Clickable Button with Pop Up**.        
Click on the **Authorize** button to allow the library to connect to your account on my.JohnSnowLabs.com and access you licenses.          
This will enable the installation and use of all licensed products for which you have a valid license.

Colab Button:
![install_button_colab.png](/assets/images/jsl_lib/install/install_button_colab.png)

Where the Pop-Up leads you to:
![install_pop_up.png](/assets/images/jsl_lib/install/install_pop_up.png)

After clicking **Authorize**:
![install_logs_colab.png](/assets/images/jsl_lib/install/install_logs_colab.png)

To quickly test your installation, run in a `Python shell`
```python

```


for alternative installation options see [Custom Installation](/docs/en/install_advanced)