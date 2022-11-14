---
layout: docs
header: true
seotitle: NLU | John Snow Labs
title: License Management & Caching
permalink: /docs/en/license_management
key: docs-concepts
modify_date: "2020-05-08"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">


### Storage of License Data and License Search behaviour

The John Snow Labs library caches license data in `~/.johnsnowlabs/licenses` whenever a new one is provided .
After having provided license data once, you don't need to specify it again since the cached licensed will be used.
Use the `local_license_number` and `remote_license_number` parameters to switch between multiple licenses.  
**Note:** Locally cached licenses are numbered in the order they have been provided, starting at 0.            
`remote_license_number=0` might not be the same as `local_license_number=0`.           
Use the following functions to see all your avaiable licenses.


</div><div class="h3-box" markdown="1">

### List all available licenses

This shows you all licenses for your account in https://my.johnsnowlabs.com/.         
Use this to decide which license number to install when installing via browser or access token.

```python
jsl.list_remote_licenses()
```

</div><div class="h3-box" markdown="1">

### List all locally cached licenses

Use this to decide which license number to use when using jsl.start() or jsl.install() to specify which local license
you want to load.

```python
jsl.list_local_licenses()
```

</div><div class="h3-box" markdown="1">

### License Search precedence

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

</div></div>