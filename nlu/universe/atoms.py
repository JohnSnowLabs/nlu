"""
Embellishments of primitive datatypes for logical building blocks in our universe for various domain specific concepts
Helper Classes for Type Checking and IDE Compiler Hints because NewType lib does not support typeChecking
"""


# I.e. OCR, NLP, NLP-HC,
class JslUniverse(str):
    pass

class NluRef(str):
    pass

# Lang ISO TODO define universe
class LanguageIso(str):
    pass


# Engine used for computation, i.e. Pandas, Spark, Modin
class ComputeContext(str):
    pass


# Backend used for Component
class ComponentBackend(str):
    pass


# Features provided from non JSL annotators
class ExternalFeature(str):
    pass


# external feature generator or from 3rd Party Lib
class ExternalAnno(str):
    pass


# Any JSL generated Feature from annotators
class JslFeature(str):
    pass


# Any JSL Anno reference. Unique Name defined by NLU to identify a specific class. Not using actual JVM/PY class name to make it language agnostic
class JslAnnoId(str):
    pass


# OutputLevel of a NLP annotator
class NlpLevel(str):
    pass


# Reference to Python class of an Annotator
class JslAnnoPyClass(str):
    pass


# Reference to Java class of an Annotator
class JslAnnoJavaClass(str):
    pass


# Reference to LicenseTypes
class LicenseType(str):
    pass

# Reference to Model Buckets
class ModelBucketType(str):
    pass
