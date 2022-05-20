"""
Collection of universes shared across all libraries (NLP/HC/OCR), which are collections of atoms
"""
from nlu.universe.atoms import LicenseType, ComponentBackend, ComputeContext, ModelBucketType


class Licenses:
    """Definition of licenses"""
    ocr = LicenseType('ocr')
    hc = LicenseType('healthcare')
    open_source = LicenseType('open_source')


class ModelBuckets:
    """Definition of licenses"""
    ocr = ModelBucketType('clinical/ocr')
    hc = ModelBucketType("clinical/models")
    open_source = None #  ModelBucketType(None)


def license_to_bucket(license_to_resole: LicenseType) -> ModelBucketType:
    if license_to_resole == Licenses.open_source:
        return ModelBuckets.open_source
    if license_to_resole == Licenses.hc:
        return ModelBuckets.hc
    if license_to_resole == Licenses.ocr:
        return ModelBuckets.ocr
    return None


class ComponentBackends:
    """Definition of various Component providers/backends"""
    ocr = ComponentBackend('spark ocr')
    hc = ComponentBackend('spark hc')
    open_source = ComponentBackend('spark nlp')
    external = ComponentBackend('external')


class ComputeContexts:
    """Computation contexts which describe where which framework is used to compute results """
    spark = ComputeContext('spark')
    pandas = ComputeContext('pandas')
    numpy = ComputeContext('numpy')
    modin = ComputeContext('modin')
    py_arrow = ComputeContext('py_arrow')
