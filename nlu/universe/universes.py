"""
Collection of universes shared across all libraries (NLP/HC/OCR), which are collections of atoms
"""
from nlu.universe.atoms import LicenseType, ComponentBackend, ComputeContext


class Licenses:
    """Definition of licenses"""
    ocr = LicenseType('ocr')
    hc = LicenseType('healthcare')
    open_source = LicenseType('open_source')


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