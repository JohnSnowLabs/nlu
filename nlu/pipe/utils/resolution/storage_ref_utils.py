import logging

from nlu.pipe.nlu_component import NluComponent
from nlu.pipe.utils.resolution import uid_to_storage_ref as uid2storageref

logger = logging.getLogger('nlu')

"""Storage Ref logic operations and utils"""


class StorageRefUtils:
    @staticmethod
    def has_storage_ref(component: NluComponent):
        """Storage ref is either on the model_anno_obj or nlu component_to_resolve defined """
        return component.has_storage_ref

    @staticmethod
    def extract_storage_ref(component: NluComponent):
        """Extract storage ref from either a NLU component_to_resolve or NLP Annotator. First checks if annotator has storage
        ref, otherwise check NLU attribute """
        if StorageRefUtils.has_storage_ref(component):
            return StorageRefUtils.nlp_extract_storage_ref_nlp_model(component)
        else:
            raise ValueError(
                f'Tried to extract storage ref from component_to_resolve which has no storageref ! Component = {component}')

    @staticmethod
    def fallback_storage_ref_resolutions(storage_ref):
        """
        For every storage ref result, we check if its storage ref is defined as its UID and if a fallback storageref
        is available. If available, alternative is returned, otherwise original
        """
        if storage_ref in uid2storageref.mappings.keys():
            return uid2storageref.mappings[storage_ref]
        else:
            return storage_ref

    @staticmethod
    def has_component_storage_ref_or_anno_storage_ref(component: NluComponent):
        """Storage ref is either on the model_anno_obj or nlu component_to_resolve defined """
        return component.has_storage_ref

    @staticmethod
    def nlp_component_has_storage_ref(model):
        """Check if a storage ref is defined on the Spark NLP Annotator model_anno_obj"""
        for k, _ in model.extractParamMap().items():
            if k.name == 'storageRef':
                return True
        return False

    @staticmethod
    def extract_storage_ref_from_component(component):
        """Extract storage ref from a NLU component_to_resolve which embellished a Spark NLP Annotator"""
        if StorageRefUtils.nlu_component_has_storage_ref(component):
            return component.info.storage_ref
        elif StorageRefUtils.nlp_component_has_storage_ref(component):
            return StorageRefUtils.nlp_extract_storage_ref_nlp_model(component)
        else:
            return ''

    @staticmethod
    def nlu_extract_storage_ref_nlp_model(component):
        """Extract storage ref from a NLU component_to_resolve which embellished a Spark NLP Annotator"""
        return component.model.extractParamMap()[component.model.getParam('storageRef')]

    @staticmethod
    def nlu_component_has_storage_ref(component):
        """Check if a storage ref is defined on the Spark NLP Annotator embellished by the NLU Component"""
        if hasattr(component.info, 'storage_ref'):
            return True
        return False

    @staticmethod
    def nlp_extract_storage_ref_nlp_model(component: NluComponent):
        """Extract storage ref from a NLU component_to_resolve which embellished a Spark NLP Annotator"""
        # Embedding Converters don't have storage ref attribute on class, but NLU component_to_resolve has attribute for it
        params = list(component.model.extractParamMap().keys())
        for p in params:
            if p.name == 'storageRef':
                storage_ref = component.model.extractParamMap()[component.model.getParam('storageRef')]
                if not storage_ref:
                    # For untrained components storage ref will be none
                    return ''
                else:
                    return storage_ref
        if not component.storage_ref:
            return ''
        return component.storage_ref
