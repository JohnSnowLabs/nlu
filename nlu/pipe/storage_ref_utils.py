import logging
logger = logging.getLogger('nlu')

import inspect
from nlu.pipe.pipe_components import SparkNLUComponent

"""Storage Ref logic operations and utils"""
class StorageRefUtils():
    @staticmethod
    def has_storage_ref(component):
        """Storage ref is either on the model or nlu component defined """
        return StorageRefUtils.has_component_storage_ref_or_anno_storage_ref(component)


    @staticmethod
    def extract_storage_ref(component):
        """Extract storage ref from either a NLU component or NLP Annotator. First cheks if annotator has storage ref, otherwise check NLU attribute"""
        if StorageRefUtils.has_storage_ref(component):
            if hasattr(component.info,'storage_ref') : return component.info.storage_ref
            if StorageRefUtils.nlp_component_has_storage_ref(component.model) : return StorageRefUtils.nlu_extract_storage_ref_nlp_model(component)
        return ''


### SUB HELPERS

    @staticmethod
    def has_component_storage_ref_or_anno_storage_ref(component):
        """Storage ref is either on the model or nlu component defined """
        if StorageRefUtils.nlp_component_has_storage_ref(component.model): return True
        if StorageRefUtils.nlu_component_has_storage_ref(component) : return True


    @staticmethod
    def nlp_component_has_storage_ref(model):
        """Check if a storage ref is defined on the Spark NLP Annotator model"""
        for k, _ in model.extractParamMap().items():
            if k.name == 'storageRef': return True
        return False


    @staticmethod
    def extract_storage_ref_from_component(component):
        """Extract storage ref from a NLU component which embelished a Spark NLP Annotator"""
        if StorageRefUtils.nlu_component_has_storage_ref(component):
            return component.info.storage_ref
        elif StorageRefUtils.nlp_component_has_storage_ref(component):
            return StorageRefUtils.nlp_extract_storage_ref_nlp_model(component)
        else:
            return ''


    @staticmethod
    def nlu_extract_storage_ref_nlp_model(component):
        """Extract storage ref from a NLU component which embelished a Spark NLP Annotator"""
        return component.model.extractParamMap()[component.model.getParam('storageRef')]

    @staticmethod
    def nlu_component_has_storage_ref(component):
        """Check if a storage ref is defined on the Spark NLP Annotator embelished by the NLU Component"""
        if hasattr(component.info, 'storage_ref'): return True
        return False

    @staticmethod
    def nlp_extract_storage_ref_nlp_model(model):
        """Extract storage ref from a NLU component which embelished a Spark NLP Annotator"""
        return model.extractParamMap()[model.model.getParam('storageRef')]

