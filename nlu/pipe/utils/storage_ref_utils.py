import logging
from  nlu.pipe.utils import uid_to_storageref as uid2storageref
logger = logging.getLogger('nlu')


"""Storage Ref logic operations and utils"""
class StorageRefUtils():
    @staticmethod
    def has_storage_ref(component):
        """Storage ref is either on the model or nlu component defined """
        return StorageRefUtils.has_component_storage_ref_or_anno_storage_ref(component)


    @staticmethod
    def extract_storage_ref(component, prefer_anno=False):
        """Extract storage ref from either a NLU component or NLP Annotator. First cheks if annotator has storage ref, otherwise check NLU attribute"""
        if StorageRefUtils.has_storage_ref(component):
            if hasattr(component.info,'storage_ref') and not prefer_anno: return StorageRefUtils.fallback_storage_ref_resolutions(component.info.storage_ref)
            if StorageRefUtils.nlp_component_has_storage_ref(component.model) :
                return StorageRefUtils.fallback_storage_ref_resolutions( StorageRefUtils.nlu_extract_storage_ref_nlp_model(component))
        return ''

    @staticmethod
    def fallback_storage_ref_resolutions(storage_ref):
        """
            For every storage ref result, we check if its storage ref is defined as its UID and if a fallback storageref is avaiable.
            If avaiable, alternative is returned, otherwise original
        """
        if storage_ref in uid2storageref.mappings.keys():
            return uid2storageref.mappings[storage_ref]
        else : return storage_ref


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


