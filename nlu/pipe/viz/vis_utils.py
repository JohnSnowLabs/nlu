"""Utils for interfacing with the Spark-NLP-Display lib"""
class VizUtils():
    @staticmethod
    def has_storage_ref(component):
        """Storage ref is either on the model or nlu component defined """
        return StorageRefUtils.has_component_storage_ref_or_anno_storage_ref(component)
