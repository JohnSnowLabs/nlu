import inspect
from nlu.pipe.pipe_components import SparkNLUComponent
from nlu.pipe.component_utils import ComponentUtils

# from nlu.pipe.pipeline import NLUPipeline
"""Pipe Level logic oprations and utils"""
class PipeUtils():
    @staticmethod
    def is_trainable_pipe(pipe):
        '''Check if pipe is trainable'''
        for c in pipe.components:
            if PipeUtils.is_untrained_model(c):return True
        return False
    @staticmethod
    def is_untrained_model(component):
        '''
        Check for a given component if it is an embelishment of an traianble model.
        In this case we will ignore embeddings requirements further down the logic pipeline
        :param component: Component to check
        :return: True if it is trainable, False if not
        '''
        if 'is_untrained' in dict(inspect.getmembers(component.info)).keys(): return True
        return False




    @staticmethod
    def extract_storage_ref(component):
        """Extract storage ref from either a NLU component or NLP Annotator. First cheks if annotator has storage ref, otherwise check NLU attribute"""
        if PipeUtils.has_storage_ref(component):
            if hasattr(component.info,'storage_ref') : return component.info.storage_ref
            if PipeUtils.nlp_component_has_storage_ref(component.model) : return PipeUtils.nlu_extract_storage_ref_nlp_model(component)
        return ''

    @staticmethod
    def nlu_extract_storage_ref_nlp_model(component):
        """Extract storage ref from a NLU component which embelished a Spark NLP Annotator"""
        return component.model.extractParamMap()[component.model.getParam('storageRef')]
    @staticmethod
    def extract_storage_ref_from_component(component):
        """Extract storage ref from a NLU component which embelished a Spark NLP Annotator"""
        if PipeUtils.nlu_component_has_storage_ref(component):
            return component.info.storage_ref
        elif PipeUtils.nlp_component_has_storage_ref(component):
            return PipeUtils.nlp_extract_storage_ref_nlp_model(component)
        else:
            return ''
    @staticmethod
    def has_component_storage_ref_or_anno_storage_ref(component):
        """Storage ref is either on the model or nlu component defined """
        if PipeUtils.nlp_component_has_storage_ref(component.model): return True
        if PipeUtils.nlu_component_has_storage_ref(component) : return True
    @staticmethod
    def has_storage_ref(component):
        """Storage ref is either on the model or nlu component defined """
        return PipeUtils.has_component_storage_ref_or_anno_storage_ref(component)

    @staticmethod
    def extract_nlp_storage_ref_from_nlp_model_if_set(model):
        """Extract storage ref from a Spark NLP Annotator model"""
        if PipeUtils.nlu_extract_storage_ref_nlp_model(model): return ''
        else : return PipeUtils.extract_nlp_storage_ref_from_nlp_model(model)

    @staticmethod
    def nlu_component_has_storage_ref(component):
        """Check if a storage ref is defined on the Spark NLP Annotator embelished by the NLU Component"""
        if hasattr(component.info, 'storage_ref'): return True
        return False

    @staticmethod
    def nlp_component_has_storage_ref(model):
        """Check if a storage ref is defined on the Spark NLP Annotator model"""
        for k, _ in model.extractParamMap().items():
            if k.name == 'storageRef': return True
        return False

    @staticmethod
    def nlp_extract_storage_ref_nlp_model(model):
        """Extract storage ref from a NLU component which embelished a Spark NLP Annotator"""
        return model.extractParamMap()[model.model.getParam('storageRef')]



    @staticmethod
    def enforece_AT_embedding_provider_output_col_name_schema_for_list_of_components  (pipe_list):
        """For every embedding provider, enforce that their output col is named <output_level>@storage_ref for output_levels word,chunk,sentence aka document , i.e. word_embed@elmo or sentence_embed@elmo etc.."""
        for c in pipe_list:
            if ComponentUtils.is_embedding_provider(c):
                level_AT_ref = ComponentUtils.extract_storage_ref_AT_column(c,'output')
                c.info.outputs = [level_AT_ref]
                c.info.spark_output_column_names = [level_AT_ref]
                c.model.setOutputCol(level_AT_ref[0])
                # if c.info.name =='ChunkEmbeddings' : c.model.setOutputCol(level_AT_ref[0])
                # else : c.model.setOutputCol(level_AT_ref)
        return pipe_list

    @staticmethod
    def enforece_AT_embedding_provider_output_col_name_schema_for_pipeline  (pipe):
        """For every embedding provider, enforce that their output col is named <output_level>@storage_ref for output_levels word,chunk,sentence aka document , i.e. word_embed@elmo or sentence_embed@elmo etc.."""
        for c in pipe.components:
            if ComponentUtils.is_embedding_provider(c):
                level_AT_ref = ComponentUtils.extract_storage_ref_AT_column(c,'output')
                c.info.outputs = [level_AT_ref]
                c.info.spark_output_column_names = [level_AT_ref]
                c.model.setOutputCol(level_AT_ref[0])
                # if c.info.name =='ChunkEmbeddings' : c.model.setOutputCol(level_AT_ref[0])
                # else : c.model.setOutputCol(level_AT_ref)
        return pipe


    @staticmethod
    def is_converter_component_resolution_reference(reference:str)-> bool:
        if 'chunk_emb' in reference : return True
        if 'chunk_emb' in reference : return True
