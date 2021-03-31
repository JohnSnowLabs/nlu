import inspect
from nlu.pipe.pipe_components import SparkNLUComponent
from nlu.pipe.component_utils import ComponentUtils
from nlu.pipe.storage_ref_utils import StorageRefUtils

# from nlu.pipe.pipeline import NLUPipeline
"""Pipe Level logic oprations and utils"""
class PipeUtils():
    @staticmethod
    def is_trainable_pipe(pipe):
        '''Check if pipe is trainable'''
        for c in pipe.components:
            if ComponentUtils.is_untrained_model(c):return True
        return False



    @staticmethod
    def enforece_AT_embedding_provider_output_col_name_schema_for_list_of_components  (pipe_list):
        """For every embedding provider, enforce that their output col is named <output_level>@storage_ref for output_levels word,chunk,sentence aka document , i.e. word_embed@elmo or sentence_embed@elmo etc.."""
        for c in pipe_list:
            if ComponentUtils.is_embedding_provider(c):
                level_AT_ref = ComponentUtils.extract_storage_ref_AT_notation(c, 'output')
                c.info.outputs = [level_AT_ref]
                c.info.spark_output_column_names = [level_AT_ref]
                c.model.setOutputCol(level_AT_ref[0])
                # if c.info.name =='ChunkEmbeddings' : c.model.setOutputCol(level_AT_ref[0])
                # else : c.model.setOutputCol(level_AT_ref)
        return pipe_list

    @staticmethod
    def enforce_AT_schema_on_embedding_processors  (pipe):
        """For every embedding provider and consumer, enforce that their output col is named <output_level>@storage_ref for output_levels word,chunk,sentence aka document , i.e. word_embed@elmo or sentence_embed@elmo etc.."""
        for c in pipe.components:
            if ComponentUtils.is_embedding_provider(c):
                level_AT_ref = ComponentUtils.extract_storage_ref_AT_notation(c, 'output')
                c.info.outputs = [level_AT_ref]
                c.info.spark_output_column_names = [level_AT_ref]
                c.model.setOutputCol(level_AT_ref[0])

            if ComponentUtils.is_embedding_consumer(c):
                input_embed_col = ComponentUtils.extract_embed_col(c)
                if '@' not in input_embed_col:
                    storage_ref = StorageRefUtils.extract_storage_ref(c)
                    new_embed_col_with_AT_notation = input_embed_col+"@"+storage_ref
                    c.info.inputs.remove(input_embed_col)
                    c.info.inputs.append(new_embed_col_with_AT_notation)
                    c.info.spark_input_column_names.remove(input_embed_col)
                    c.info.spark_input_column_names.append(new_embed_col_with_AT_notation)
                    c.model.setInputCols(c.info.inputs)

                # if c.info.name =='ChunkEmbeddings' : c.model.setOutputCol(level_AT_ref[0])
                # else : c.model.setOutputCol(level_AT_ref)
        return pipe


    @staticmethod
    def is_converter_component_resolution_reference(reference:str)-> bool:
        if 'chunk_emb' in reference : return True
        if 'chunk_emb' in reference : return True
