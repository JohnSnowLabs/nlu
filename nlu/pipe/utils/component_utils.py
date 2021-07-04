from typing import List
import logging
logger = logging.getLogger('nlu')
import inspect
from nlu.pipe.pipe_components import SparkNLUComponent
from nlu.pipe.utils.storage_ref_utils import StorageRefUtils

class ComponentUtils():
    """Component and Column Level logic operations and utils"""
    @staticmethod
    def config_chunk_embed_converter(converter:SparkNLUComponent)->SparkNLUComponent:
        '''For a Chunk to be added to a pipeline, configure its input/output and set storage ref to amtch the storage ref and
        enfore storage ref notation. This will be used to infer backward later which component should feed this consumer'''
        storage_ref = StorageRefUtils.extract_storage_ref(converter)
        input_embed_col = ComponentUtils.extract_embed_col(converter)
        new_embed_col_with_AT_notation = input_embed_col+"@"+storage_ref
        converter.info.inputs.remove(input_embed_col)
        converter.info.inputs.append(new_embed_col_with_AT_notation)
        converter.info.spark_input_column_names.remove(input_embed_col)
        converter.info.spark_input_column_names.append(new_embed_col_with_AT_notation)
        converter.model.setInputCols(converter.info.inputs)


        return converter


    @staticmethod
    def clean_irrelevant_features(feature_list, remove_AT_notation = False):
        '''
        Remove irrelevant features from a list of component features
        Also remove the @notation from names, since they are irrelevant for ordering
        :param feature_list: list of features
        :param remove_AT_notation: remove AT notation from c names if true. Used for sorting
        :return: list with only relevant feature names
        '''
        # remove irrelevant missing features for pretrained models
        if 'text'            in feature_list: feature_list.remove('text')
        if 'raw_text'        in feature_list: feature_list.remove('raw_text')
        if 'raw_texts'       in feature_list: feature_list.remove('raw_texts')
        if 'label'           in feature_list: feature_list.remove('label')
        if 'sentiment_label' in feature_list: feature_list.remove('sentiment_label')
        if '%%%feature_elements%%%' in feature_list : feature_list.remove('%%%feature_elements%%%')
        if remove_AT_notation:
            new_cs = []
            for c in feature_list :new_cs.append(c.split("@")[0])
            return new_cs
        return feature_list


    @staticmethod
    def component_has_embeddings_requirement(component:SparkNLUComponent):
        '''
        Check for the input component, wether it depends on some embedding. Returns True if yes, otherwise False.
        :param component:  The component to check
        :return: True if the component needs some specifc embedding (i.e.glove, bert, elmo etc..). Otherwise returns False
        '''

        if type(component) == list or type(component) == set:
            for feature in component:
                if 'embed' in feature: return True
            return False
        else:
            if component.info.type == 'word_embeddings': return False
            if component.info.type == 'sentence_embeddings': return False
            if component.info.type == 'chunk_embeddings': return False
            for feature in component.info.inputs:
                if 'embed' in feature: return True
        return False
    @staticmethod
    def component_has_embeddings_provisions(component:SparkNLUComponent):
        '''
        Check for the input component, wether it depends on some embedding. Returns True if yes, otherwise False.
        :param component:  The component to check
        :return: True if the component needs some specifc embedding (i.e.glove, bert, elmo etc..). Otherwise returns False
        '''
        if type(component) == type(list) or type(component) == type(set):
            for feature in component:
                if 'embed' in feature: return True
            return False
        else:
            for feature in component.info.outputs:
                if 'embed' in feature: return True
        return False


    @staticmethod
    def extract_storage_ref_AT_notation_for_embeds(component:SparkNLUComponent, col='input'):
        '''
        Extract <col>_embed_col@storage_ref notation from a component if it has a storage ref, otherwise '

        :param component:  To extract notation from
        :cols component:  Wether to extract for the input or output col
        :return: '' if no storage_ref, <col>_embed_col@storage_ref otherwise
        '''
        # if not StorageRefUtils.has_storage_ref(component) :
        #     if   col =='input' : return component.info.inputs
        #     elif col =='output': return component.info.outputs
        if   col =='input'  : e_col    = next(filter(lambda s : 'embed' in s, component.info.inputs))
        elif col =='output' : e_col    = next(filter(lambda s : 'embed' in s, component.info.outputs))

        stor_ref = StorageRefUtils.extract_storage_ref(component)
        return e_col + '@' + stor_ref

    @staticmethod
    def is_embedding_provider(component:SparkNLUComponent) -> bool:
        """Check if a NLU Component returns embeddings """
        if 'embed' in component.info.outputs[0]:
            return True
        else:
            return False

    @staticmethod
    def is_embedding_consumer(component:SparkNLUComponent) -> bool:
        """Check if a NLU Component consumes embeddings """
        return any('embed' in i for i in component.info.inputs)


    @staticmethod
    def is_embedding_converter  (component : SparkNLUComponent) -> bool:
        """Check if NLU component is embedding converter """
        if component.info.name in ['chunk_embedding_converter', 'sentence_embedding_converter']: return True
        return False


    @staticmethod
    def is_NER_provider(component:SparkNLUComponent) -> bool:
        """Check if a NLU Component wraps a NER/NER-Medical model """
        return component.info.name.split('@')[0] in ['named_entity_recognizer_dl', 'medical_named_entity_recognizer_dl','ner', 'named_entity_recognizer_dl_healthcare', 'med_ner']

    @staticmethod
    def is_NER_converter(component:SparkNLUComponent) -> bool:
        """Check if a NLU Component wraps a NER-IOB to NER-Pr etty converter """
        return component.info.name in ['NerToChunkConverter','ner_to_chunk_converter','ner_to_chunk_converter_licensed']

    @staticmethod
    def extract_NER_col(component:SparkNLUComponent, column='input')->str:
        """Extract the exact name of the NER column in the component"""
        if column == 'input':
            for f in component.info.spark_input_column_names:
                if 'ner' in f : return f
        if column == 'output':
            for f in component.info.spark_output_column_names:
                if 'ner' in f : return f

    @staticmethod
    def extract_NER_converter_col(component:SparkNLUComponent, column='input')->str:
        """Extract the exact name of the NER-converter  column in the component"""
        if column == 'input':
            for f in component.info.spark_input_column_names:
                if 'ner' in f : return f
        if column == 'output':
            for f in component.info.spark_output_column_names:
                if 'entities' in f : return f



    @staticmethod
    def extract_embed_col(component:SparkNLUComponent, column='input')->str:
        """Extract the exact name of the embed column in the component"""
        if column == 'input':
            for c in component.info.spark_input_column_names:
                if 'embed' in c : return c
        if column == 'output':
            for c in component.info.spark_output_column_names:
                if 'embed' in c : return c

    @staticmethod
    def is_untrained_model(component:SparkNLUComponent)->bool:
        '''
        Check for a given component if it is an embelishment of an traianble model.
        In this case we will ignore embeddings requirements further down the logic pipeline
        :param component: Component to check
        :return: True if it is trainable, False if not
        '''
        if 'is_untrained' in dict(inspect.getmembers(component.info)).keys(): return True
        return False



    @staticmethod
    def set_storage_ref_attribute_of_embedding_converters(pipe_list):
        """For every embedding converter, we set storage ref attr on it, based on what the storage ref from it's provider is """
        for converter in pipe_list:
            if ComponentUtils.is_embedding_provider(converter) and  ComponentUtils.is_embedding_converter(converter) :
                embed_col = ComponentUtils.extract_embed_col(converter)
                for provider in pipe_list:
                    if embed_col in provider.info.outputs:
                        converter.info.storage_ref = StorageRefUtils.extract_storage_ref(provider)
                # if c.info.name =='ChunkEmbeddings' : c.model.setOutputCol(level_AT_ref[0])
                # else : c.model.setOutputCol(level_AT_ref)
        return pipe_list

    @staticmethod
    def extract_embed_level_identity(component, col='input'):
        """Figure out if component feeds on chunk/sent aka doc/word emb for either nput or output cols"""
        if col =='input':
            if any(filter(lambda s : 'document_embed' in s , component.info.inputs)): return 'document_embeddings'
            if any(filter(lambda s : 'sentence_embed' in s , component.info.inputs)): return 'sentence_embeddings'
            if any(filter(lambda s : 'chunk_embed' in s , component.info.inputs)): return 'chunk_embeddings'
            if any(filter(lambda s : 'token_embed' in s , component.info.inputs)): return 'token_embeddings'
        elif col == 'output':
            if any(filter(lambda s : 'document_embed' in s , component.info.outputs)): return 'document_embeddings'
            if any(filter(lambda s : 'sentence_embed' in s , component.info.outputs)): return 'sentence_embeddings'
            if any(filter(lambda s : 'chunk_embed' in s , component.info.outputs)): return 'chunk_embeddings'
            if any(filter(lambda s : 'token_embed' in s , component.info.outputs)): return 'token_embeddings'

    @staticmethod
    def are_producer_consumer_matches(e_consumer:SparkNLUComponent,e_provider:SparkNLUComponent) -> bool:
        """Check for embedding_consumer and embedding_producer if they match storage_ref and output level wise wise """
        if StorageRefUtils.extract_storage_ref(e_consumer)== StorageRefUtils.extract_storage_ref(e_provider):
            if ComponentUtils.extract_embed_level_identity(e_consumer, 'input') ==  ComponentUtils.extract_embed_level_identity(e_provider, 'output'):
                return True
            ## TODO FALL BACK FOR BAD MATCHES WHICH ACTUALLY MATCH-> consult name space
        return False


    @staticmethod
    def get_nlu_ref_identifier(component:SparkNLUComponent) -> str:
        """The tail of a NLU ref after splitting on '.' gives a unique identifier for NON-Aliased components
         If result is '' , model UID will be used as identifier
         """
        tail = ''

        if hasattr(component.info, 'nlu_ref') and component.info.nlu_ref !='':
            tail = component.info.nlu_ref.split('.')[-1].split('@')[-1]
        if tail == '' :
            logger.warning(f"Could not deduct tail from component={component}. This is intended for CustomModelComponents used in offline mode")
            tail = str(component.model)
        return tail

    @staticmethod
    def remove_storage_ref_from_features(features:List[str]):
        """Clean storage ref from every str in list """
        return [f.split('@')[0] for f in features]
