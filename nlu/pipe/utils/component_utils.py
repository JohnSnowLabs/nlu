import logging
from typing import List

from nlu.pipe.nlu_component import NluComponent
from nlu.universe.feature_node_ids import NLP_NODE_IDS, NLP_HC_NODE_IDS
from nlu.universe.logic_universes import AnnoTypes

logger = logging.getLogger('nlu')
from nlu.pipe.utils.resolution.storage_ref_utils import StorageRefUtils
from nlu.universe.feature_universes import NLP_FEATURES, OCR_FEATURES


class ComponentUtils:
    """Component and Column Level logic operations and utils"""

    @staticmethod
    def config_chunk_embed_converter(converter: NluComponent) -> NluComponent:
        '''For a Chunk to be added to a pipeline, configure its input/output and set storage ref to amtch the storage ref and
        enfore storage ref notation. This will be used to infer backward later which component_to_resolve should feed this consumer'''
        storage_ref = StorageRefUtils.extract_storage_ref(converter)
        input_embed_col = ComponentUtils.extract_embed_col(converter)
        new_embed_col_with_AT_notation = input_embed_col + "@" + storage_ref
        converter.info.inputs.remove(input_embed_col)
        converter.info.inputs.append(new_embed_col_with_AT_notation)
        converter.info.spark_input_column_names.remove(input_embed_col)
        converter.info.spark_input_column_names.append(new_embed_col_with_AT_notation)
        converter.model.setInputCols(converter.info.inputs)

        return converter

    @staticmethod
    def clean_irrelevant_features(feature_list, remove_AT_notation=False, remove_text=True):
        '''
        Remove irrelevant features from a list of component_to_resolve features
        Also remove the @notation from names, since they are irrelevant for ordering
        :param feature_list: list of features
        :param remove_AT_notation: remove AT notation from os_components names if true. Used for sorting
        :return: list with only relevant feature names
        '''
        # remove irrelevant missing features for pretrained models
        # Most of these should be provided externally by the user and cannot be resolved
        if 'text' in feature_list and remove_text:
            feature_list.remove('text')
        if 'raw_text' in feature_list:
            feature_list.remove('raw_text')
        if 'raw_texts' in feature_list:
            feature_list.remove('raw_texts')
        if 'label' in feature_list:
            feature_list.remove('label')
        if 'sentiment_label' in feature_list:
            feature_list.remove('sentiment_label')
        if '%%%feature_elements%%%' in feature_list:
            feature_list.remove('%%%feature_elements%%%')
        if NLP_FEATURES.ANY in feature_list:
            feature_list.remove(NLP_FEATURES.ANY)
        if NLP_FEATURES.RAW_QUESTION in feature_list:
            feature_list.remove(NLP_FEATURES.RAW_QUESTION)
        if NLP_FEATURES.RAW_QUESTION_CONTEXT in feature_list:
            feature_list.remove(NLP_FEATURES.RAW_QUESTION_CONTEXT)

        if OCR_FEATURES.BINARY_IMG in feature_list:
            feature_list.remove(OCR_FEATURES.BINARY_IMG)
        if OCR_FEATURES.FILE_PATH in feature_list:
            feature_list.remove(OCR_FEATURES.FILE_PATH)
        if OCR_FEATURES.BINARY_DOCX in feature_list:
            feature_list.remove(OCR_FEATURES.BINARY_DOCX)
        if OCR_FEATURES.BINARY_PDF in feature_list:
            feature_list.remove(OCR_FEATURES.BINARY_PDF)

        if remove_AT_notation:
            new_cs = []
            for c in feature_list:
                new_cs.append(c.split("@")[0])
            return new_cs
        return feature_list

    @staticmethod
    def component_has_embeddings_requirement(component: NluComponent):
        '''
        Check for the input component_to_resolve, wether it depends on some embedding. Returns True if yes, otherwise False.
        :param component:  The component_to_resolve to check
        :return: True if the component_to_resolve needs some specifc embedding (i.e.glove, bert, elmo etc..). Otherwise returns False
        '''
        return component.is_storage_ref_consumer

    @staticmethod
    def extract_storage_ref_AT_notation_for_embeds(component: NluComponent, col='input'):
        '''
        Extract <col>_embed_col@storage_ref notation from a component_to_resolve if it has a storage ref, otherwise '
        :param component:  To extract notation from
        :cols component_to_resolve:  Wether to extract for the input or output col
        :return: '' if no storage_ref, <col>_embed_col@storage_ref otherwise
        '''
        if col == 'input':
            e_col = next(filter(lambda s: 'embed' in s, component.spark_input_column_names))
        elif col == 'output':
            e_col = next(filter(lambda s: 'embed' in s, component.spark_output_column_names))
        stor_ref = StorageRefUtils.extract_storage_ref(component)
        return e_col + '@' + stor_ref

    @staticmethod
    def is_embedding_provider(component: NluComponent) -> bool:
        """Check if a NLU Component returns/generates embeddings """
        return component.is_storage_ref_producer

    @staticmethod
    def is_embedding_consumer(component: NluComponent) -> bool:
        """Check if a NLU Component consumes embeddings """
        return component.is_storage_ref_consumer

    @staticmethod
    def is_embedding_converter(component: NluComponent) -> bool:
        """Check if NLU component_to_resolve is embedding converter """
        return component.name in [NLP_NODE_IDS.SENTENCE_EMBEDDINGS_CONVERTER,
                                  NLP_NODE_IDS.SENTENCE_EMBEDDINGS_CONVERTER]

    @staticmethod
    def is_NER_provider(component: NluComponent) -> bool:
        """Check if a NLU Component wraps a NER/NER-Medical model_anno_obj """

        if component.name in [NLP_HC_NODE_IDS.MEDICAL_NER, NLP_HC_NODE_IDS.TRAINABLE_MEDICAL_NER, NLP_NODE_IDS.NER_DL,
                              NLP_NODE_IDS.TRAINABLE_NER_DL, NLP_NODE_IDS.TRAINABLE_NER_CRF,
                              NLP_NODE_IDS.NER_CRF]: return True
        if component.type == AnnoTypes.TRANSFORMER_TOKEN_CLASSIFIER: return True



    @staticmethod
    def is_NER_IOB_token_classifier(component: NluComponent) -> bool:
        """Check if a Token Classifier uses IOB PRediction format"""
        if not hasattr(component.model,'getClasses'):
            return False
        return any(['-' in label for label in component.model.getClasses()])

    @staticmethod
    def is_NER_converter(component: NluComponent) -> bool:
        """Check if a NLU Component wraps a NER-IOB to NER-Pr etty converter """
        return component.name in [NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL, NLP_NODE_IDS.NER_CONVERTER]

    @staticmethod
    def extract_NER_col(component: NluComponent, column='input') -> str:
        """Extract the exact name of the NER IOB column in the component_to_resolve"""
        if column == 'input':
            for f in component.in_types:
                if f in [NLP_FEATURES.NAMED_ENTITY_IOB, NLP_FEATURES.TOKEN_CLASSIFICATION]:
                    return f
        if column == 'output':
            for f in component.out_types:
                if f in [NLP_FEATURES.NAMED_ENTITY_IOB, NLP_FEATURES.TOKEN_CLASSIFICATION]:
                    return f
        raise ValueError(f"Could not find NER col for component_to_resolve ={component}")

    @staticmethod
    def extract_NER_converter_col(component: NluComponent, column='input') -> str:
        """Extract the exact name of the NER-converter  column in the component_to_resolve"""
        if column == 'input':
            for f in component.in_types:
                if f == NLP_FEATURES.NAMED_ENTITY_IOB:
                    return f
        if column == 'output':
            for f in component.out_types:
                if f == NLP_FEATURES.NAMED_ENTITY_CONVERTED:
                    return f
        raise ValueError(f"Could not find NER Converter col for component_to_resolve ={component}")

    @staticmethod
    def extract_embed_col(component: NluComponent, column='input') -> str:
        """Extract the exact name of the embed column in the component_to_resolve"""
        if column == 'input':
            for c in component.spark_input_column_names:
                if 'embed' in c: return c
        if column == 'output':
            for c in component.spark_output_column_names:
                if 'embed' in c: return c
        raise ValueError(f"Could not find Embed col for component_to_resolve ={component}")

    @staticmethod
    def set_storage_ref_attribute_of_embedding_converters(pipe_list: List[NluComponent]):
        """For every embedding converter, we set storage ref attr on it, based on what the storage ref from it's provider is """
        for converter in pipe_list:
            if ComponentUtils.is_embedding_provider(converter) and ComponentUtils.is_embedding_converter(converter):
                # First find the embed col of the converter
                embed_col = ComponentUtils.extract_embed_col(converter)
                for provider in pipe_list:
                    # Now find the Embedding generator that is feeding the converter
                    if embed_col in provider.spark_input_column_names:
                        converter.storage_ref = StorageRefUtils.nlp_extract_storage_ref_nlp_model(provider.model)
                        # converter.storage_ref = StorageRefUtils.extract_storage_ref(provider)
        return pipe_list

    @staticmethod
    def extract_embed_level_identity(component, col='input'):
        """Figure out if component_to_resolve feeds on chunk/sent aka doc/word emb for either nput or output cols"""
        if col == 'input':
            if any(filter(lambda s: 'document_embed' in s, component.info.inputs)): return 'document_embeddings'
            if any(filter(lambda s: 'sentence_embed' in s, component.info.inputs)): return 'sentence_embeddings'
            if any(filter(lambda s: 'chunk_embed' in s, component.info.inputs)): return 'chunk_embeddings'
            if any(filter(lambda s: 'token_embed' in s, component.info.inputs)): return 'token_embeddings'
        elif col == 'output':
            if any(filter(lambda s: 'document_embed' in s, component.out_types)): return 'document_embeddings'
            if any(filter(lambda s: 'sentence_embed' in s, component.out_types)): return 'sentence_embeddings'
            if any(filter(lambda s: 'chunk_embed' in s, component.out_types)): return 'chunk_embeddings'
            if any(filter(lambda s: 'token_embed' in s, component.out_types)): return 'token_embeddings'

    @staticmethod
    def are_producer_consumer_matches(e_consumer: NluComponent, e_provider: NluComponent) -> bool:
        """Check for embedding_consumer and embedding_producer if they match storage_ref and output level wise wise """
        if StorageRefUtils.extract_storage_ref(e_consumer) == StorageRefUtils.extract_storage_ref(e_provider):
            if ComponentUtils.extract_embed_level_identity(e_consumer,
                                                           'input') == ComponentUtils.extract_embed_level_identity(
                e_provider, 'output'):
                return True
            ## TODO FALL BACK FOR BAD MATCHES WHICH ACTUALLY MATCH-> consult name space
        return False

    @staticmethod
    def get_nlu_ref_identifier(component: NluComponent) -> str:
        """The tail of a NLU ref after splitting on '.' gives a unique identifier for NON-Aliased components
         If result is '' , model_anno_obj UID will be used as identifier
         """
        tail = component.nlu_ref.split('.')[-1].split('@')[-1]
        if tail == '':
            logger.warning(
                f"Could not deduct tail from component_to_resolve={component}. This is intended for CustomModelComponents used in offline mode")
            tail = str(component.model)
        return tail

    @staticmethod
    def remove_storage_ref_from_features(features: List[str]):
        """Clean storage ref from every str in list """
        return [f.split('@')[0] for f in features]
