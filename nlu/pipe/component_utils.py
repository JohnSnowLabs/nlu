import inspect
from nlu.pipe.pipe_components import SparkNLUComponent

"""Component and Column Level logic operations and utils"""
class ComponentUtils():
    @staticmethod
    def config_chunk_embed_converter(converter:SparkNLUComponent):
        '''For a Chunk to be added to a pipeline, configure its input/output and set storage ref to amtch the storage ref'''
        storage_ref = converter.info.outputs[0].split('@')[0]
        converter.info.storage_ref = storage_ref


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
        if 'text' in feature_list: feature_list.remove('text')
        if 'raw_text' in feature_list: feature_list.remove('raw_text')
        if 'raw_texts' in feature_list: feature_list.remove('raw_texts')
        if 'label' in feature_list: feature_list.remove('label')
        if 'sentiment_label' in feature_list: feature_list.remove('sentiment_label')
        if remove_AT_notation:
            new_cs = []
            for c in feature_list :new_cs.append(c.split("@")[0])
            return new_cs
        return feature_list


    @staticmethod
    def component_has_embeddings_requirement(component):
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
    def component_has_embeddings_provisions(component):
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
    def extract_storage_ref_AT_column(component, col='input'):
        '''
        Extract <col>_embed_col@storage_ref notation from a component if it has a storage ref, otherwise '

        :param component:  To extract notation from
        :cols component:  Wether to extract for the input or output col
        :return: '' if no storage_ref, <col>_embed_col@storage_ref otherwise
        '''
        if not PipeUtils.has_storage_ref(component) :
            if   col =='input' : return component.info.inputs
            elif col =='output': return component.info.outputs
        if   col =='input'  : e_col    = next(filter(lambda s : 'embed' in s, component.info.inputs))
        elif col =='output' : e_col    = next(filter(lambda s : 'embed' in s, component.info.outputs))

        stor_ref = PipeUtils.extract_storage_ref_from_component(component)
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
    def is_embedding_converter  (component : SparkNLUComponent):
        """Check if NLU component is embedding converter """
        if component.info.name in ['chunk_embedding_converter', 'sentence_embedding_converter']: return True
        return False
