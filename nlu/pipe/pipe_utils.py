import inspect
from nlu.pipe.pipe_components import SparkNLUComponent

class PipeUtils():
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
    def clean_irrelevant_features(component_list, remove_AT_notation = False ):
        '''
        Remove irrelevant features from a list of component features
        Also remove the @notation from names, since they are irrelevant for ordering
        :param component_list: list of features
        :param remove_AT_notation: remove AT notation from c names if true. Used for sorting
        :return: list with only relevant feature names
        '''
        # remove irrelevant missing features for pretrained models
        if 'text' in component_list: component_list.remove('text')
        if 'raw_text' in component_list: component_list.remove('raw_text')
        if 'raw_texts' in component_list: component_list.remove('raw_texts')
        if 'label' in component_list: component_list.remove('label')
        if 'sentiment_label' in component_list: component_list.remove('sentiment_label')
        if remove_AT_notation:
            new_cs = []
            for c in component_list :new_cs.append(c.split("@")[0])
            return new_cs
        return component_list

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

