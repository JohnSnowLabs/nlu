import streamlit as st
from sparknlp.annotator import *
from nlu.universe.feature_node_ids import NLP_NODE_IDS, NLP_HC_NODE_IDS
from nlu.universe.logic_universes import AnnoTypes
from nlu.universe.component_universes import ComponentUniverse, jsl_id_to_empty_component
from nlu.universe.universes import Licenses


class EntityManifoldUtils():
    classifers_OS = [ClassifierDLModel, LanguageDetectorDL, MultiClassifierDLModel, NerDLModel, NerCrfModel,
                     YakeKeywordExtraction, PerceptronModel, SentimentDLModel,
                     SentimentDetectorModel, ViveknSentimentModel, DependencyParserModel, TypedDependencyParserModel,
                     T5Transformer, MarianTransformer, NerConverter]

    @staticmethod
    def insert_chunk_embedder_to_pipe_if_missing(pipe):

        """Scan component_list for chunk_embeddings. If missing, add new. Validate NER model_anno_obj is loaded"""
        # component_list.predict('Donald Trump and Angela Merkel love Berlin')

        classifier_cols = []
        has_ner = False
        has_chunk_embeds = True
        ner_component_names = ['named_entity_recognizer_dl', 'named_entity_recognizer_dl_healthcare']
        for c in pipe.components:
            if c.name == NLP_NODE_IDS.NER_DL or c.name == NLP_HC_NODE_IDS.MEDICAL_NER or c.type == AnnoTypes.TRANSFORMER_TOKEN_CLASSIFIER:
                has_ner = True
            if c.name == NLP_NODE_IDS.CHUNK_EMBEDDINGS_CONVERTER:
                return pipe
        if not has_ner:
            raise ValueError(
                "You Need to load a NER model_anno_obj or this visualization. Try nlu.load('ner').viz_streamlit_entity_embed_manifold(text)")

        ner_conveter_c, word_embed_c = None, None

        for c in pipe.components:
            if c.type == AnnoTypes.TOKEN_EMBEDDING:
                word_embed_c = c
            if c.name == NLP_NODE_IDS.NER_CONVERTER:
                ner_conveter_c = c
            if c.name == NLP_HC_NODE_IDS.NER_CONVERTER_INTERNAL:
                ner_conveter_c = c

        chunker = jsl_id_to_empty_component(NLP_NODE_IDS.CHUNK_EMBEDDINGS_CONVERTER)
        chunker.set_metadata(
            chunker.get_default_model(),
            'chunker', 'chunker', 'xx', False, Licenses.open_source)

        # chunker = embeddings_chunker.EmbeddingsChunker(nlu_ref='chunk_embeddings')

        chunker.model.setInputCols(ner_conveter_c.spark_output_column_names + word_embed_c.spark_output_column_names)
        chunker.model.setOutputCol('chunk_embedding')
        chunker.spark_input_column_names = ner_conveter_c.spark_output_column_names + word_embed_c.spark_output_column_names
        chunker.spark_output_column_names = ['chunk_embedding']
        # chunker.inputs = ner_conveter_c.spark_output_column_names + word_embed_c.spark_output_column_names
        # chunker.out_types = ['chunk_embedding']

        pipe.components.append(chunker)
        pipe.is_fitted = False
        pipe.fit()

        return pipe

    @staticmethod
    def find_chunk_embed_col(df):
        return 'chunk_embedding_chunker'
        for c in df.columns:
            ss = set(c.split('_'))
            if 'chunk' in ss: ss.remove('chunk')
            if 'embedding' in ss: ss.remove('embedding')
            if 'embeddings' in ss: ss.remove('embeddings')
            if len(ss) == 0: return c
        raise ValueError('Could not find chunk embed col')

    @staticmethod
    def get_ner_cols(df):
        """find NER pred, conf and class cols.
        Cols[0] =  entity_class_col
        Cols[1] =  entity_confidence
        Cols[2] =  entity_chunk

        """

        entity_class_col, entity_confidence_col, entity_chunk_col = None, None, None

        for c in df.columns:
            if 'entities' in c and 'class' in c:
                entity_class_col = c
            if 'entities' in c and 'confidence' in c:
                entity_confidence_col = c
            if 'entities' in c and 'confidence' not in c and 'confidence' not in c and 'origin' not in c:
                entity_chunk_col = c
        cols = [entity_class_col, entity_confidence_col, entity_chunk_col]
        if not any(cols):
            raise ValueError(f'Failure to resolve entities col for df cols = {df.columns}')
        return cols

    @staticmethod
    def find_entity_embed_col_pd(df, search_multi=False):
        """Find col that contains embed in pandas df """
        if not search_multi:
            for c in df.columns:
                if 'embed_entitiy' in c: return c
        else:
            e_cols = []
            for c in df.columns:
                if 'embed' in c: e_cols.append(c)
        return e_cols

    @staticmethod
    def find_embed_component(p):
        """Find first embed  component_to_resolve in component_list"""
        for c in p.components:
            if 'embed' in c.out_types[0]: return c
        st.warning("No Embed model_anno_obj in component_list")
        return None
