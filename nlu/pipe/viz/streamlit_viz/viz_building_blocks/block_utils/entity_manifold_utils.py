import streamlit as st
from sparknlp.annotator import *
from nlu.components import embeddings_chunker

class EntityManifoldUtils():
    classifers_OS = [ ClassifierDLModel, LanguageDetectorDL, MultiClassifierDLModel, NerDLModel, NerCrfModel, YakeKeywordExtraction, PerceptronModel, SentimentDLModel,
                      SentimentDetectorModel, ViveknSentimentModel, DependencyParserModel, TypedDependencyParserModel, T5Transformer, MarianTransformer, NerConverter]
    @staticmethod
    def insert_chunk_embedder_to_pipe_if_missing(pipe):

        """Scan component_list for chunk_embeddings. If missing, add new. Validate NER model is loaded"""
        # component_list.predict('Donald Trump and Angela Merkel love Berlin')

        classifier_cols = []
        has_ner = False
        has_chunk_embeds = True
        ner_component_names = ['named_entity_recognizer_dl', 'named_entity_recognizer_dl_healthcare']
        for c in pipe.components:
            if c.info.name in ner_component_names : has_ner = True
            if c.info.name == 'chunk_embedding_converter' : return pipe
        if not has_ner : ValueError("You Need to load a NER model or this visualization. Try nlu.load('ner').viz_streamlit_entity_embed_manifold(text)")

        ner_conveter_c, word_embed_c = None,None

        for c in pipe.components:
            if c.info.type =='word_embeddings' : word_embed_c = c
            if c.info.type =='word_embeddings' : word_embed_c = c
            if c.info.type == 'ner_to_chunk_converter' : ner_conveter_c = c
            if c.info.type == 'ner_to_chunk_converter_licensed' : ner_conveter_c = c


        chunker = embeddings_chunker.EmbeddingsChunker(nlu_ref='chunk_embeddings')
        chunker.model.setInputCols(ner_conveter_c.info.spark_output_column_names + word_embed_c.info.spark_output_column_names )
        chunker.model.setOutputCol('chunk_embedding')
        chunker.info.spark_input_column_names  = ner_conveter_c.info.spark_output_column_names + word_embed_c.info.spark_output_column_names
        chunker.info.spark_output_column_names = ['chunk_embedding']
        chunker.info.inputs  = ner_conveter_c.info.spark_output_column_names + word_embed_c.info.spark_output_column_names
        chunker.out_types = ['chunk_embedding']

        pipe.components.append(chunker)
        pipe.is_fitted=False
        pipe.fit()



        return pipe

    @staticmethod
    def find_chunk_embed_col(df):
        for c in df.columns:
            ss = set(c.split('_'))
            if 'chunk' in ss : ss.remove('chunk')
            if 'embedding' in ss : ss.remove('embedding')
            if 'embeddings' in ss : ss.remove('embeddings')
            if len(ss) == 0: return c
        ValueError('Could not find chunk embed col')
    @staticmethod
    def get_ner_cols(df):
        """find NER pred, conf and class cols"""
        return ['entities_class','entities','entities_confidence']



    @staticmethod
    def find_entity_embed_col_pd(df, search_multi=False):
        """Find col that contains embed in pandas df """
        if not search_multi:
            for c in df.columns:
                if 'embed_entitiy'in c : return c
        else:
            e_cols =[]
            for c in df.columns:
                if 'embed'in c : e_cols.append(c)
        return e_cols


    @staticmethod
    def find_embed_component(p):
        """Find first embed  component in component_list"""
        for c in p.components :
            if 'embed' in c.out_types[0] : return c
        st.warning("No Embed model in component_list")
        return None

