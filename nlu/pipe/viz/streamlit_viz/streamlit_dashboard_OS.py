import nlu
from nlu.discovery import Discoverer
from nlu.pipe.utils.storage_ref_utils import StorageRefUtils
from typing import List, Tuple, Optional, Dict, Union
import streamlit as st
from nlu.utils.modelhub.modelhub_utils import ModelHubUtils
import numpy as np
import pandas as pd
from nlu.pipe.viz.streamlit_viz.streamlit_utils_OS import StreamlitUtilsOS
from nlu.pipe.viz.streamlit_viz.gen_streamlit_code import get_code_for_viz
from nlu.pipe.viz.streamlit_viz.styles import _set_block_container_style
import random
from nlu.pipe.viz.streamlit_viz.streamlit_viz_tracker import StreamlitVizTracker
from nlu.pipe.viz.streamlit_viz.viz_building_blocks.dep_tree import DepTreeStreamlitBlock
from nlu.pipe.viz.streamlit_viz.viz_building_blocks.classifier import ClassifierStreamlitBlock
from nlu.pipe.viz.streamlit_viz.viz_building_blocks.token_features import TokenFeaturesStreamlitBlock
from nlu.pipe.viz.streamlit_viz.viz_building_blocks.ner import NERStreamlitBlock
from nlu.pipe.viz.streamlit_viz.viz_building_blocks.word_similarity import WordSimilarityStreamlitBlock
from nlu.pipe.viz.streamlit_viz.viz_building_blocks.word_embedding_manifold import WordEmbeddingManifoldStreamlitBlock



class StreamlitVizBlockHandler():
    """Internal API to access any on the Streamlit building blocks. This is part of the Controller of the MVC pattern"""
    @staticmethod
    def viz_streamlit_dashboard(
            pipe,
            # Base Params
            text:Union[str, List[str], pd.DataFrame, pd.Series],
            model_selection:List[str]=[],
            # NER PARAMS
            # default_ner_model2viz:Union[str, List[str]] = 'en.ner.onto.electra.base',
            # SIMILARITY PARAMS
            similarity_texts:Tuple[str,str]= ('I love NLU <3', 'I love Streamlit <3'),
            title:str = 'NLU ❤️ Streamlit - Prototype your NLP startup in 0 lines of code' ,
            sub_title:str = 'Play with over 1000+ scalable enterprise NLP models',
            side_info:str = None,
            # UI PARAMS
            visualizers:List[str] = ( "dependency_tree", "ner",  "similarity", "token_features", 'classification','manifold'),
            show_models_info:bool = True,
            show_model_select:bool = False,
            show_viz_selection:bool = False,
            show_logo:bool=True,
            set_wide_layout_CSS:bool=True,
            show_code_snippets:bool=False,
            model_select_position:str = 'side' , # main or side
            display_infos:bool=True,
            key:str = "NLU_streamlit",
            display_footer :bool =  True ,
            num_similarity_cols:int=2,

            # NEW PARAMS
            # MANIfold
            num_manifold_cols:int=3,
            manifold_algos:List[str]=('TSNE'),

            # SIMY
            similarity_algos:List[str]=('COSINE'),
    )-> None:
        """Visualize either individual building blocks for streamlit or a full UI to experiment and explore models with"""
        StreamlitVizTracker.footer_displayed = not display_footer
        if set_wide_layout_CSS : _set_block_container_style()
        if title: st.title(title)
        if sub_title: st.subheader(sub_title)
        if show_logo :StreamlitVizTracker.show_logo()
        if side_info : st.sidebar.markdown(side_info)
        text    = st.text_area("Enter text you want to visualize below", text, key=key)
        ner_model_2_viz     = pipe.nlu_ref
        if show_model_select :
            show_code_snippets = st.sidebar.checkbox('Generate code snippets', value=show_code_snippets)
            if model_selection == [] : model_selection = Discoverer.get_components('ner',include_pipes=True)
            model_selection.sort()
            if model_select_position == 'side':
                if pipe.nlu_ref.split(' ')[0] in  model_selection :
                    ner_model_2_viz = st.sidebar.selectbox("Select a NER model.",model_selection,index=model_selection.index(pipe.nlu_ref.split(' ')[0]))
                else :
                    ner_model_2_viz = st.sidebar.selectbox("Select a NER model.",model_selection,index=model_selection.index('en.ner'))
            else :
                if pipe.nlu_ref.split(' ')[0] in  model_selection :
                    ner_model_2_viz = st.selectbox("Select a NER model",model_selection,index=model_selection.index(pipe.nlu_ref.split(' ')[0]))
                else :
                    ner_model_2_viz = st.selectbox("Select a NER model.",index=model_selection.index('en.ner'))

        active_visualizers = visualizers
        if show_viz_selection: active_visualizers = st.sidebar.multiselect("Visualizers",options=visualizers,default=visualizers,key=key)

        all_models = ner_model_2_viz + ' en.dep.typed '  if 'dependency_tree' in active_visualizers  else ner_model_2_viz
        ner_pipe, tree_pipe =  None,None
        if 'ner' in active_visualizers :
            ner_pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
            NERStreamlitBlock.visualize_ner(ner_pipe, text, generate_code_sample=show_code_snippets, key=key, show_model_select=False, show_text_input=True, show_logo=False, show_infos=False)
        if 'dependency_tree' in active_visualizers :
            tree_pipe = StreamlitUtilsOS.get_pipe('en.dep.typed') # if not ValidateVizPipe.viz_tree_satisfied(pipe) else pipe
            DepTreeStreamlitBlock.visualize_dep_tree(tree_pipe, text, generate_code_sample=show_code_snippets, key=key, show_infos=False, show_logo=False)
        if 'token_features' in active_visualizers:
            ner_pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
            TokenFeaturesStreamlitBlock.visualize_tokens_information(ner_pipe, text, generate_code_sample=show_code_snippets, key=key, model_select_position=model_select_position, show_infos=False, show_logo=False, )
        if 'classification' in active_visualizers:
            ner_pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
            ClassifierStreamlitBlock.visualize_classes(ner_pipe, text, generate_code_sample=show_code_snippets, key=key, model_select_position=model_select_position, show_infos=False, show_logo=False)
        if 'similarity' in active_visualizers:
            ner_pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
            WordSimilarityStreamlitBlock.display_word_similarity(ner_pipe, similarity_texts,generate_code_sample=show_code_snippets, model_select_position=model_select_position, show_infos=False,show_logo=False, num_cols=num_similarity_cols)
        if 'manifold' in active_visualizers :
            ner_pipe = pipe if ner_model_2_viz in pipe.nlu_ref.split(' ')  else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
            WordEmbeddingManifoldStreamlitBlock.viz_streamlit_word_embed_manifold(ner_pipe, similarity_texts, generate_code_sample=show_code_snippets, model_select_position=model_select_position, show_infos=False, show_logo=False, num_cols=num_manifold_cols)

        models_to_display_info_for = []
        if ner_pipe  is not None : models_to_display_info_for .append(ner_pipe)
        if tree_pipe is not None : models_to_display_info_for .append(tree_pipe)
        if show_models_info      :StreamlitVizTracker.display_model_info(all_models, models_to_display_info_for)
        if display_infos         : StreamlitVizTracker.display_footer()

    @staticmethod
    def viz_streamlit_word_embed_manifold(
            pipe, # nlu pipe
            default_texts: List[str] = ("Donald Trump likes to party!", "Angela Merkel likes to party!", 'Peter HATES TO PARTTY!!!! :('),
            title: Optional[str] = "Lower dimensional Manifold visualization for word embeddings",
            sub_title: Optional[str] = "Apply any of the 11 `Manifold` or `Matrix Decomposition` algorithms to reduce the dimensionality of `Word Embeddings` to `1-D`, `2-D` and `3-D` ",
            write_raw_pandas : bool = False ,
            default_algos_to_apply : List[str] = ("TSNE", "PCA"),#,'LLE','Spectral Embedding','MDS','ISOMAP','SVD aka LSA','DictionaryLearning','FactorAnalysis','FastICA','KernelPCA',),  # LatentDirichletAllocation 'NMF',
            target_dimensions : List[int] = (1,2,3),
            show_algo_select : bool = True,
            show_embed_select : bool = True,
            show_color_select: bool = True,
            MAX_DISPLAY_NUM:int=100,
            display_embed_information:bool=True,
            set_wide_layout_CSS:bool=True,
            num_cols: int = 3,
            model_select_position:str = 'side', # side or main
            key:str = "NLU_streamlit",
            additional_classifiers_for_coloring:List[str]=['pos', 'sentiment'],
            generate_code_sample:bool = False,
            show_infos:bool = True,
            show_logo:bool = True,
            n_jobs: Optional[int] = 3, # False
    ): WordEmbeddingManifoldStreamlitBlock.viz_streamlit_word_embed_manifold(
        pipe,
        default_texts,
        title,
        sub_title,
        write_raw_pandas,
        default_algos_to_apply,
        target_dimensions,
        show_algo_select,
        show_embed_select,
        show_color_select,
        MAX_DISPLAY_NUM,
        display_embed_information,
        set_wide_layout_CSS,
        num_cols,
        model_select_position,
        key,
        additional_classifiers_for_coloring,
        generate_code_sample,
        show_infos,
        show_logo,
        n_jobs,
    )


    @staticmethod
    def visualize_dep_tree(
            pipe, #nlu pipe
            text:str = 'Billy likes to swim',
            title: Optional[str] = "Dependency Parse & Part-of-speech tags",
            sub_title: Optional[str] = 'POS tags define a `grammatical label` for `each token` and the `Dependency Tree` classifies `Relations between the tokens` ',
            set_wide_layout_CSS:bool=True,
            generate_code_sample:bool = False,
            key = "NLU_streamlit",
            show_infos:bool = True,
            show_logo:bool = True,
            show_text_input:bool = True,
    ): DepTreeStreamlitBlock.visualize_dep_tree(pipe,
                                                text,
                                                title,
                                                sub_title,
                                                set_wide_layout_CSS,
                                                generate_code_sample,
                                                key,show_infos,
                                                show_logo,
                                                show_text_input,)

    @staticmethod
    def display_word_similarity(
            pipe, #nlu pipe
            default_texts: Tuple[str, str] = ("Donald Trump likes to party!", "Angela Merkel likes to party!"),
            threshold: float = 0.5,
            title: Optional[str] = "Embeddings Similarity Matrix &  Visualizations  ",
            sub_tile :Optional[str]="Visualize `word-wise similarity matrix` and calculate `similarity scores` for `2 texts` and every `word embedding` loaded",
            write_raw_pandas : bool = False,
            display_embed_information:bool = True,
            similarity_matrix = True,
            show_algo_select : bool = True,
            dist_metrics:List[str]  =('cosine'),
            set_wide_layout_CSS:bool=True,
            generate_code_sample:bool = False,
            key:str = "NLU_streamlit",
            num_cols:int=2,
            display_scalar_similarities : bool = False ,
            display_similarity_summary:bool = False,
            model_select_position:str = 'side' , # main or side
            show_infos:bool = True,
            show_logo:bool = True,
    ):WordSimilarityStreamlitBlock.display_word_similarity(pipe,
                                                           default_texts,
                                                           threshold,
                                                           title,
                                                           sub_tile,
                                                           write_raw_pandas,
                                                           display_embed_information,
                                                           similarity_matrix,
                                                           show_algo_select,
                                                           dist_metrics,
                                                           set_wide_layout_CSS,
                                                           generate_code_sample,
                                                           key,
                                                           num_cols,
                                                           display_scalar_similarities,
                                                           display_similarity_summary,
                                                           model_select_position,
                                                           show_infos,
                                                           show_logo,)








    @staticmethod
    def visualize_tokens_information(
            pipe, # nlu pipe
            text:str,
            title: Optional[str] = "Token Features",
            sub_title: Optional[str] ='Pick from `over 1000+ models` on the left and `view the generated features`',
            show_feature_select:bool =True,
            features:Optional[List[str]] = None,
            full_metadata: bool = True,
            output_level:str = 'token',
            positions:bool = False,
            set_wide_layout_CSS:bool=True,
            generate_code_sample:bool = False,
            key = "NLU_streamlit",
            show_model_select = True,
            model_select_position:str = 'side' , # main or side
            show_infos:bool = True,
            show_logo:bool = True,
            show_text_input:bool = True,
    ) -> None: TokenFeaturesStreamlitBlock.visualize_tokens_information(
        pipe,
        text,
        title,
        sub_title,
        show_feature_select,
        features,
        full_metadata,
        output_level,
        positions,
        set_wide_layout_CSS,
        generate_code_sample,
        key,
        show_model_select,
        model_select_position,
        show_infos,
        show_logo,
        show_text_input,
    )


    @staticmethod
    def visualize_ner(
            pipe, # Nlu pipe
            text:str,
            ner_tags: Optional[List[str]] = None,
            show_label_select: bool = True,
            show_table: bool = False,
            title: Optional[str] = "Named Entities",
            sub_title: Optional[str] = "Recognize various `Named Entities (NER)` in text entered and filter them. You can select from over `100 languages` in the dropdown.",
            colors: Dict[str, str] = {},
            show_color_selector: bool = False,
            set_wide_layout_CSS:bool=True,
            generate_code_sample:bool = False,
            key = "NLU_streamlit",
            model_select_position:str = 'side',
            show_model_select : bool = True,
            show_text_input:bool = True,
            show_infos:bool = True,
            show_logo:bool = True,

    ): NERStreamlitBlock.visualize_ner(
        pipe,
        text,
        ner_tags,
        show_label_select,
        show_table,
        title,
        sub_title,
        colors,
        show_color_selector,
        set_wide_layout_CSS,
        generate_code_sample,
        key,
        model_select_position,
        show_model_select,
        show_text_input,
        show_infos,
        show_logo,
    )


    @staticmethod
    def visualize_classes(
            pipe, # nlu pipe
            text:Union[str,list,pd.DataFrame, pd.Series, List[str]]=('I love NLU and Streamlit and sunny days!', 'I hate rainy daiys','CALL NOW AND WIN 1000$M'),
            output_level:Optional[str]='document',
            title: Optional[str] = "Text Classification",
            sub_title: Optional[str] = 'View predicted `classes` and `confidences` for `hundreds of text classifiers` in `over 200 languages`',
            metadata : bool = False,
            positions : bool = False,
            set_wide_layout_CSS:bool=True,
            generate_code_sample:bool = False,
            key:str = "NLU_streamlit",
            show_model_selector : bool = True ,
            model_select_position:str = 'side' ,
            show_infos:bool = True,
            show_logo:bool = True,
    )->None: ClassifierStreamlitBlock.visualize_classes(
        pipe,
        text,
        output_level,
        title,
        sub_title,
        metadata,
        positions,
        set_wide_layout_CSS,
        generate_code_sample,
        key,
        show_model_selector,
        model_select_position,
        show_infos,
        show_logo,
    )



