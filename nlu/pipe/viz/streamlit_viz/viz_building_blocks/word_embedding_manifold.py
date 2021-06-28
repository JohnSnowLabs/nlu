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
class WordEmbeddingManifoldStreamlitBlock():
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
            additional_classifiers_for_coloring:List[str]=['pos', 'sentiment.imdb'],
            generate_code_sample:bool = False,
            show_infos:bool = True,
            show_logo:bool = True,
            n_jobs: Optional[int] = 3, # False
    ):
        from nlu.pipe.viz.streamlit_viz.streamlit_utils_OS import StreamlitUtilsOS
        StreamlitVizTracker.footer_displayed=False
        try :
            import plotly.express as px
            from sklearn.metrics.pairwise import distance_metrics
        except :st.error("You need the sklearn and plotly package in your Python environment installed for similarity visualizations. Run <pip install sklearn plotly>")
        if len(default_texts) > MAX_DISPLAY_NUM : default_texts = default_texts[:MAX_DISPLAY_NUM]
        if show_logo :StreamlitVizTracker.show_logo()
        if set_wide_layout_CSS : _set_block_container_style()
        if title:st.header(title)
        if sub_title:st.subheader(sub_title)
        # if show_logo :VizUtilsStreamlitOS.show_logo()

        # VizUtilsStreamlitOS.loaded_word_embeding_pipes = []


        data = st.text_area('Enter N texts, seperated by new lines to visualize Word Embeddings for ','\n'.join(default_texts))
        if len(data) > MAX_DISPLAY_NUM : data = data[:MAX_DISPLAY_NUM]
        data_split = data.split("\n")
        while '' in data_split : data_split.remove('')
        data       = data_split.copy()
        while '' in data : data.remove('')

        if len(data)<=1:
            st.error("Please enter more than 2 lines of text, seperated by new lines (hit <ENTER>)")
            return
        # TODO dynamic color inference for plotting
        if show_color_select:
            if model_select_position == 'side' : feature_to_color_by =  st.sidebar.selectbox('Pick a feature to color points in manifold by ',['pos','sentiment',],0)
            else:feature_to_color_by =  st.selectbox('Feature to color plots by ',['pos','sentiment',],0)
        text_col = 'token'
        embed_algos_to_load = []
        new_embed_pipes = []
        e_coms = StreamlitUtilsOS.find_all_embed_components(pipe)

        if show_algo_select :
            exp = st.beta_expander("Select additional manifold and dimension reduction techniques to apply")

            algos = exp.multiselect(
                "Reduce embedding dimensionality to something visualizable",
                options=("TSNE", "ISOMAP",'LLE','Spectral Embedding','MDS','PCA','SVD aka LSA','DictionaryLearning','FactorAnalysis','FastICA','KernelPCA',),default=default_algos_to_apply,)

            emb_components_usable = [e for e in Discoverer.get_components('embed',True, include_aliases=True) if 'chunk' not in e and 'sentence' not in e]
            loaded_embed_nlu_refs = []
            loaded_classifier_nlu_refs = []
            loaded_storage_refs = []
            for c in e_coms :
                if not  hasattr(c.info,'nlu_ref'): continue
                r = c.info.nlu_ref
                if 'en.' not in r and 'embed.' not  in r and 'ner' not in r : loaded_embed_nlu_refs.append('en.embed.' + r)
                elif 'en.'  in r and 'embed.' not  in r  and 'ner' not in r:
                    r = r.split('en.')[0]
                    loaded_embed_nlu_refs.append('en.embed.' + r)
                else :
                    loaded_embed_nlu_refs.append(StorageRefUtils.extract_storage_ref(c))
                loaded_storage_refs.append(StorageRefUtils.extract_storage_ref(c))

            for p in StreamlitVizTracker.loaded_word_embeding_pipes :
                if p != pipe : loaded_embed_nlu_refs.append(p.nlu_ref)
            loaded_embed_nlu_refs = list(set(loaded_embed_nlu_refs))

            for l in loaded_embed_nlu_refs:
                if l not in emb_components_usable : emb_components_usable.append(l)
            emb_components_usable.sort()
            loaded_embed_nlu_refs.sort()
            if model_select_position =='side':
                embed_algo_selection   = st.sidebar.multiselect("Pick additional Word Embeddings for the Dimension Reduction",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
                embed_algo_selection=[embed_algo_selection[-1]]
            else :
                exp = st.beta_expander("Pick additional Word Embeddings")
                embed_algo_selection   = exp.multiselect("Pick additional Word Embeddings for the Dimension Reduction",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
                embed_algo_selection=[embed_algo_selection[-1]]
            embed_algos_to_load = list(set(embed_algo_selection) - set(loaded_embed_nlu_refs))
        for embedder in embed_algos_to_load:new_embed_pipes.append(nlu.load(embedder))# + f' {" ".join(additional_classifiers_for_coloring)}'))
        StreamlitVizTracker.loaded_word_embeding_pipes+=new_embed_pipes
        if pipe not in StreamlitVizTracker.loaded_word_embeding_pipes: StreamlitVizTracker.loaded_word_embeding_pipes.append(pipe)
        for nlu_ref in additional_classifiers_for_coloring :
            already_loaded=False
            if 'pos' in nlu_ref : continue
            # for p in  VizUtilsStreamlitOS.loaded_document_classifier_pipes:
            #     if p.nlu_ref == nlu_ref : already_loaded = True
            # if not already_loaded : VizUtilsStreamlitOS.loaded_token_level_classifiers.append(nlu.load(nlu_ref))
            else :
                for p in  StreamlitVizTracker.loaded_document_classifier_pipes:
                    if p.nlu_ref == nlu_ref : already_loaded = True
                if not already_loaded :
                    already_loaded=True
                    StreamlitVizTracker.loaded_document_classifier_pipes.append(nlu.load(nlu_ref))

        col_index = 0
        cols = st.beta_columns(num_cols)
        def are_cols_full(): return col_index == num_cols
        token_feature_pipe = StreamlitUtilsOS.get_pipe('pos')
        #not all pipes have sentiment/pos etc.. models for hueing loaded....
        ## Lets FIRST predict with the classifiers/Token level feature generators and THEN apply embed pipe
        for p in StreamlitVizTracker.loaded_word_embeding_pipes :
            data = data_split.copy()

            classifier_cols = []

            for class_p in StreamlitVizTracker.loaded_document_classifier_pipes:

                data = class_p.predict(data, output_level='document',multithread=False).dropna()
                classifier_cols.append(StreamlitUtilsOS.get_classifier_cols(class_p))
                data['text'] = data_split
                # drop embeds of classifiers because bad conversion
                for c in data.columns :
                    if 'embedding' in c : data.drop(c, inplace=True,axis=1)
            # data['text']
            # =data['document']
            data['text'] = data_split
            for c in data.columns :
                if 'sentence_embedding' in c : data.drop(c,inplace=True,axis=1)

            p = StreamlitUtilsOS.merge_token_classifiers_with_embed_pipe(p, token_feature_pipe)
            # if'token' in data.columns : data.drop(['token','pos'],inplace=True,)
            if'pos' in data.columns : data.drop('pos',inplace=True,axis=1)
            predictions =   p.predict(data,output_level='token',multithread=False).dropna()
            e_col = StreamlitUtilsOS.find_embed_col(predictions)
            e_com = StreamlitUtilsOS.find_embed_component(p)
            e_com_storage_ref = StorageRefUtils.extract_storage_ref(e_com, True)
            # embedder_name = StreamlitUtilsOS.extract_name(e_com)
            emb = predictions[e_col]
            mat = np.array([x for x in emb])
            for algo in algos :
                if len(mat.shape)>2 : mat = mat.reshape(len(emb),mat.shape[-1])
                hover_data = ['token','text','sentiment', 'pos']  # TODO DEDUCT
                # calc reduced dimensionality with every algo
                if 1 in target_dimensions:
                    low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo,1,n_jobs).fit_transform(mat)
                    x = low_dim_data[:,0]
                    y = np.zeros(low_dim_data[:,0].shape)

                    tsne_df =  pd.DataFrame({'x':x,'y':y, 'text':predictions[text_col], 'pos':predictions.pos, 'sentiment' : predictions.sentiment,'token':predictions.token})
                    fig = px.scatter(tsne_df, x="x", y="y",color=feature_to_color_by, hover_data=hover_data)
                    subh = f"""Word-Embeddings =`{e_com_storage_ref}`, Manifold-Algo =`{algo}` for `D=1`"""
                    cols[col_index].markdown(subh)
                    cols[col_index].write(fig,key=key)
                    col_index+=1
                    if are_cols_full() :
                        cols = st.beta_columns(num_cols)
                        col_index = 0
                if 2 in target_dimensions:
                    low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo,2,n_jobs).fit_transform(mat)
                    x = low_dim_data[:,0]
                    y = low_dim_data[:,1]
                    tsne_df =  pd.DataFrame({'x':x,'y':y, 'text':predictions[text_col], 'pos':predictions.pos, 'sentiment' : predictions.sentiment,'token':predictions.token})

                    fig = px.scatter(tsne_df, x="x", y="y",color=feature_to_color_by, hover_data=hover_data)
                    subh = f"""Word-Embeddings =`{e_com_storage_ref}`, Manifold-Algo =`{algo}` for `D=2`"""
                    cols[col_index].markdown(subh)
                    cols[col_index].write(fig,key=key)
                    # st.write(fig)
                    col_index+=1
                    if are_cols_full() :
                        cols = st.beta_columns(num_cols)
                        col_index = 0
                if 3 in target_dimensions:
                    low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo,3,n_jobs).fit_transform(mat)
                    x = low_dim_data[:,0]
                    y = low_dim_data[:,1]
                    z = low_dim_data[:,2]
                    tsne_df =  pd.DataFrame({'x':x,'y':y,'z':z, 'text':predictions[text_col], 'pos':predictions.pos, 'sentiment':predictions.sentiment,'token':predictions.token })

                    fig = px.scatter_3d(tsne_df, x="x", y="y", z='z',color=feature_to_color_by, hover_data=hover_data)
                    subh = f"""Word-Embeddings =`{e_com_storage_ref}`, Manifold-Algo =`{algo}` for `D=3`"""
                    cols[col_index].markdown(subh)
                    cols[col_index].write(fig,key=key)

                    # st.write(fig)
                    col_index+=1
                    if are_cols_full() :
                        cols = st.beta_columns(num_cols)
                        col_index = 0
            # Todo fancy embed infos etc
            # if display_embed_information: display_embed_vetor_information(e_com,mat)

        # if display_embed_information:
        #     exp = st.beta_expander("Embedding vector information")
        #     exp.write(embed_vector_info)
        if show_infos :
            # VizUtilsStreamlitOS.display_infos()
            StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
            StreamlitVizTracker.display_footer()