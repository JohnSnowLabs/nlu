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
class WordSimilarityStreamlitBlock():
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
    ):

        """We visualize the following cases :
        1. Simmilarity between 2 words - > sim (word_emb1, word_emb2)
        2. Simmilarity between 2 sentences -> let weTW stand word word_emb of token T and sentence S
            2.1. Raw token level with merged embeddings -> sim([we11,we21,weT1], [we12,we22,weT2])
            2.2  Autogenerate sentemb, basically does 2.1 in the Spark NLP backend
            2.3 Already using sentence_embedder model -> sim(se1,se2)
        3. Simmilarity between token and sentence -> sim([we11,w21,wT1], se2)
        4. Mirrored 3
         """
        # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
        StreamlitVizTracker.footer_displayed=False
        try :
            import plotly.express as px
            from sklearn.metrics.pairwise import distance_metrics
        except :st.error("You need the sklearn and plotly package in your Python environment installed for similarity visualizations. Run <pip install sklearn plotly>")
        if set_wide_layout_CSS : _set_block_container_style()
        if title:st.header(title)
        if show_logo :StreamlitVizTracker.show_logo()
        if sub_tile : st.subheader(sub_tile)

        StreamlitVizTracker.loaded_word_embeding_pipes = []
        dist_metric_algos =distance_metrics()
        dist_algos = list(dist_metric_algos.keys())
        # TODO NORMALIZE DISTANCES TO [0,1] for non cosine
        if 'haversine'   in dist_algos    : dist_algos.remove('haversine') # not applicable in >2D
        if 'precomputed' in dist_algos  : dist_algos.remove('precomputed') # Not a dist
        cols = st.beta_columns(2)
        text1 = cols[0].text_input("Text or word1",default_texts[0],key = key)
        text2 = cols[1].text_input("Text or word2",default_texts[1], key=key) if len(default_texts) >1  else cols[1].text_input("Text or word2",'Please enter second string',key = key)
        # exp = st.sidebar.beta_expander("Select additional Embedding Models and distance metric to compare ")
        e_coms = StreamlitUtilsOS.find_all_embed_components(pipe)
        embed_algos_to_load = []
        embed_pipes = [pipe]
        dist_algo_selection = dist_metrics
        if show_algo_select :
            # emb_components_usable = Discoverer.get_components('embed')
            emb_components_usable = [e for e in Discoverer.get_components('embed',True, include_aliases=True) if 'chunk' not in e and 'sentence' not in e]
            loaded_embed_nlu_refs = []
            loaded_storage_refs = []
            loaded_embed_nlu_refs = list(set(loaded_embed_nlu_refs))

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
            for l in loaded_embed_nlu_refs:
                if l not in emb_components_usable : emb_components_usable.append(l)
            # embed_algo_selection = exp.multiselect("Click to pick additional Embedding Algorithm",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
            # dist_algo_selection = exp.multiselect("Click to pick additional Distance Metric", options=dist_algos, default=dist_metrics, key = key)
            emb_components_usable.sort()
            loaded_embed_nlu_refs.sort()
            dist_algos.sort()
            # dist_metrics.sort()
            if model_select_position =='side':
                embed_algo_selection   = st.sidebar.multiselect("Pick additional Word Embeddings for the Similarity Matrix",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
                dist_algo_selection =  st.sidebar.multiselect("Pick additional Similarity Metrics ", options=dist_algos, default=dist_metrics, key = key)
            else :
                exp = st.beta_expander("Pick additional Word Embeddings and Similarity Metrics")
                embed_algo_selection   = exp.multiselect("Pick additional Word Embeddings for the Similarity Matrix",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
                dist_algo_selection =  exp.multiselect("Pick additional Similarity Metrics ", options=dist_algos, default=dist_metrics, key = key)
            embed_algos_to_load = list(set(embed_algo_selection) - set(loaded_embed_nlu_refs))

        for embedder in embed_algos_to_load:embed_pipes.append(nlu.load(embedder))

        if generate_code_sample:st.code(get_code_for_viz('SIMILARITY',[StreamlitUtilsOS.extract_name(p) for p  in embed_pipes],default_texts))

        StreamlitVizTracker.loaded_word_embeding_pipes+=embed_pipes
        similarity_metrics = {}
        embed_vector_info = {}
        cols_full = True
        col_index=0
        # for p in embed_pipes :
        for p in StreamlitVizTracker.loaded_word_embeding_pipes :
            data1 = p.predict(text1,output_level='token').dropna()
            data2 = p.predict(text2,output_level='token').dropna()
            e_coms = StreamlitUtilsOS.find_all_embed_components(p)
            modelhub_links = [ModelHubUtils.get_url_by_nlu_refrence(c.info.nlu_ref) if hasattr(c.info,'nlu_ref') else ModelHubUtils.get_url_by_nlu_refrence('') for c in e_coms]
            e_cols = StreamlitUtilsOS.get_embed_cols(p)
            for num_emb,e_col in enumerate(e_cols):
                if col_index == num_cols-1 :cols_full=True
                if cols_full :
                    cols = st.beta_columns(num_cols)
                    col_index = 0
                    cols_full = False
                else:col_index+=1
                tok1 = data1['token']
                tok2 = data2['token']
                emb1 = data1[e_col]
                emb2 = data2[e_col]
                embed_mat1 = np.array([x for x in emb1])
                embed_mat2 = np.array([x for x in emb2])
                # e_name = e_col.split('word_embedding_')[-1]
                e_name = e_coms[num_emb].info.nlu_ref if hasattr(e_coms[num_emb].info,'nlu_ref') else e_col.split('word_embedding_')[-1] if 'en.' in e_col else e_col
                e_name = e_name.split('embed.')[-1] if 'en.' in e_name else e_name
                if 'ner' in e_name : e_name = loaded_storage_refs[num_emb]

                embed_vector_info[e_name]= {"Vector Dimension ":embed_mat1.shape[1],
                                            "Num Vectors":embed_mat1.shape[0] + embed_mat1.shape[0],
                                            "NLU_reference":e_coms[num_emb].info.nlu_ref if hasattr(e_coms[num_emb].info,'nlu_ref') else ' ',
                                            "Spark_NLP_reference":ModelHubUtils.NLU_ref_to_NLP_ref(e_coms[num_emb].info.nlu_ref if hasattr(e_coms[num_emb].info,'nlu_ref') else ' '),
                                            "Storage Reference":loaded_storage_refs[num_emb],
                                            'Modelhub info': modelhub_links[num_emb]}
                for dist_algo in dist_algo_selection:
                    # scalar_similarities[e_col][dist_algo]={}
                    sim_score = dist_metric_algos[dist_algo](embed_mat1,embed_mat2)
                    sim_score = pd.DataFrame(sim_score)
                    sim_score.index   = tok1.values
                    sim_score.columns = tok2.values
                    sim_score.columns = StreamlitVizTracker.pad_duplicate_tokens(list(sim_score.columns))
                    sim_score.index   = StreamlitVizTracker.pad_duplicate_tokens(list(sim_score.index))
                    if write_raw_pandas :st.write(sim_score,key = key)
                    if sim_score.shape == (1,1) :
                        sim_score = sim_score.iloc[0][0]
                        sim_score = round(sim_score,2)
                        if sim_score > threshold:
                            st.success(sim_score)
                            st.success(f'Scalar Similarity={sim_score} for distance metric={dist_algo}')
                            st.error('No similarity matrix for only 2 tokens. Try entering at least 1 sentences in a field')
                        else:
                            st.error(f'Scalar Similarity={sim_score} for distance metric={dist_algo}')
                    else :
                        ploty_avaiable = True
                        # for tok emb, sum rows and norm by rows, then sum cols and norm by cols to generate a scalar from matrix
                        scalar_sim_score  = np.sum((np.sum(sim_score,axis=0) / sim_score.shape[0])) / sim_score.shape[1]
                        scalar_sim_score = round(scalar_sim_score,2)

                        if display_scalar_similarities:
                            if scalar_sim_score > threshold:st.success(f'Scalar Similarity :{scalar_sim_score} for distance metric={dist_algo}')
                            else: st.error(f'Scalar Similarity :{scalar_sim_score} for embedder={e_col} distance metric={dist_algo}')
                        if similarity_matrix:
                            if ploty_avaiable :
                                fig = px.imshow(sim_score, labels=dict(color="similarity"))#, title=f'Simmilarity Matrix for embedding_model={e_name} distance metric={dist_algo}')
                                # st.write(fig,key =key)
                                similarity_metrics[f'{e_name}_{dist_algo}_similarity']={
                                    'scalar_similarity' : scalar_sim_score,
                                    'dist_metric' : dist_algo,
                                    'embedding_model': e_name,
                                    'modelhub_info' : modelhub_links[num_emb],
                                }
                                subh = f"""Embedding-Model=`{e_name}`, Similarity-Score=`{scalar_sim_score}`,  distance metric=`{dist_algo}`"""
                                cols[col_index].markdown(subh)
                                cols[col_index].write(fig, key=key)
                            else : pass # todo fallback plots

        if display_similarity_summary:
            exp = st.beta_expander("Similarity summary")
            exp.write(similarity_metrics)
        if display_embed_information:
            exp = st.beta_expander("Embedding vector information")
            exp.write(embed_vector_info)
        if show_infos :
            # VizUtilsStreamlitOS.display_infos()
            StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
            StreamlitVizTracker.display_footer()
