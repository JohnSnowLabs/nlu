import nlu
from nlu.discovery import Discoverer
from nlu.pipe.utils.resolution.storage_ref_utils import StorageRefUtils
from typing import List, Optional
import streamlit as st
import numpy as np
import pandas as pd
from nlu.pipe.viz.streamlit_viz.styles import _set_block_container_style
from nlu.pipe.viz.streamlit_viz.streamlit_viz_tracker import StreamlitVizTracker
from nlu.pipe.viz.streamlit_viz.viz_building_blocks.block_utils.entity_manifold_utils import EntityManifoldUtils


class EntityEmbeddingManifoldStreamlitBlock():
    @staticmethod
    def viz_streamlit_entity_embed_manifold(
            pipe,  # nlu component_list
            default_texts: List[str] = ("Donald Trump likes to visit New York", "Angela Merkel likes to visit Berlin!", 'Peter hates visiting Paris'),
            title: Optional[str] = "Lower dimensional Manifold visualization for Entity embeddings",
            sub_title: Optional[str] = "Apply any of the 10+ `Manifold` or `Matrix Decomposition` algorithms to reduce the dimensionality of `Entity Embeddings` to `1-D`, `2-D` and `3-D` ",
            default_algos_to_apply: List[str] = ("TSNE", "PCA"),
            target_dimensions: List[int] = (1, 2, 3),
            show_algo_select: bool = True,
            set_wide_layout_CSS: bool = True,
            num_cols: int = 3,
            model_select_position: str = 'side',  # side or main
            key: str = "NLU_streamlit",
            show_infos: bool = True,
            show_logo: bool = True,
            n_jobs: Optional[int] = 3,  # False
    ):

        from nlu.pipe.viz.streamlit_viz.streamlit_utils_OS import StreamlitUtilsOS
        StreamlitVizTracker.footer_displayed = False

        try:
            import plotly.express as px
            from sklearn.metrics.pairwise import distance_metrics
        except:
            st.error(
                "You need the sklearn and plotly package in your Python environment installed for similarity visualizations. Run <pip install sklearn plotly>")

        if show_logo: StreamlitVizTracker.show_logo()
        if set_wide_layout_CSS: _set_block_container_style()
        if title: st.header(title)
        if sub_title: st.subheader(sub_title)
        # if show_logo :VizUtilsStreamlitOS.show_logo()
        # VizUtilsStreamlitOS.loaded_word_embeding_pipes = []
        if isinstance(default_texts, list) : default_texts = '\n'.join(default_texts)
        data = st.text_area('Enter N texts, seperated by new lines to visualize Sentence Embeddings for ',
                            default_texts).split('\n')
        output_level = 'chunk'
        ner_emebed_pipe_algo_selection = []
        loaded_ner_embed_nlu_refs = []
        algos = ['TSNE']
        # A component_list should have a NER and a Word Embedding
        if pipe not in StreamlitVizTracker.loaded_ner_word_embeding_pipes: StreamlitVizTracker.loaded_ner_word_embeding_pipes.append(
            pipe)
        if pipe not in StreamlitVizTracker.loaded_word_embeding_pipes: StreamlitVizTracker.loaded_word_embeding_pipes.append(
            pipe)

        if show_algo_select:
            # Manifold Selection
            exp = st.expander("Select additional manifold and dimension reduction techniques to apply")
            algos = exp.multiselect(
                "Reduce embedding dimensionality to something visualizable",
                options=(
                    "TSNE", "ISOMAP", 'LLE', 'Spectral Embedding', 'MDS', 'PCA', 'SVD aka LSA', 'DictionaryLearning',
                    'FactorAnalysis', 'FastICA', 'KernelPCA', 'LatentDirichletAllocation'),
                default=default_algos_to_apply, )
            ner_emb_components_usable = [e for e in Discoverer.get_components('ner', True, include_aliases=True) if
                                         'embed' not in e and 'sentence' not in e]

            # Find nlu_ref of currenlty loaded component_list
            for p in StreamlitVizTracker.loaded_ner_word_embeding_pipes:
                loaded_ner_embed_nlu_refs.append(p.nlu_ref)

            # NER Selection
            if model_select_position == 'side':
                ner_emebed_pipe_algo_selection = st.sidebar.multiselect(
                    "Pick additional NER Models for the Dimension Reduction", options=ner_emb_components_usable,
                    default=loaded_ner_embed_nlu_refs, key=key)
            else:
                ner_emebed_pipe_algo_selection = exp.multiselect(
                    "Pick additional NER Models for the Dimension Reduction", options=ner_emb_components_usable,
                    default=loaded_ner_embed_nlu_refs, key=key)

        for ner_nlu_ref in ner_emebed_pipe_algo_selection:
            load = True
            for ner_p in StreamlitVizTracker.loaded_ner_word_embeding_pipes:
                if ner_p.nlu_ref == ner_nlu_ref:
                    load = False
                    break
            if not load: continue
            p = nlu.load(ner_nlu_ref)
            if p not in StreamlitVizTracker.loaded_ner_word_embeding_pipes: StreamlitVizTracker.loaded_ner_word_embeding_pipes.append(
                p)
            if p not in StreamlitVizTracker.loaded_word_embeding_pipes: StreamlitVizTracker.loaded_word_embeding_pipes.append(
                p)

        col_index = 0
        cols = st.columns(num_cols)

        def are_cols_full():
            return col_index == num_cols

        for p in StreamlitVizTracker.loaded_ner_word_embeding_pipes:
            p = EntityManifoldUtils.insert_chunk_embedder_to_pipe_if_missing(p)
            predictions = p.predict(data, metadata=True, output_level=output_level, multithread=False).dropna()
            entity_cols = EntityManifoldUtils.get_ner_cols(predictions)
            chunk_embed_col = EntityManifoldUtils.find_chunk_embed_col(predictions)

            # TODO get cols for non default NER? or multi ner setups?
            # features = predictions[EntityManifoldUtils.get_ner_cols(predictions)]
            # e_col = StreamlitUtilsOS.find_embed_col(predictions)
            e_com = StreamlitUtilsOS.find_embed_component(p)
            e_com_storage_ref = StorageRefUtils.extract_storage_ref(e_com)
            emb = predictions[chunk_embed_col]
            mat = np.array([x for x in emb])
            # for ner_emb_p in ps:
            for algo in algos:
                # Only pos values for latent Dirchlet
                if algo == 'LatentDirichletAllocation': mat = np.square(mat)
                if len(mat.shape) > 2: mat = mat.reshape(len(emb), mat.shape[-1])
                hover_data = entity_cols + ['text']
                # calc reduced dimensionality with every algo
                feature_to_color_by = entity_cols[0]
                if 1 in target_dimensions:
                    low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo, 1, n_jobs).fit_transform(mat)
                    x = low_dim_data[:, 0]
                    y = np.zeros(low_dim_data[:, 0].shape)

                    # predictions['text'] = original_text
                    tsne_df = pd.DataFrame({**{'x': x, 'y': y},
                                            **{k: predictions[k] for k in entity_cols},
                                            **{'text': predictions[entity_cols[-1]]}
                                            })
                    fig = px.scatter(tsne_df, x="x", y="y", color=feature_to_color_by, hover_data=hover_data)
                    subh = f"""Word-Embeddings =`{e_com_storage_ref}`, NER-Model =`{p.nlu_ref}`, Manifold-Algo =`{algo}` for `D=1`"""
                    cols[col_index].markdown(subh)
                    cols[col_index].write(fig, key=key)
                    col_index += 1
                    if are_cols_full():
                        cols = st.columns(num_cols)
                        col_index = 0
                if 2 in target_dimensions:
                    low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo, 2, n_jobs).fit_transform(mat)
                    x = low_dim_data[:, 0]
                    y = low_dim_data[:, 1]
                    tsne_df = pd.DataFrame({**{'x': x, 'y': y},
                                            **{k: predictions[k] for k in entity_cols},
                                            **{'text': predictions[entity_cols[-1]]}
                                            })
                    fig = px.scatter(tsne_df, x="x", y="y", color=feature_to_color_by, hover_data=hover_data)
                    subh = f"""Word-Embeddings =`{e_com_storage_ref}`, NER-Model =`{p.nlu_ref}`, Manifold-Algo =`{algo}` for `D=2`"""
                    cols[col_index].markdown(subh)
                    cols[col_index].write(fig, key=key)
                    col_index += 1
                    if are_cols_full():
                        cols = st.columns(num_cols)
                        col_index = 0
                if 3 in target_dimensions:
                    low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo, 3, n_jobs).fit_transform(mat)
                    x = low_dim_data[:, 0]
                    y = low_dim_data[:, 1]
                    z = low_dim_data[:, 2]
                    tsne_df = pd.DataFrame({**{'x': x, 'y': y, 'z': z},
                                            **{k: predictions[k] for k in entity_cols},
                                            **{'text': predictions[entity_cols[-1]]}
                                            })
                    fig = px.scatter_3d(tsne_df, x="x", y="y", z='z', color=feature_to_color_by, hover_data=hover_data)
                    subh = f"""Word-Embeddings =`{e_com_storage_ref}`, NER-Model =`{p.nlu_ref}`, Manifold-Algo =`{algo}` for `D=3`"""
                    cols[col_index].markdown(subh)
                    cols[col_index].write(fig, key=key)
                    col_index += 1
                    if are_cols_full():
                        cols = st.columns(num_cols)
                        col_index = 0

                # Todo fancy embed infos etc
                # if display_embed_information: display_embed_vetor_information(e_com,mat)

            # if display_embed_information:
            #     exp = st.expander("Embedding vector information")
            #     exp.write(embed_vector_info)

        if show_infos:
            # VizUtilsStreamlitOS.display_infos()
            StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes=[pipe])
            StreamlitVizTracker.display_footer()
