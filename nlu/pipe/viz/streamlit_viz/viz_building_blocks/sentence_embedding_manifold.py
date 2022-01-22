import nlu
from nlu.discovery import Discoverer
from nlu.pipe.utils.resolution.storage_ref_utils import StorageRefUtils
from typing import List, Optional
import streamlit as st
import numpy as np
import pandas as pd
from nlu.pipe.viz.streamlit_viz.styles import _set_block_container_style
from nlu.pipe.viz.streamlit_viz.streamlit_viz_tracker import StreamlitVizTracker


class SentenceEmbeddingManifoldStreamlitBlock():
    @staticmethod
    def viz_streamlit_sentence_embed_manifold(
            pipe,  # nlu component_list
            default_texts: List[str] = (
            "Donald Trump likes to party!", "Angela Merkel likes to party!", 'Peter HATES TO PARTTY!!!! :('),
            title: Optional[str] = "Lower dimensional Manifold visualization for Sentence embeddings",
            sub_title: Optional[
                str] = "Apply any of the 11 `Manifold` or `Matrix Decomposition` algorithms to reduce the dimensionality of `Sentence Embeddings` to `1-D`, `2-D` and `3-D` ",
            write_raw_pandas: bool = False,
            default_algos_to_apply: List[str] = ("TSNE", "PCA"),
            # ,'LLE','Spectral Embedding','MDS','ISOMAP','SVD aka LSA','DictionaryLearning','FactorAnalysis','FastICA','KernelPCA',),  # LatentDirichletAllocation 'NMF',
            target_dimensions: List[int] = (1, 2, 3),
            show_algo_select: bool = True,
            show_embed_select: bool = True,
            show_color_select: bool = True,
            MAX_DISPLAY_NUM: int = 200000,
            display_embed_information: bool = True,
            set_wide_layout_CSS: bool = True,
            num_cols: int = 3,
            model_select_position: str = 'side',  # side or main
            key: str = "NLU_streamlit",
            additional_classifiers_for_coloring: List[str] = ['sentiment.imdb'],
            generate_code_sample: bool = False,
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
        # if len(default_texts) > MAX_DISPLAY_NUM : default_texts = default_texts[:MAX_DISPLAY_NUM]
        if show_logo: StreamlitVizTracker.show_logo()
        if set_wide_layout_CSS: _set_block_container_style()
        if title: st.header(title)
        if sub_title: st.subheader(sub_title)
        # if show_logo :VizUtilsStreamlitOS.show_logo()

        # VizUtilsStreamlitOS.loaded_word_embeding_pipes = []

        data = st.text_area('Enter N texts, seperated by new lines to visualize Sentence Embeddings for ',
                            default_texts)
        # detect_sentence = False # TODO ITNEGRATE PARAM
        output_level = 'document'  # if not detect_sentence else 'sentence'
        classifier_cols = []
        original_text = nlu.load('tokenize').predict(data.split("\n"), output_level=output_level)[output_level].values
        original_text = original_text
        original_text = original_text[original_text != '']
        original_text = original_text[~pd.isna(original_text)]

        text_col = output_level
        embed_algos_to_load = []
        class_algos_to_load = []
        new_embed_pipes = []
        new_class_pipes = []
        e_coms = StreamlitUtilsOS.find_all_embed_components(pipe)

        if show_algo_select:
            exp = st.expander("Select additional manifold and dimension reduction techniques to apply")

            algos = exp.multiselect(
                "Reduce embedding dimensionality to something visualizable",
                options=(
                "TSNE", "ISOMAP", 'LLE', 'Spectral Embedding', 'MDS', 'PCA', 'SVD aka LSA', 'DictionaryLearning',
                'FactorAnalysis', 'FastICA', 'KernelPCA', 'LatentDirichletAllocation'),
                default=default_algos_to_apply, )

            emb_components_usable = [e for e in Discoverer.get_components('embed', True, include_aliases=True) if
                                     'chunk' not in e and 'sentence' in e]
            # Todo, multi-classifiers excluded
            classifier_components_usable = [e for e in Discoverer.get_components('classify', True, include_aliases=True)
                                            if 'xx' not in e and 'toxic' not in e and 'e2e' not in e]
            # Storage Ref extraction
            loaded_embed_nlu_refs, loaded_storage_refs = StreamlitUtilsOS.extract_all_sentence_storage_refs_or_nlu_refs(
                e_coms)
            loaded_classifier_nlu_refs = additional_classifiers_for_coloring  # + all classifier NLU_refs?

            # Get loaded Embed NLU Refs
            for embed_pipe in StreamlitVizTracker.loaded_sentence_embeding_pipes:
                if embed_pipe != pipe: loaded_embed_nlu_refs.append(embed_pipe.nlu_ref)
            loaded_embed_nlu_refs = list(set(loaded_embed_nlu_refs))

            # Get loaded Classifier NLU Refs
            for embed_pipe in StreamlitVizTracker.loaded_document_classifier_pipes:
                if embed_pipe != pipe: loaded_classifier_nlu_refs.append(embed_pipe.nlu_ref)
            loaded_classifier_nlu_refs = list(set(loaded_classifier_nlu_refs))

            # fix default selector
            for l in loaded_embed_nlu_refs:
                if l not in emb_components_usable: emb_components_usable.append(l)

            # fix default selector
            for l in loaded_classifier_nlu_refs:
                if l not in classifier_components_usable: classifier_components_usable.append(l)

            emb_components_usable.sort()
            loaded_embed_nlu_refs.sort()
            classifier_components_usable.sort()
            loaded_classifier_nlu_refs.sort()
            if model_select_position == 'side':
                embed_algo_selection = st.sidebar.multiselect(
                    "Pick additional Sentence Embeddings for the Dimension Reduction", options=emb_components_usable,
                    default=loaded_embed_nlu_refs, key=key)
                embed_algo_selection = [embed_algo_selection[-1]]

                exp = st.expander("Pick additional Classifiers")
                class_algo_selection = exp.multiselect("Pick additional Classifiers to load for coloring points",
                                                       options=classifier_components_usable,
                                                       default=loaded_classifier_nlu_refs, key=key)
                class_algo_selection = [class_algo_selection[-1]]

            else:
                exp = st.expander("Pick additional Sentence Embeddings")
                embed_algo_selection = exp.multiselect(
                    "Pick additional Sentence Embeddings for the Dimension Reduction", options=emb_components_usable,
                    default=loaded_embed_nlu_refs, key=key)
                embed_algo_selection = [embed_algo_selection[-1]]

                exp = st.expander("Pick additional Classifiers")
                class_algo_selection = exp.multiselect("Pick additional Classifiers to load for coloring points",
                                                       options=classifier_components_usable,
                                                       default=loaded_classifier_nlu_refs, key=key)
                class_algo_selection = [class_algo_selection[-1]]

            embed_algos_to_load = list(set(embed_algo_selection) - set(loaded_embed_nlu_refs))
            class_algos_to_load = list(set(class_algo_selection) - set(loaded_classifier_nlu_refs))

        for embedder in embed_algos_to_load: new_embed_pipes.append(nlu.load(embedder))
        for classifier in class_algos_to_load: new_class_pipes.append(nlu.load(classifier))

        StreamlitVizTracker.loaded_sentence_embeding_pipes += new_embed_pipes
        StreamlitVizTracker.loaded_document_classifier_pipes += new_class_pipes
        if pipe not in StreamlitVizTracker.loaded_sentence_embeding_pipes: StreamlitVizTracker.loaded_sentence_embeding_pipes.append(
            pipe)

        for nlu_ref in additional_classifiers_for_coloring:  # TODO REMVOVE< INTEGRATE INTO THE AUT LOAD THING REDUNDAND
            already_loaded = False
            for embed_pipe in StreamlitVizTracker.loaded_document_classifier_pipes:
                if embed_pipe.nlu_ref == nlu_ref: already_loaded = True
            if not already_loaded:
                already_loaded = True
                StreamlitVizTracker.loaded_document_classifier_pipes.append(nlu.load(nlu_ref))

        col_index = 0
        cols = st.columns(num_cols)

        data = original_text.copy()
        # Get classifier predictions
        classifier_cols = []
        for class_pipe in StreamlitVizTracker.loaded_document_classifier_pipes:
            data = class_pipe.predict(data, output_level=output_level, multithread=False)
            classifier_cols += StreamlitUtilsOS.get_classifier_cols(class_pipe)
            data['text'] = original_text
            # drop embeds of classifiers because bad conversion
            for c in data.columns:
                if 'embedding' in c: data.drop(c, inplace=True, axis=1)

        data['text'] = original_text
        if show_color_select:
            if model_select_position == 'side':
                feature_to_color_by = st.sidebar.selectbox('Pick a feature to color points in manifold by ',
                                                           classifier_cols, 0)
            else:
                feature_to_color_by = st.selectbox('Feature to color plots by ', classifier_cols, 0)

        def are_cols_full():
            return col_index == num_cols

        for embed_pipe in StreamlitVizTracker.loaded_sentence_embeding_pipes:
            predictions = embed_pipe.predict(data, output_level=output_level, multithread=False).dropna()
            e_col = StreamlitUtilsOS.find_embed_col(predictions)
            e_com = StreamlitUtilsOS.find_embed_component(embed_pipe)
            e_com_storage_ref = StorageRefUtils.extract_storage_ref(e_com)
            emb = predictions[e_col]
            mat = np.array([x for x in emb])
            for algo in algos:
                # Only pos values for latent Dirchlet
                if algo == 'LatentDirichletAllocation': mat = np.square(mat)
                if len(mat.shape) > 2: mat = mat.reshape(len(emb), mat.shape[-1])
                hover_data = classifier_cols + ['text']
                # calc reduced dimensionality with every algo
                if 1 in target_dimensions:
                    low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo, 1, n_jobs).fit_transform(mat)
                    x = low_dim_data[:, 0]
                    y = np.zeros(low_dim_data[:, 0].shape)
                    predictions['text'] = original_text
                    tsne_df = pd.DataFrame({**{'x': x, 'y': y},
                                            **{k: predictions[k] for k in classifier_cols},
                                            **{'text': original_text}
                                            })
                    fig = px.scatter(tsne_df, x="x", y="y", color=feature_to_color_by, hover_data=hover_data)
                    subh = f"""Sentence-Embeddings =`{e_com_storage_ref}`, Manifold-Algo =`{algo}` for `D=1`"""
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
                                            **{k: predictions[k] for k in classifier_cols},
                                            **{'text': original_text}
                                            })
                    fig = px.scatter(tsne_df, x="x", y="y", color=feature_to_color_by, hover_data=hover_data)
                    subh = f"""Sentence-Embeddings =`{e_com_storage_ref}`, Manifold-Algo =`{algo}` for `D=2`"""
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
                                            **{k: predictions[k] for k in classifier_cols},
                                            **{'text': original_text}
                                            })
                    fig = px.scatter_3d(tsne_df, x="x", y="y", z='z', color=feature_to_color_by, hover_data=hover_data)
                    subh = f"""Sentence-Embeddings =`{e_com_storage_ref}`, Manifold-Algo =`{algo}` for `D=3`"""
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

# vodafonegmbh 40875 Radtingen
