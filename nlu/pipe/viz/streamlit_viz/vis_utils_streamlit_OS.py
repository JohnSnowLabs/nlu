from sparknlp.annotator import NerConverter,DependencyParserModel
from typing import List, Tuple, Optional, Dict, Union
import streamlit as st
from nlu.utils.modelhub.modelhub_utils import ModelHubUtils
import numpy as np
import pandas as pd
from sparknlp.annotator import *
from nlu.pipe.viz.streamlit_viz.streamlit_utils_OS import StreamlitUtilsOS


class VizUtilsStreamlitOS():
    """Utils for displaying various NLU viz to streamlit"""
    @staticmethod
    def visualize_classes(
            pipe, # nlu pipe
            text:Union[str,list,pd.DataFrame, pd.Series]='I love NLU and Streamlit and sunny days!',
            output_level:Optional[str]='document',
            title: Optional[str] = "Text Classification",
            metadata : bool = False,
            positions : bool = False
            )->None:
        if title:st.header(title)
        df = pipe.predict(text, output_level=output_level, metadata=metadata, positions=positions)
        classifier_cols = StreamlitUtilsOS.get_classifier_cols(pipe)
        for c in classifier_cols :
            if c not in df.columns : classifier_cols.remove(c)

        if 'text' in df.columns: classifier_cols += ['text']
        elif 'document' in df.columns: classifier_cols += ['document']
        st.write(df[classifier_cols])

    @staticmethod
    def display_model_info(model2viz):
        """Display Links to Modelhub for every NLU Ref loaded for a whitespace seperated string of NLU references"""
        default_modelhub_link = 'https://modelshub.johnsnowlabs.com/'
        nlu_refs = set(model2viz.split(' '))
        for nlu_ref in nlu_refs :
            model_hub_link = ModelHubUtils.get_url_by_nlu_refrence(nlu_ref)
            if model_hub_link is not None :
                st.sidebar.write(f"[Model info for {nlu_ref}]({model_hub_link})")
            else :
                st.sidebar.write(f"[Model info for {nlu_ref}]({default_modelhub_link})")

    @staticmethod
    def visualize_tokens_information(
            pipe, # nlu pipe
            text,
            title: Optional[str] = "Token attributes",
            show_feature_select:bool =True,
            features:Optional[List[str]] = None,
            full_metadata: bool = True,
            output_level:str = 'token',
            positions:bool = False
    ) -> None:
        """Visualizer for token attributes."""
        if title:st.header(title)
        df = pipe.predict(text, output_level=output_level, metadata=full_metadata,positions=positions)
        if not features : features = df.columns
        if show_feature_select :
            exp = st.beta_expander("Select token attributes")
            features = exp.multiselect(
                "Token attributes",
                options=list(df.columns),
                default=list(df.columns),
            )
        st.dataframe(df[features])


    @staticmethod
    def viz_streamlit(

    ): pass
    @staticmethod
    def show_logo():
        HTML_logo = """
    <div>
      <a href="https://www.johnsnowlabs.com/">
         <img src="https://nlp.johnsnowlabs.com/assets/images/logo.png" width="300"  height="100" >
       </a>
    </div>
        """
        st.sidebar.markdown(HTML_logo, unsafe_allow_html=True)


    @staticmethod
    def display_model_info(model2viz):
        """Display Links to Modelhub for every NLU Ref loaded"""
        default_modelhub_link = 'https://modelshub.johnsnowlabs.com/'
        nlu_refs = set(model2viz.split(' '))
        for nlu_ref in nlu_refs :
            model_hub_link = ModelHubUtils.get_url_by_nlu_refrence(nlu_ref)
            if model_hub_link is not None :
                st.sidebar.write(f"[Model info for {nlu_ref}]({model_hub_link})")
            else :
                st.sidebar.write(f"[Model info for {nlu_ref}]({default_modelhub_link})")

    @staticmethod
    def display_embed_vetor_information(embed_component,embed_mat):
        name = StreamlitUtilsOS.extract_name(embed_component)
        if name =='': name = 'See modelshub for more details'
        exp = st.beta_expander("Vector information")
        exp.code({"Vector Dimension ":embed_mat.shape[1],
                  "Num Vectors":embed_mat.shape[0] + embed_mat.shape[0],
                  'Vector Name':name})

    @staticmethod
    def visualize_dep_tree(
            pipe, #nlu pipe
            text:str = 'Billy likes to swim',
            title: Optional[str] = "Dependency Parse & Part-of-speech tags",
    ):
        if title:st.header(title)
        pipe.viz(text,write_to_streamlit=True,viz_type='dep')



    @staticmethod
    def visualize_ner(
            pipe, # Nlu pipe
            text:str,
            ner_tags: Optional[List[str]] = None,
            show_label_select: bool = True,
            show_table: bool = True,
            title: Optional[str] = "Named Entities",
            colors: Dict[str, str] = {},
            show_color_selector: bool = False,
    ):
        if not show_color_selector :
            if title: st.header(title)
            if ner_tags is None: ner_tags = StreamlitUtilsOS.get_NER_tags_in_pipe(pipe)
            if show_label_select:
                exp = st.beta_expander("Select entity labels to highlight")
                label_select = exp.multiselect(
                    "These labels are predicted by the NER model. Select which ones you want to display",
                    options=ner_tags,default=list(ner_tags),)
            else : label_select = ner_tags
            pipe.viz(text,write_to_streamlit=True, viz_type='ner',labels_to_viz=label_select,viz_colors=colors)
        else : # TODO WIP color select
            cols = st.beta_columns(3)
            exp = cols[0].beta_expander("Select entity labels to display")
            color = st.color_picker('Pick A Color', '#00f900')
            color = cols[2].color_picker('Pick A Color for a specific entity label', '#00f900')
            tag2color = cols[1].selectbox('Pick a ner tag to color', ner_tags)
            colors[tag2color]=color
        if show_table : st.write(pipe.predict(text, output_level='chunk'))

    @staticmethod
    def display_word_similarity(
            pipe, #nlu pipe
            default_texts: Tuple[str, str] = ("Donald Trump likes to party!", "Angela Merkel likes to party!"),
            threshold: float = 0.5,
            title: Optional[str] = "Vectors & Scalar Similarity & Vector Similarity & Embedding Visualizations  ",
            write_raw_pandas : bool = False ,
            display_embed_information:bool = True,
            similarity_matrix = True,
            show_dist_algo_select : bool = True,
            dist_metrics:List[str]  =('cosine','euclidean'),
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
        from sklearn.metrics.pairwise import distance_metrics
        if title:st.header(title)
        dist_metric_algos =distance_metrics()
        dist_algos = list(dist_metric_algos.keys())
        # TODO NORMALIZE DISTANCES TO [0,1] for non cosine
        if 'haversine'   in dist_algos    : dist_algos.remove('haversine') # not applicable in >2D
        if 'precomputed' in dist_algos  : dist_algos.remove('precomputed') # Not a dist
        exp = st.beta_expander("Select distance metric to compare vectors with")
        if show_dist_algo_select :
            dist_algo_selection = exp.multiselect("Applicable distance metrics",options=dist_algos,default=dist_metrics)
        else :
            dist_algo_selection = dist_metrics
        cols = st.beta_columns(2)
        text1 = cols[0].text_input("Text or word1",default_texts[0])
        text2 = cols[1].text_input("Text or word2",default_texts[1]) if len(default_texts) >1  else cols[1].text_input("Text or word2",'Please enter second string')
        data1 = pipe.predict(text1,output_level='token')
        data2 = pipe.predict(text2,output_level='token')
        e_col = StreamlitUtilsOS.find_embed_col(data1)
        e_com = StreamlitUtilsOS.find_embed_component(pipe)
        # get tokens for making indexes later
        tok1 = data1['token']
        tok2 = data2['token']
        emb2 = data2[e_col]
        emb1 = data1[e_col]
        embed_mat1 = np.array([x for x in emb1])
        embed_mat2 = np.array([x for x in emb2])
        if display_embed_information: VizUtilsStreamlitOS.display_embed_vetor_information(e_com,embed_mat1)

        for dist_algo in dist_algo_selection:
            sim_score = dist_metric_algos[dist_algo](embed_mat1,embed_mat2)
            sim_score = pd.DataFrame(sim_score)
            sim_score.index   = tok1.values
            sim_score.columns = tok2.values
            if write_raw_pandas :st.write(sim_score)
            if sim_score.shape == (1,1) :
                sim_score = sim_score.iloc[0][0]
                if sim_score > threshold:
                    st.success(sim_score)
                    st.success(f'Scalar Similarity={sim_score} for distance metric={dist_algo}')
                    st.error('No similarity matrix for only 2 tokens. Try entering at least 1 sentences in a field')
                else:
                    st.error(f'Scalar Similarity={sim_score} for distance metric={dist_algo}')
            else :
                # todo try error plotly import
                import plotly.express as px
                # for tok emb, sum rows and norm by rows, then sum cols and norm by cols to generate a scalar from matrix
                scalar_sim_score  = np.sum((np.sum(sim_score,axis=0) / sim_score.shape[0])) / sim_score.shape[1]
                if scalar_sim_score > threshold:
                    st.success(f'Scalar Similarity :{scalar_sim_score} for distance metric={dist_algo}')
                else:
                    st.error(f'Scalar Similarity :{scalar_sim_score} for distance metric={dist_algo}')
                if similarity_matrix:
                    fig = px.imshow(sim_score, title=f'Simmilarity Matrix for distance metric={dist_algo}')
                    st.write(fig)


