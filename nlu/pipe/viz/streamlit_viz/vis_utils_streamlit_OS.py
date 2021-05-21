from sparknlp.annotator import NerConverter,DependencyParserModel
from typing import List, Tuple, Optional, Dict
import streamlit as st
from nlu.utils.modelhub.modelhub_utils import ModelHubUtils
import numpy as np
import pandas as pd
from sparknlp.annotator import *

class VizUtilsStreamlitOS():
    classifers_OS = [ ClassifierDLModel, LanguageDetectorDL, MultiClassifierDLModel, NerDLModel, NerCrfModel, YakeModel, PerceptronModel, SentimentDLModel,
                      SentimentDetectorModel, ViveknSentimentModel, DependencyParserModel, TypedDependencyParserModel, T5Transformer, MarianTransformer, NerConverter]
    @staticmethod
    def get_classifier_cols(pipe):
        classifier_cols = []
        for c in pipe.components:
            if type(c.model) in VizUtilsStreamlitOS.classifers_OS :
                classifier_cols += pipe.anno2final_cols[c.model]
        return  classifier_cols
    @staticmethod
    def visualize_classes(
            pipe, # nlu pipe
            text:str='I love NLU and Streamlit and sunny days!',
            output_level:Optional[str]='document',
            title: Optional[str] = "Text Classification",
            metadata : bool = True,
            )->None:
        if title:st.header(title)
        df = pipe.predict(text, output_level=output_level, metadata=metadata)
        classifier_cols = VizUtilsStreamlitOS.get_classifier_cols(pipe)
        for c in classifier_cols :
            if c not in df.columns : classifier_cols.remove(c)

        st.write(df[classifier_cols])


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
    """Utils for displaying various NLU viz to streamlit"""
    @staticmethod
    def infer_viz_open_source(pipe)->str:
        """For a given NLUPipeline with only open source components, infers which visualizations are applicable. """
        for c in pipe.components:
            if isinstance(c.model, NerConverter) : return 'ner'
            if isinstance(c.model, NerConverter) : return 'ner'
            if isinstance(c.model, DependencyParserModel) : return 'dep'

    @staticmethod
    def find_embed_col(df, search_multi=False):
        """Find col that contains embed"""
        if not search_multi:
            for c in df.columns:
                if 'embed'in c : return c
        else:
            e_cols =[]
            for c in df.columns:
                if 'embed'in c : e_cols.append(c)
        return  e_cols
    @staticmethod
    def find_embed_component(p):
        """Find NER component in pipe"""
        for c in p.components :
            if 'embed' in c.info.outputs[0] : return c
        st.warning("No Embed model in pipe")
        return None
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
    def extract_name(component_or_pipe):
        name =''
        if hasattr(component_or_pipe.info,'nlu_ref') : name = component_or_pipe.info.nlu_ref
        elif hasattr(component_or_pipe,'storage_ref') : name = component_or_pipe.info.storage_ref
        elif hasattr(component_or_pipe,'nlp_ref') : name = component_or_pipe.info.nlp_ref
        return name

    @staticmethod
    def display_embed_vetor_information(embed_component,embed_mat):
        name = VizUtilsStreamlitOS.extract_name(embed_component)
        if name =='': name = 'See modelshub for more details'
        exp = st.beta_expander("Vector information")
        exp.code({"Vector Dimension ":embed_mat.shape[1],
                  "Num Vectors":embed_mat.shape[0] + embed_mat.shape[0],
                  'Vector Name':name})

    @staticmethod
    def display_dep_tree(
            pipe, #nlu pipe
            text,
            title: Optional[str] = "Dependency Parse & Part-of-speech tags",
    ):
        if title:st.header(title)
        pipe.viz(text,write_to_streamlit=True,viz_type='dep')
    @staticmethod
    def find_ner_model(p):
        """Find NER component in pipe"""
        from sparknlp.annotator import NerDLModel,NerCrfModel
        for c in p.components :
            if isinstance(c.model,(NerDLModel,NerCrfModel)):return c.model
        st.warning("No NER model in pipe")
        return None

    @staticmethod
    def get_NER_tags_in_pipe(p):
        """Get NER tags in pipe, used for showing visualizable tags"""
        n = VizUtilsStreamlitOS.find_ner_model(p)
        if n is None : return []
        classes_predicted_by_ner_model = n.getClasses()
        split_iob_tags = lambda s : s.split('-')[1] if '-' in s else ''
        classes_predicted_by_ner_model = list(map(split_iob_tags,classes_predicted_by_ner_model))
        while '' in classes_predicted_by_ner_model : classes_predicted_by_ner_model.remove('')
        classes_predicted_by_ner_model = list(set(classes_predicted_by_ner_model))
        return classes_predicted_by_ner_model


    @staticmethod
    def display_ner(
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
            if ner_tags is None: ner_tags = VizUtilsStreamlitOS.get_NER_tags_in_pipe(pipe)
            if show_label_select:
                exp = st.beta_expander("Select entity labels to highlight")
                label_select = exp.multiselect(
                    "These labels are predicted by the NER model. Select which ones you want to display",
                    options=ner_tags,default=list(ner_tags),)
            pipe.viz(text,write_to_streamlit=True, viz_type='ner',labels_to_viz=label_select,viz_colors=colors)
        else : # TODO WIP color select
            cols = st.beta_columns(3)
            exp = cols[0].beta_expander("Select entity labels to display")
            color = st.color_picker('Pick A Color', '#00f900')
            color = cols[2].color_picker('Pick A Color for a specific entity label', '#00f900')
            tag2color = cols[1].selectbox('Pick a ner tag to color', ner_tags)
            colors[tag2color]=color

    @staticmethod
    def display_word_simmilarity(
            pipe, #nlu pipe
            default_texts: Tuple[str, str] = ("Donald Trump likes to party!", "Angela Merkel likes to party!"),
            threshold: float = 0.5,
            title: Optional[str] = "Vectors & Scalar Similarity & Vector Similarity & Embedding Visualizations  ",
            write_raw_pandas : bool = False ,
            display_embed_information:bool = True,
            display_embed_matrix = True,
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
        dist_algo_selection = exp.multiselect("Applicable distance metrics",options=dist_algos,default=['cosine'],)

        cols = st.beta_columns(2)
        text1 = cols[0].text_input("Text or word1",default_texts[0])
        text2 = cols[1].text_input("Text or word2",default_texts[1])
        data1 = pipe.predict(text1,output_level='token')
        data2 = pipe.predict(text2,output_level='token')
        e_col = VizUtilsStreamlitOS.find_embed_col(data1)
        e_com = VizUtilsStreamlitOS.find_embed_component(pipe)
        # get tokens for making indexes later
        tok1 = data1['token']
        tok2 = data2['token']
        emb2 = data2[e_col]
        emb1 = data1[e_col]
        embed_mat1 = np.array([x for x in emb1])
        embed_mat2 = np.array([x for x in emb2])
        if display_embed_information: VizUtilsStreamlitOS.display_embed_vetor_information(e_com,embed_mat1)
        def calc_sim(embed_mat1,embed_mat2,metric=''):
            sim_mat = dist_metric_algos[metric](embed_mat1,embed_mat2)
            return sim_mat
        for dist_algo in dist_algo_selection:
            sim_score = calc_sim(embed_mat1,embed_mat2,dist_algo)
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
                if display_embed_matrix:
                    fig = px.imshow(sim_score, title=f'Simmilarity Matrix for distance metric={dist_algo}')
                    st.write(fig)



