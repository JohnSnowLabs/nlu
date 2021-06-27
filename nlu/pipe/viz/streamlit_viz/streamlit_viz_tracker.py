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

## TODO THIS BECOME VIZ TRACKER!! MODEL-VIEW-CONTROLLER Pattern!!!  THis is the MODEL, STREAMLIT IS the VIEW, STreamlit-UI+Methods called by that are CONTROLLER
class StreamlitVizTracker():
    """Track the status of the visualizations and models loaded in the Streamlit Web View. This is the Model part of the MVC pattern"""
    _set_block_container_style()
    loaded_word_embeding_pipes = []
    loaded_sentence_embeding_pipes = [] # todo track
    loaded_document_classifier_pipes = []
    loaded_token_pipes = []
    loaded_token_level_classifiers = []
    footer_displayed = False



    @staticmethod
    def pad_duplicate_tokens(tokens):
        """For every duplicate token in input list, ads N whitespaces for the Nth duplicate"""
        duplicates = {}

        for i,s in enumerate(tokens) :
            if s in duplicates.keys():duplicates[s].append(i)
            else :                    duplicates[s]=[i]
        for i, d in enumerate(duplicates.items()):
            for i,idx in enumerate(d[1]):tokens[idx]=d[0]+' '*i
        return tokens
    @staticmethod
    def RAW_HTML_link(text,url,CSS_class):return f"""<p class="{CSS_class}" style="padding:0px;" > <a href="{url}">{text}</a> </p>"""
    @staticmethod
    def style_link(text,url, CSS_class):
        return  f"""
<p class="{CSS_class}" style="
width: fit-content;
padding:0px;
color : #1E77B7; 
font-family: 'Roboto', sans-serif;
font-weight: bold;
font-size: 14px;
line-height: 17px;
box-sizing: content-box;
overflow: hidden;
display: block;
color: #0098da !important;
word-wrap: break-word;
" >
<a href="{url}">{text}</a> 
</p>"""
    @staticmethod
    def style_model_link(model,text,url, CSS_class):
        return  f"""
<p class="{CSS_class}" style="
width: fit-content;
padding:0px;
color : #1E77B7; 
font-family: 'Roboto', sans-serif;
font-weight: bold;
font-size: 14px;
line-height: 17px;
box-sizing: content-box;
overflow: hidden;
display: block;
color: #0098da !important;
word-wrap: break-word;
" >
<a href="{url}">{text}<div style:"color=rgb(246, 51, 102);">{model}</div></a> 
</p>"""
    @staticmethod
    def display_infos():
        FOOTER       = """<span style="font-size: 0.75em">{}</span>"""
        field_info   = """**INFO:** You can type in the model selection fields to search and filter."""
        iso_info     = """**INFO:** NLU model references have the structure: `<iso_language_code>.<model_name>.<dataset>` . [Based on the `ISO Language Codes`](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes). If no language defined, `en.` will be assumed as default ' ,"""
        ISO_FOOTER   = FOOTER.format(field_info)
        FIELD_FOOTER = FOOTER.format(iso_info)
        st.sidebar.markdown(ISO_FOOTER, unsafe_allow_html=True)
        st.sidebar.markdown(FIELD_FOOTER, unsafe_allow_html=True)
    @staticmethod
    def display_footer():
        if not StreamlitVizTracker.footer_displayed:
            StreamlitVizTracker.display_infos()
            nlu_link     = 'https://nlu.johnsnowlabs.com/'
            nlp_link     = 'http://nlp.johnsnowlabs.com/'
            jsl_link     = 'https://johnsnowlabs.com/'
            doc_link     = 'TODO'
            powerd_by    = f"""Powerd by [`NLU`]({nlu_link }) & [`Spark NLP`]({nlp_link}) from [`John Snow Labs `]({jsl_link}) Checkout [`The Docs`]({doc_link}) for more infos"""
            FOOTER       = """<span style="font-size: 0.75em">{}</span>"""
            POWER_FOOTER = FOOTER.format(powerd_by)
            st.sidebar.markdown(POWER_FOOTER, unsafe_allow_html=True)
            StreamlitVizTracker.footer_displayed = True
    @staticmethod
    def show_logo(sidebar=True):
        HTML_logo = """
    <div>
      <a href="https://www.johnsnowlabs.com/">
         <img src="https://nlp.johnsnowlabs.com/assets/images/logo.png" width="300"  height="100" >
       </a>
    </div>
        """
        if sidebar : st.sidebar.markdown(HTML_logo, unsafe_allow_html=True)
        else: st.markdown(HTML_logo, unsafe_allow_html=True)
    @staticmethod
    def display_embed_vetor_information(embed_component,embed_mat):
        name = StreamlitUtilsOS.extract_name(embed_component)
        if name =='': name = 'See modelshub for more details'
        exp = st.beta_expander("Vector information")
        exp.code({"Vector Dimension ":embed_mat.shape[1],
                  "Num Vectors":embed_mat.shape[0] + embed_mat.shape[0],
                  'Vector Name':name})
    @staticmethod
    def display_model_info(model2viz=' ',pipes=[],apply_style=True, display_component_wise_info=True,display_component_summary=True):
        """Display Links to Modelhub for every NLU Ref loaded and also every component in pipe"""
        default_modelhub_link = 'https://nlp.johnsnowlabs.com/models'
        nlu_refs = model2viz.split(' ')
        # for p in classifier_pipes + embed_pipes + token_pipes  :nlu_refs.append(p.nlu_ref)

        for p in StreamlitVizTracker.loaded_word_embeding_pipes + StreamlitVizTracker.loaded_document_classifier_pipes + StreamlitVizTracker.loaded_token_pipes  : nlu_refs.append(p.nlu_ref)
        nlu_refs = set(nlu_refs)
        st.sidebar.subheader("NLU pipeline components info")
        nlu_ref_infos = []
        nlu_ref_infos.append(StreamlitVizTracker.style_link("Search over 1000+ scalable SOTA models in John Snow Labs Modelhub", default_modelhub_link, CSS_class='nlu_model_info'))
        for nlu_ref in nlu_refs :
            model_hub_link = ModelHubUtils.get_url_by_nlu_refrence(nlu_ref)
            link_text =f'JSL Modelhub page for '# {nlu_ref}'
            if model_hub_link is None or model_hub_link == default_modelhub_link :
                continue
                # link_text =f'More infos here {nlu_ref}'
                # model_hub_link = default_modelhub_link
            if apply_style:
                nlu_ref_info = StreamlitVizTracker.style_model_link(nlu_ref, link_text, model_hub_link, CSS_class='nlu_model_info')
                nlu_ref_infos.append(nlu_ref_info)
                # st.sidebar.write(VizUtilsStreamlitOS.style_link(link_text,model_hub_link),unsafe_allow_html=True)
            else :
                nlu_ref_info = StreamlitVizTracker.RAW_HTML_link(link_text, model_hub_link, CSS_class='nlu_model_info')
                nlu_ref_infos.append(nlu_ref_info)
                # st.sidebar.write(nlu_ref_info)

        n = '\n'
        HTML_INFO = f"<p>{n.join(nlu_ref_infos)}</p>"
        st.sidebar.markdown(HTML_INFO, unsafe_allow_html=True)
        c_names = []
        if display_component_wise_info :
            for pipe in pipes :
                if pipe is None : continue
                for c in pipe.components :
                    c_name = f"`{type(c.model).__name__}`"
                    c_names.append(c_name)
            if model2viz[-1]==' ': model2viz = model2viz[:-1]
        FOOTER       = """<span style="font-size: 0.75em">{}</span>"""
        component_info = f"**Info:** You can load all models active in 1 line via `nlu.load('{' '.join(nlu_refs)}')` which provides this with a optimized CPU build and components: {', '.join(set(c_names))}"
        component_info = FOOTER.format(component_info)
        st.sidebar.markdown(component_info, unsafe_allow_html=True)
        if display_component_summary:
            parameter_infos = {}
            for p in pipes :
                if p is None : continue
                parameter_infos[p.nlu_ref]=StreamlitVizTracker.get_pipe_param_dict(p)
            exp = st.beta_expander("NLU Pipeline components and parameters information")
            exp.write(parameter_infos)
    @staticmethod
    def get_pipe_param_dict(pipe):
        # loop over ever model in pipeline stages  and then loop over the models params
        all_params = {}
        from sparknlp.base import LightPipeline
        stages = pipe.spark_transformer_pipe.pipeline_model.stages if isinstance(pipe.spark_transformer_pipe, (LightPipeline)) else pipe.spark_transformer_pipe.stages
        for stage in stages:
            all_params[str(stage)]={}
            params = stage.extractParamMap()
            for param_name, param_value in params.items():
                # print(f'model={stage} param_name={param_name}, param_value={param_value}')
                all_params[str(stage)][param_name.name]=param_value

        return all_params



    # @staticmethod
    # def viz_streamlit(
    #         pipe,
    #         # Base Params
    #         text:Union[str, List[str], pd.DataFrame, pd.Series],
    #         model_selection:List[str]=[],
    #         # NER PARAMS
    #         # default_ner_model2viz:Union[str, List[str]] = 'en.ner.onto.electra.base',
    #         # SIMILARITY PARAMS
    #         similarity_texts:Tuple[str,str]= ('I love NLU <3', 'I love Streamlit <3'),
    #         title:str = 'NLU ❤️ Streamlit - Prototype your NLP startup in 0 lines of code' ,
    #         sub_title:str = 'Play with over 1000+ scalable enterprise NLP models',
    #         side_info:str = None,
    #         # UI PARAMS
    #         visualizers:List[str] = ( "dependency_tree", "ner",  "similarity", "token_features", 'classification','manifold'),
    #         show_models_info:bool = True,
    #         show_model_select:bool = True,
    #         show_viz_selection:bool = False,
    #         show_logo:bool=True,
    #         set_wide_layout_CSS:bool=True,
    #         show_code_snippets:bool=False,
    #         model_select_position:str = 'side' , # main or side
    #         display_infos:bool=True,
    #         key:str = "NLU_streamlit",
    #         display_footer :bool =  True ,
    #         num_similarity_cols:int=2,
    #
    #         # NEW PARAMS
    #         # MANIfold
    #         num_manifold_cols:int=3,
    #         manifold_algos:List[str]=('TSNE'),
    #
    #         # SIMY
    #         similarity_algos:List[str]=('COSINE'),
    # )-> None:
    #     """Visualize either individual building blocks for streamlit or a full UI to experiment and explore models with"""
    #     StreamlitVizTracker.footer_displayed = not display_footer
    #     if set_wide_layout_CSS : _set_block_container_style()
    #     if title: st.title(title)
    #     if sub_title: st.subheader(sub_title)
    #     if show_logo :StreamlitVizTracker.show_logo()
    #     if side_info : st.sidebar.markdown(side_info)
    #     text    = st.text_area("Enter text you want to visualize below", text, key=key)
    #     ner_model_2_viz     = pipe.nlu_ref
    #     if show_model_select :
    #         show_code_snippets = st.sidebar.checkbox('Generate code snippets', value=show_code_snippets)
    #         if model_selection == [] : model_selection = Discoverer.get_components('ner',include_pipes=True)
    #         model_selection.sort()
    #         if model_select_position == 'side':ner_model_2_viz = st.sidebar.selectbox("Select a NER model.",model_selection,index=model_selection.index(pipe.nlu_ref.split(' ')[0]))
    #         else : ner_model_2_viz = st.selectbox("Select a NER model",model_selection,index=model_selection.index(pipe.nlu_ref.split(' ')[0]))
    #
    #     active_visualizers = visualizers
    #     if show_viz_selection: active_visualizers = st.sidebar.multiselect("Visualizers",options=visualizers,default=visualizers,key=key)
    #
    #     all_models = ner_model_2_viz + ' en.dep.typed '  if 'dependency_tree' in active_visualizers  else ner_model_2_viz
    #     ner_pipe, tree_pipe =  None,None
    #     if 'ner' in active_visualizers :
    #         ner_pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
    #         StreamlitVizTracker.visualize_ner(ner_pipe, text, generate_code_sample=show_code_snippets, key=key, show_model_select=False, show_text_input=True, show_logo=False, show_infos=False)
    #     if 'dependency_tree' in active_visualizers :
    #         tree_pipe = StreamlitUtilsOS.get_pipe('en.dep.typed') # if not ValidateVizPipe.viz_tree_satisfied(pipe) else pipe
    #         StreamlitVizTracker.visualize_dep_tree(tree_pipe, text, generate_code_sample=show_code_snippets, key=key, show_infos=False, show_logo=False)
    #     if 'token_features' in active_visualizers:
    #         ner_pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
    #         StreamlitVizTracker.visualize_tokens_information(ner_pipe, text, generate_code_sample=show_code_snippets, key=key, model_select_position=model_select_position, show_infos=False, show_logo=False, )
    #     if 'classification' in active_visualizers:
    #         ner_pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
    #         StreamlitVizTracker.visualize_classes(ner_pipe, text, generate_code_sample=show_code_snippets, key=key, model_select_position=model_select_position, show_infos=False, show_logo=False)
    #     if 'similarity' in active_visualizers:
    #         ner_pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
    #         StreamlitVizTracker.display_word_similarity(ner_pipe, similarity_texts, generate_code_sample=show_code_snippets, model_select_position=model_select_position, show_infos=False, show_logo=False, num_cols=num_similarity_cols)
    #     if 'manifold' in active_visualizers :
    #         ner_pipe = pipe if ner_model_2_viz in pipe.nlu_ref.split(' ')  else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
    #         StreamlitVizTracker.display_low_dim_embed_viz_token(ner_pipe, similarity_texts, generate_code_sample=show_code_snippets, model_select_position=model_select_position, show_infos=False, show_logo=False, num_cols=num_manifold_cols)
    #
    #     StreamlitVizTracker.display_model_info(all_models, [ner_pipe, tree_pipe])
    #     if show_models_info :
    #         pass
    #     if display_infos : StreamlitVizTracker.display_footer()
    #
    #
    # @staticmethod
    # def visualize_classes(
    #         pipe, # nlu pipe
    #         text:Union[str,list,pd.DataFrame, pd.Series, List[str]]=('I love NLU and Streamlit and sunny days!', 'I hate rainy daiys','CALL NOW AND WIN 1000$M'),
    #         output_level:Optional[str]='document',
    #         title: Optional[str] = "Text Classification",
    #         sub_title: Optional[str] = 'View predicted `classes` and `confidences` for `hundreds of text classifiers` in `over 200 languages`',
    #         metadata : bool = False,
    #         positions : bool = False,
    #         set_wide_layout_CSS:bool=True,
    #         generate_code_sample:bool = False,
    #         key:str = "NLU_streamlit",
    #         show_model_selector : bool = True ,
    #         model_select_position:str = 'side' ,
    #         show_infos:bool = True,
    #         show_logo:bool = True,
    #         )->None:
    #     if show_logo :StreamlitVizTracker.show_logo()
    #     if set_wide_layout_CSS : _set_block_container_style()
    #     if title:st.header(title)
    #     if sub_title:st.subheader(sub_title)
    #
    #     # if generate_code_sample: st.code(get_code_for_viz('CLASSES',StreamlitUtilsOS.extract_name(pipe),text))
    #     if not isinstance(text, (pd.DataFrame, pd.Series)):
    #         text = st.text_area('Enter N texts, seperated by new lines to view classification results for','\n'.join(text) if isinstance(text,list) else text, key=key)
    #         text = text.split("\n")
    #         while '' in text : text.remove('')
    #     classifier_pipes = [pipe]
    #     classifier_components_usable = [e for e in Discoverer.get_components('classify',True, include_aliases=True)]
    #     classifier_components = StreamlitUtilsOS.find_all_classifier_components(pipe)
    #     loaded_classifier_nlu_refs = [c.info.nlu_ref for c in classifier_components]
    #
    #     for l in loaded_classifier_nlu_refs:
    #         if 'converter' in l :
    #             loaded_classifier_nlu_refs.remove(l)
    #             continue
    #         if l not in classifier_components_usable : classifier_components_usable.append(l)
    #     classifier_components_usable.sort()
    #     loaded_classifier_nlu_refs.sort()
    #     if show_model_selector :
    #         if model_select_position =='side':classifier_components_selection   = st.sidebar.multiselect("Pick additional Classifiers",options=classifier_components_usable,default=loaded_classifier_nlu_refs,key = key)
    #         else:classifier_components_selection   = st.multiselect("Pick additional Classifiers",options=classifier_components_usable,default=loaded_classifier_nlu_refs,key = key)
    #     # else : ValueError("Please define model_select_position as main or side")
    #     classifier_algos_to_load = list(set(classifier_components_selection) - set(loaded_classifier_nlu_refs))
    #     for classifier in classifier_algos_to_load:classifier_pipes.append(nlu.load(classifier))
    #     StreamlitVizTracker.loaded_document_classifier_pipes+= classifier_pipes
    #     if generate_code_sample:st.code(get_code_for_viz('CLASSES',[StreamlitUtilsOS.extract_name(p) for p  in classifier_pipes],text))
    #
    #     dfs = []
    #     all_classifier_cols=[]
    #     for p in classifier_pipes :
    #         df = p.predict(text, output_level=output_level, metadata=metadata, positions=positions)
    #         classifier_cols = StreamlitUtilsOS.get_classifier_cols(p)
    #         for c in classifier_cols :
    #             if c not in df.columns : classifier_cols.remove(c)
    #
    #         if 'text' in df.columns: classifier_cols += ['text']
    #         elif 'document' in df.columns: classifier_cols += ['document']
    #         all_classifier_cols+= classifier_cols
    #         dfs.append(df)
    #     df = pd.concat(dfs, axis=1)
    #     df = df.loc[:,~df.columns.duplicated()]
    #     for c in all_classifier_cols :
    #         if c not in df.columns : all_classifier_cols.remove(c)
    #     all_classifier_cols = list(set(all_classifier_cols))
    #
    #     if len(all_classifier_cols) == 0: st.error('No classes detected')
    #     else :st.write(df[all_classifier_cols],key=key)
    #     if show_infos :
    #         # VizUtilsStreamlitOS.display_infos()
    #         StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
    #         StreamlitVizTracker.display_footer()
    #
    #
    #
    # @staticmethod
    # def visualize_tokens_information(
    #         pipe, # nlu pipe
    #         text:str,
    #         title: Optional[str] = "Token Features",
    #         sub_title: Optional[str] ='Pick from `over 1000+ models` on the left and `view the generated features`',
    #         show_feature_select:bool =True,
    #         features:Optional[List[str]] = None,
    #         full_metadata: bool = True,
    #         output_level:str = 'token',
    #         positions:bool = False,
    #         set_wide_layout_CSS:bool=True,
    #         generate_code_sample:bool = False,
    #         key = "NLU_streamlit",
    #         show_model_select = True,
    #         model_select_position:str = 'side' , # main or side
    #         show_infos:bool = True,
    #         show_logo:bool = True,
    #         show_text_input:bool = True,
    # ) -> None:
    #     """Visualizer for token features."""
    #     StreamlitVizTracker.footer_displayed=False
    #     if show_logo :StreamlitVizTracker.show_logo()
    #     if set_wide_layout_CSS : _set_block_container_style()
    #     if title:st.header(title)
    #     # if generate_code_sample: st.code(get_code_for_viz('TOKEN',StreamlitUtilsOS.extract_name(pipe),text))
    #     if sub_title:st.subheader(sub_title)
    #     token_pipes = [pipe]
    #     if show_text_input : text = st.text_area("Enter text you want to view token features for", text, key=key)
    #     if show_model_select :
    #         token_pipes_components_usable = [e for e in Discoverer.get_components(get_all=True)]
    #         loaded_nlu_refs = [c.info.nlu_ref for c in pipe.components]
    #
    #         for l in loaded_nlu_refs:
    #             if 'converter' in l :
    #                 loaded_nlu_refs.remove(l)
    #                 continue
    #             if l not in token_pipes_components_usable : token_pipes_components_usable.append(l)
    #         token_pipes_components_usable = list(set(token_pipes_components_usable))
    #         loaded_nlu_refs = list(set(loaded_nlu_refs))
    #         if '' in loaded_nlu_refs : loaded_nlu_refs.remove('')
    #         if ' ' in loaded_nlu_refs : loaded_nlu_refs.remove(' ')
    #         token_pipes_components_usable.sort()
    #         loaded_nlu_refs.sort()
    #         if model_select_position =='side':model_selection   = st.sidebar.multiselect("Pick any additional models for token features",options=token_pipes_components_usable,default=loaded_nlu_refs,key = key)
    #         else:model_selection   = st.multiselect("Pick any additional models for token features",options=token_pipes_components_usable,default=loaded_nlu_refs,key = key)
    #         # else : ValueError("Please define model_select_position as main or side")
    #         models_to_load = list(set(model_selection) - set(loaded_nlu_refs))
    #         for model in models_to_load:token_pipes.append(nlu.load(model))
    #         StreamlitVizTracker.loaded_token_pipes+= token_pipes
    #     if generate_code_sample:st.code(get_code_for_viz('TOKEN',[StreamlitUtilsOS.extract_name(p) for p  in token_pipes],text))
    #     dfs = []
    #     for p in token_pipes:
    #         df = p.predict(text, output_level=output_level, metadata=full_metadata,positions=positions)
    #         dfs.append(df)
    #
    #
    #     df = pd.concat(dfs,axis=1)
    #     df = df.loc[:,~df.columns.duplicated()]
    #     if show_feature_select :
    #         exp = st.beta_expander("Select token features to display")
    #         features = exp.multiselect(
    #             "Token features",
    #             options=list(df.columns),
    #             default=list(df.columns)
    #         )
    #     st.dataframe(df[features])
    #     if show_infos :
    #         # VizUtilsStreamlitOS.display_infos()
    #         StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
    #         StreamlitVizTracker.display_footer()
    #
    #
    # @staticmethod
    # def visualize_dep_tree(
    #         pipe, #nlu pipe
    #         text:str = 'Billy likes to swim',
    #         title: Optional[str] = "Dependency Parse & Part-of-speech tags",
    #         sub_title: Optional[str] = 'POS tags define a `grammatical label` for `each token` and the `Dependency Tree` classifies `Relations between the tokens` ',
    #         set_wide_layout_CSS:bool=True,
    #         generate_code_sample:bool = False,
    #         key = "NLU_streamlit",
    #         show_infos:bool = True,
    #         show_logo:bool = True,
    #         show_text_input:bool = True,
    # ):
    #     StreamlitVizTracker.footer_displayed=False
    #     if show_logo :StreamlitVizTracker.show_logo()
    #     if set_wide_layout_CSS : _set_block_container_style()
    #     if title:st.header(title)
    #     if show_text_input : text = st.text_area("Enter text you want to visualize dependency tree for ", text, key=key)
    #     if sub_title:st.subheader(sub_title)
    #     if generate_code_sample: st.code(get_code_for_viz('TREE',StreamlitUtilsOS.extract_name(pipe),text))
    #     pipe.viz(text,write_to_streamlit=True,viz_type='dep', streamlit_key=key)
    #     if show_infos :
    #         # VizUtilsStreamlitOS.display_infos()
    #         StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
    #         StreamlitVizTracker.display_footer()
    #
    #
    #
    # @staticmethod
    # def visualize_ner(
    #         pipe, # Nlu pipe
    #         text:str,
    #         ner_tags: Optional[List[str]] = None,
    #         show_label_select: bool = True,
    #         show_table: bool = False,
    #         title: Optional[str] = "Named Entities",
    #         sub_title: Optional[str] = "Recognize various `Named Entities (NER)` in text entered and filter them. You can select from over `100 languages` in the dropdown.",
    #         colors: Dict[str, str] = {},
    #         show_color_selector: bool = False,
    #         set_wide_layout_CSS:bool=True,
    #         generate_code_sample:bool = False,
    #         key = "NLU_streamlit",
    #         model_select_position:str = 'side',
    #         show_model_select : bool = True,
    #         show_text_input:bool = True,
    #         show_infos:bool = True,
    #         show_logo:bool = True,
    #
    # ):
    #     StreamlitVizTracker.footer_displayed=False
    #     if set_wide_layout_CSS : _set_block_container_style()
    #     if show_logo :StreamlitVizTracker.show_logo()
    #     if show_model_select :
    #         model_selection = Discoverer.get_components('ner',include_pipes=True)
    #         model_selection.sort()
    #         if model_select_position == 'side':ner_model_2_viz = st.sidebar.selectbox("Select a NER model",model_selection,index=model_selection.index(pipe.nlu_ref.split(' ')[0]))
    #         else : ner_model_2_viz = st.selectbox("Select a NER model",model_selection,index=model_selection.index(pipe.nlu_ref.split(' ')[0]))
    #         pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
    #     if title: st.header(title)
    #     if show_text_input : text = st.text_area("Enter text you want to visualize NER classes for below", text, key=key)
    #     if sub_title : st.subheader(sub_title)
    #     if generate_code_sample: st.code(get_code_for_viz('NER',StreamlitUtilsOS.extract_name(pipe),text))
    #     if ner_tags is None: ner_tags = StreamlitUtilsOS.get_NER_tags_in_pipe(pipe)
    #
    #     if not show_color_selector :
    #         if show_label_select:
    #             exp = st.beta_expander("Select entity labels to highlight")
    #             label_select = exp.multiselect(
    #                 "These labels are predicted by the NER model. Select which ones you want to display",
    #                 options=ner_tags,default=list(ner_tags))
    #         else : label_select = ner_tags
    #         pipe.viz(text,write_to_streamlit=True, viz_type='ner',labels_to_viz=label_select,viz_colors=colors, streamlit_key=key)
    #     else : # TODO WIP color select
    #         cols = st.beta_columns(3)
    #         exp = cols[0].beta_expander("Select entity labels to display")
    #         color = st.color_picker('Pick A Color', '#00f900',key = key)
    #         color = cols[2].color_picker('Pick A Color for a specific entity label', '#00f900',key = key)
    #         tag2color = cols[1].selectbox('Pick a ner tag to color', ner_tags,key = key)
    #         colors[tag2color]=color
    #     if show_table : st.write(pipe.predict(text, output_level='chunk'),key = key)
    #
    #     if show_infos :
    #         # VizUtilsStreamlitOS.display_infos()
    #         StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
    #         StreamlitVizTracker.display_footer()
    #
    # @staticmethod
    # def display_word_similarity(
    #         pipe, #nlu pipe
    #         default_texts: Tuple[str, str] = ("Donald Trump likes to party!", "Angela Merkel likes to party!"),
    #         threshold: float = 0.5,
    #         title: Optional[str] = "Embeddings Similarity Matrix &  Visualizations  ",
    #         sub_tile :Optional[str]="Visualize `word-wise similarity matrix` and calculate `similarity scores` for `2 texts` and every `word embedding` loaded",
    #         write_raw_pandas : bool = False,
    #         display_embed_information:bool = True,
    #         similarity_matrix = True,
    #         show_algo_select : bool = True,
    #         dist_metrics:List[str]  =('cosine'),
    #         set_wide_layout_CSS:bool=True,
    #         generate_code_sample:bool = False,
    #         key:str = "NLU_streamlit",
    #         num_cols:int=2,
    #         display_scalar_similarities : bool = False ,
    #         display_similarity_summary:bool = False,
    #         model_select_position:str = 'side' , # main or side
    #         show_infos:bool = True,
    #         show_logo:bool = True,
    # ):
    #
    #     """We visualize the following cases :
    #     1. Simmilarity between 2 words - > sim (word_emb1, word_emb2)
    #     2. Simmilarity between 2 sentences -> let weTW stand word word_emb of token T and sentence S
    #         2.1. Raw token level with merged embeddings -> sim([we11,we21,weT1], [we12,we22,weT2])
    #         2.2  Autogenerate sentemb, basically does 2.1 in the Spark NLP backend
    #         2.3 Already using sentence_embedder model -> sim(se1,se2)
    #     3. Simmilarity between token and sentence -> sim([we11,w21,wT1], se2)
    #     4. Mirrored 3
    #      """
    #     # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
    #     StreamlitVizTracker.footer_displayed=False
    #     try :
    #         import plotly.express as px
    #         from sklearn.metrics.pairwise import distance_metrics
    #     except :st.error("You need the sklearn and plotly package in your Python environment installed for similarity visualizations. Run <pip install sklearn plotly>")
    #     if set_wide_layout_CSS : _set_block_container_style()
    #     if title:st.header(title)
    #     if show_logo :StreamlitVizTracker.show_logo()
    #     if sub_tile : st.subheader(sub_tile)
    #
    #     StreamlitVizTracker.loaded_word_embeding_pipes = []
    #     dist_metric_algos =distance_metrics()
    #     dist_algos = list(dist_metric_algos.keys())
    #     # TODO NORMALIZE DISTANCES TO [0,1] for non cosine
    #     if 'haversine'   in dist_algos    : dist_algos.remove('haversine') # not applicable in >2D
    #     if 'precomputed' in dist_algos  : dist_algos.remove('precomputed') # Not a dist
    #     cols = st.beta_columns(2)
    #     text1 = cols[0].text_input("Text or word1",default_texts[0],key = key)
    #     text2 = cols[1].text_input("Text or word2",default_texts[1], key=key) if len(default_texts) >1  else cols[1].text_input("Text or word2",'Please enter second string',key = key)
    #     # exp = st.sidebar.beta_expander("Select additional Embedding Models and distance metric to compare ")
    #     e_coms = StreamlitUtilsOS.find_all_embed_components(pipe)
    #     embed_algos_to_load = []
    #     embed_pipes = [pipe]
    #     dist_algo_selection = dist_metrics
    #     if show_algo_select :
    #         # emb_components_usable = Discoverer.get_components('embed')
    #         emb_components_usable = [e for e in Discoverer.get_components('embed',True, include_aliases=True) if 'chunk' not in e and 'sentence' not in e]
    #         loaded_embed_nlu_refs = []
    #         loaded_storage_refs = []
    #         for c in e_coms :
    #             if not  hasattr(c.info,'nlu_ref'): continue
    #             r = c.info.nlu_ref
    #             if 'en.' not in r and 'embed.' not  in r and 'ner' not in r : loaded_embed_nlu_refs.append('en.embed.' + r)
    #             elif 'en.'  in r and 'embed.' not  in r  and 'ner' not in r:
    #                 r = r.split('en.')[0]
    #                 loaded_embed_nlu_refs.append('en.embed.' + r)
    #             else :
    #                 loaded_embed_nlu_refs.append(StorageRefUtils.extract_storage_ref(c))
    #             loaded_storage_refs.append(StorageRefUtils.extract_storage_ref(c))
    #
    #         for l in loaded_embed_nlu_refs:
    #             if l not in emb_components_usable : emb_components_usable.append(l)
    #         # embed_algo_selection = exp.multiselect("Click to pick additional Embedding Algorithm",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
    #         # dist_algo_selection = exp.multiselect("Click to pick additional Distance Metric", options=dist_algos, default=dist_metrics, key = key)
    #         emb_components_usable.sort()
    #         loaded_embed_nlu_refs.sort()
    #         dist_algos.sort()
    #         # dist_metrics.sort()
    #         if model_select_position =='side':
    #             embed_algo_selection   = st.sidebar.multiselect("Pick additional Word Embeddings for the Similarity Matrix",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
    #             dist_algo_selection =  st.sidebar.multiselect("Pick additional Similarity Metrics ", options=dist_algos, default=dist_metrics, key = key)
    #         else :
    #             exp = st.beta_expander("Pick additional Word Embeddings and Similarity Metrics")
    #             embed_algo_selection   = exp.multiselect("Pick additional Word Embeddings for the Similarity Matrix",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
    #             dist_algo_selection =  exp.multiselect("Pick additional Similarity Metrics ", options=dist_algos, default=dist_metrics, key = key)
    #         embed_algos_to_load = list(set(embed_algo_selection) - set(loaded_embed_nlu_refs))
    #
    #     for embedder in embed_algos_to_load:embed_pipes.append(nlu.load(embedder))
    #
    #     if generate_code_sample:st.code(get_code_for_viz('SIMILARITY',[StreamlitUtilsOS.extract_name(p) for p  in embed_pipes],default_texts))
    #
    #     StreamlitVizTracker.loaded_word_embeding_pipes+=embed_pipes
    #     similarity_metrics = {}
    #     embed_vector_info = {}
    #     cols_full = True
    #     col_index=0
    #     for p in embed_pipes :
    #         data1 = p.predict(text1,output_level='token').dropna()
    #         data2 = p.predict(text2,output_level='token').dropna()
    #         e_coms = StreamlitUtilsOS.find_all_embed_components(p)
    #         modelhub_links = [ModelHubUtils.get_url_by_nlu_refrence(c.info.nlu_ref) if hasattr(c.info,'nlu_ref') else ModelHubUtils.get_url_by_nlu_refrence('') for c in e_coms]
    #         e_cols = StreamlitUtilsOS.get_embed_cols(p)
    #         for num_emb,e_col in enumerate(e_cols):
    #             if col_index == num_cols-1 :cols_full=True
    #             if cols_full :
    #                 cols = st.beta_columns(num_cols)
    #                 col_index = 0
    #                 cols_full = False
    #             else:col_index+=1
    #             tok1 = data1['token']
    #             tok2 = data2['token']
    #             emb1 = data1[e_col]
    #             emb2 = data2[e_col]
    #             embed_mat1 = np.array([x for x in emb1])
    #             embed_mat2 = np.array([x for x in emb2])
    #             # e_name = e_col.split('word_embedding_')[-1]
    #             e_name = e_coms[num_emb].info.nlu_ref if hasattr(e_coms[num_emb].info,'nlu_ref') else e_col.split('word_embedding_')[-1] if 'en.' in e_col else e_col
    #             e_name = e_name.split('embed.')[-1] if 'en.' in e_name else e_name
    #             if 'ner' in e_name : e_name = loaded_storage_refs[num_emb]
    #
    #             embed_vector_info[e_name]= {"Vector Dimension ":embed_mat1.shape[1],
    #                                         "Num Vectors":embed_mat1.shape[0] + embed_mat1.shape[0],
    #                                         "NLU_reference":e_coms[num_emb].info.nlu_ref if hasattr(e_coms[num_emb].info,'nlu_ref') else ' ',
    #                                         "Spark_NLP_reference":ModelHubUtils.NLU_ref_to_NLP_ref(e_coms[num_emb].info.nlu_ref if hasattr(e_coms[num_emb].info,'nlu_ref') else ' '),
    #                                         "Storage Reference":loaded_storage_refs[num_emb],
    #                                         'Modelhub info': modelhub_links[num_emb]}
    #             for dist_algo in dist_algo_selection:
    #                 # scalar_similarities[e_col][dist_algo]={}
    #                 sim_score = dist_metric_algos[dist_algo](embed_mat1,embed_mat2)
    #                 sim_score = pd.DataFrame(sim_score)
    #                 sim_score.index   = tok1.values
    #                 sim_score.columns = tok2.values
    #                 sim_score.columns = StreamlitVizTracker.pad_duplicate_tokens(list(sim_score.columns))
    #                 sim_score.index   = StreamlitVizTracker.pad_duplicate_tokens(list(sim_score.index))
    #                 if write_raw_pandas :st.write(sim_score,key = key)
    #                 if sim_score.shape == (1,1) :
    #                     sim_score = sim_score.iloc[0][0]
    #                     sim_score = round(sim_score,2)
    #                     if sim_score > threshold:
    #                         st.success(sim_score)
    #                         st.success(f'Scalar Similarity={sim_score} for distance metric={dist_algo}')
    #                         st.error('No similarity matrix for only 2 tokens. Try entering at least 1 sentences in a field')
    #                     else:
    #                         st.error(f'Scalar Similarity={sim_score} for distance metric={dist_algo}')
    #                 else :
    #                     ploty_avaiable = True
    #                     # for tok emb, sum rows and norm by rows, then sum cols and norm by cols to generate a scalar from matrix
    #                     scalar_sim_score  = np.sum((np.sum(sim_score,axis=0) / sim_score.shape[0])) / sim_score.shape[1]
    #                     scalar_sim_score = round(scalar_sim_score,2)
    #
    #                     if display_scalar_similarities:
    #                         if scalar_sim_score > threshold:st.success(f'Scalar Similarity :{scalar_sim_score} for distance metric={dist_algo}')
    #                         else: st.error(f'Scalar Similarity :{scalar_sim_score} for embedder={e_col} distance metric={dist_algo}')
    #                     if similarity_matrix:
    #                         if ploty_avaiable :
    #                             fig = px.imshow(sim_score, labels=dict(color="similarity"))#, title=f'Simmilarity Matrix for embedding_model={e_name} distance metric={dist_algo}')
    #                             # st.write(fig,key =key)
    #                             similarity_metrics[f'{e_name}_{dist_algo}_similarity']={
    #                                 'scalar_similarity' : scalar_sim_score,
    #                                 'dist_metric' : dist_algo,
    #                                 'embedding_model': e_name,
    #                                 'modelhub_info' : modelhub_links[num_emb],
    #                             }
    #                             subh = f"""Embedding-Model=`{e_name}`, Similarity-Score=`{scalar_sim_score}`,  distance metric=`{dist_algo}`"""
    #                             cols[col_index].markdown(subh)
    #                             cols[col_index].write(fig, key=key)
    #                         else : pass # todo fallback plots
    #
    #     if display_similarity_summary:
    #         exp = st.beta_expander("Similarity summary")
    #         exp.write(similarity_metrics)
    #     if display_embed_information:
    #         exp = st.beta_expander("Embedding vector information")
    #         exp.write(embed_vector_info)
    #
    #     if show_infos :
    #         # VizUtilsStreamlitOS.display_infos()
    #         StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
    #         StreamlitVizTracker.display_footer()
    #
    #
    #
    #
    # @staticmethod
    # def display_low_dim_embed_viz_token(
    #         pipe, # nlu pipe
    #         default_texts: List[str] = ("Donald Trump likes to party!", "Angela Merkel likes to party!", 'Peter HATES TO PARTTY!!!! :('),
    #         title: Optional[str] = "Lower dimensional Manifold visualization for word embeddings",
    #         sub_title: Optional[str] = "Apply any of the 11 `Manifold` or `Matrix Decomposition` algorithms to reduce the dimensionality of `Word Embeddings` to `1-D`, `2-D` and `3-D` ",
    #         write_raw_pandas : bool = False ,
    #         default_applicable_algos : List[str] = ('TSNE','PCA',),
    #         applicable_algos : List[str] = ("TSNE", "PCA"),#,'LLE','Spectral Embedding','MDS','ISOMAP','SVD aka LSA','DictionaryLearning','FactorAnalysis','FastICA','KernelPCA',),  # LatentDirichletAllocation 'NMF',
    #         target_dimensions : List[int] = (1,2,3),
    #         show_algo_select : bool = True,
    #         show_embed_select : bool = True,
    #         show_color_select: bool = True,
    #         MAX_DISPLAY_NUM:int=100,
    #         display_embed_information:bool=True,
    #         set_wide_layout_CSS:bool=True,
    #         num_cols: int = 3,
    #         model_select_position:str = 'side', # side or main
    #         key:str = "NLU_streamlit",
    #         additional_classifiers_for_coloring:List[str]=['pos', 'sentiment'],
    #         extra_NLU_models_for_hueing: List[str] = ('pos','sentiment'),
    #         generate_code_sample:bool = False,
    #         show_infos:bool = True,
    #         show_logo:bool = True,
    # ):
    #     # TODO dynamic columns infer for mouse over, TOKEN LEVEL FEATURS APPLICABLE!!!!!
    #     # NIOT CRASH [1], [a b], [ab]
    #     # todo dynamic deduct Tok vs Sent vs Doc vs Chunk embeds
    #     # todo selectable color features
    #     # todo selectable mouseover features
    #     from nlu.pipe.viz.streamlit_viz.streamlit_utils_OS import StreamlitUtilsOS
    #
    #     # VizUtilsStreamlitOS.footer_displayed=False
    #     try :
    #         import plotly.express as px
    #         from sklearn.metrics.pairwise import distance_metrics
    #     except :st.error("You need the sklearn and plotly package in your Python environment installed for similarity visualizations. Run <pip install sklearn plotly>")
    #     if len(default_texts) > MAX_DISPLAY_NUM : default_texts = default_texts[:MAX_DISPLAY_NUM]
    #     if set_wide_layout_CSS : _set_block_container_style()
    #     if title:st.header(title)
    #     if sub_title:st.subheader(sub_title)
    #     # if show_logo :VizUtilsStreamlitOS.show_logo()
    #
    #     # VizUtilsStreamlitOS.loaded_word_embeding_pipes = []
    #     loaded_word_embeding_pipes = []
    #
    #
    #     data = st.text_area('Enter N texts, seperated by new lines to visualize Word Embeddings for ','\n'.join(default_texts))
    #     data = data.split("\n")
    #     while '' in data : data.remove('')
    #     if len(data)<=1:
    #         st.error("Please enter more than 2 lines of text, seperated by new lines (hit <ENTER>)")
    #         return
    #     else : algos = default_applicable_algos
    #     # TODO dynamic color inference for plotting??
    #     if show_color_select: feature_to_color_by =  st.selectbox('Feature to color plots by ',['pos','sentiment',],0)
    #     text_col = 'token'
    #     embed_algos_to_load = []
    #     embed_pipes = [pipe]
    #     e_coms = StreamlitUtilsOS.find_all_embed_components(pipe)
    #
    #     if show_algo_select :
    #         exp = st.beta_expander("Select dimension reduction technique to apply")
    #         algos = exp.multiselect(
    #             "Reduce embedding dimensionality to something visualizable",
    #             options=("TSNE", "ISOMAP",'LLE','Spectral Embedding','MDS','PCA','SVD aka LSA','DictionaryLearning','FactorAnalysis','FastICA','KernelPCA',),default=applicable_algos,)
    #
    #         emb_components_usable = [e for e in Discoverer.get_components('embed',True, include_aliases=True) if 'chunk' not in e and 'sentence' not in e]
    #         loaded_embed_nlu_refs = []
    #         loaded_classifier_nlu_refs = []
    #         loaded_storage_refs = []
    #         for c in e_coms :
    #             if not  hasattr(c.info,'nlu_ref'): continue
    #             r = c.info.nlu_ref
    #             if 'en.' not in r and 'embed.' not  in r and 'ner' not in r : loaded_embed_nlu_refs.append('en.embed.' + r)
    #             elif 'en.'  in r and 'embed.' not  in r  and 'ner' not in r:
    #                 r = r.split('en.')[0]
    #                 loaded_embed_nlu_refs.append('en.embed.' + r)
    #             else :
    #                 loaded_embed_nlu_refs.append(StorageRefUtils.extract_storage_ref(c))
    #             loaded_storage_refs.append(StorageRefUtils.extract_storage_ref(c))
    #
    #         for p in StreamlitVizTracker.loaded_word_embeding_pipes : loaded_embed_nlu_refs.append(p.nlu_ref)
    #         loaded_embed_nlu_refs = list(set(loaded_embed_nlu_refs))
    #         for l in loaded_embed_nlu_refs:
    #             if l not in emb_components_usable : emb_components_usable.append(l)
    #         emb_components_usable.sort()
    #         loaded_embed_nlu_refs.sort()
    #         if model_select_position =='side':
    #             embed_algo_selection   = st.sidebar.multiselect("Pick additional Word Embeddings for the Dimension Reduction",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
    #         else :
    #             exp = st.beta_expander("Pick additional Word Embeddings")
    #             embed_algo_selection   = exp.multiselect("Pick additional Word Embeddings for the Dimension Reduction",options=emb_components_usable,default=loaded_embed_nlu_refs,key = key)
    #         embed_algos_to_load = list(set(embed_algo_selection) - set(loaded_embed_nlu_refs))
    #
    #     for embedder in embed_algos_to_load:embed_pipes.append(nlu.load(embedder + f' {" ".join(additional_classifiers_for_coloring)}'))
    #     StreamlitVizTracker.loaded_word_embeding_pipes+=embed_pipes
    #
    #     # TODO load/update classifier pipes
    #     for nlu_ref in additional_classifiers_for_coloring :
    #         already_loaded=False
    #         if 'pos' in nlu_ref : continue
    #             # for p in  VizUtilsStreamlitOS.loaded_document_classifier_pipes:
    #             #     if p.nlu_ref == nlu_ref : already_loaded = True
    #             # if not already_loaded : VizUtilsStreamlitOS.loaded_token_level_classifiers.append(nlu.load(nlu_ref))
    #         else :
    #             for p in  StreamlitVizTracker.loaded_document_classifier_pipes:
    #                 if p.nlu_ref == nlu_ref : already_loaded = True
    #             if not already_loaded : StreamlitVizTracker.loaded_document_classifier_pipes.append(nlu.load(nlu_ref))
    #
    #     col_index = 0
    #     cols = st.beta_columns(num_cols)
    #     def are_cols_full(): return col_index == num_cols
    #     token_feature_pipe = StreamlitUtilsOS.get_pipe('en.dep.typed')
    #     ## TODO , not all pipes have sentiment/pos etc.. models for hueing loaded....
    #     ## Lets FIRST predict with the classifiers/Token level feature generators and THEN apply embed pipe??
    #     for p in StreamlitVizTracker.loaded_word_embeding_pipes :
    #         # TODO, run all classifiers pipes. FOr Sentence/Doc level stuff, we can only use Senc/Doc/Input dependent level annotators
    #         #  TODO token features TYPED DEP/ UNTYPED DEP/ POS  ---> LOAD DEP/UNTYPED DEP/ POS and then APPEN NLU_COMPONENTS!!!!! TO EXISTING PIPE
    #         classifier_cols = []
    #
    #         for class_p in StreamlitVizTracker.loaded_document_classifier_pipes:
    #             data = class_p.predict(data, output_level='document').dropna()
    #             classifier_cols.append(StreamlitUtilsOS.get_classifier_cols(class_p))
    #
    #         p = StreamlitUtilsOS.merge_token_classifiers_with_embed_pipe(p, token_feature_pipe)
    #         predictions =   p.predict(data,output_level='token').dropna()
    #         e_col = StreamlitUtilsOS.find_embed_col(predictions)
    #         e_com = StreamlitUtilsOS.find_embed_component(p)
    #         embedder_name = StreamlitUtilsOS.extract_name(e_com)
    #         emb = predictions[e_col]
    #         mat = np.array([x for x in emb])
    #         for algo in algos :
    #             if len(mat.shape)>2 : mat =mat.reshape(len(emb),mat.shape[-1])
    #
    #             # calc reduced dimensionality with every algo
    #             #todo  try/catch block for embed failures?
    #             if 1 in target_dimensions:
    #                 low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo,1).fit_transform(mat)
    #                 x = low_dim_data[:,0]
    #                 y = np.zeros(low_dim_data[:,0].shape)
    #                 tsne_df =  pd.DataFrame({'x':x,'y':y, 'text':predictions[text_col], 'pos':predictions.pos, 'sentiment' : predictions.sentiment})
    #                 fig = px.scatter(tsne_df, x="x", y="y",color=feature_to_color_by, hover_data=['token','text','sentiment', 'pos'])
    #                 subh = f"""Word-Embeddings =`{embedder_name}`, Manifold-Algo =`{algo}` for `D=1`"""
    #                 cols[col_index].markdown(subh)
    #                 cols[col_index].write(fig,key=key)
    #                 col_index+=1
    #                 if are_cols_full() :
    #                     cols = st.beta_columns(num_cols)
    #                     col_index = 0
    #             if 2 in target_dimensions:
    #                 low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo,2).fit_transform(mat)
    #                 x = low_dim_data[:,0]
    #                 y = low_dim_data[:,1]
    #                 tsne_df =  pd.DataFrame({'x':x,'y':y, 'text':predictions[text_col], 'pos':predictions.pos, 'sentiment':predictions.sentiment, })
    #                 fig = px.scatter(tsne_df, x="x", y="y",color=feature_to_color_by, hover_data=['text'])
    #                 subh = f"""Word-Embeddings =`{embedder_name}`, Manifold-Algo =`{algo}` for `D=2`"""
    #                 cols[col_index].markdown(subh)
    #                 cols[col_index].write(fig,key=key)
    #                 # st.write(fig)
    #                 col_index+=1
    #                 if are_cols_full() :
    #                     cols = st.beta_columns(num_cols)
    #                     col_index = 0
    #             if 3 in target_dimensions:
    #                 low_dim_data = StreamlitUtilsOS.get_manifold_algo(algo,3).fit_transform(mat)
    #                 x = low_dim_data[:,0]
    #                 y = low_dim_data[:,1]
    #                 z = low_dim_data[:,2]
    #                 tsne_df =  pd.DataFrame({'x':x,'y':y,'z':z, 'text':predictions[text_col], 'pos':predictions.pos, 'sentiment':predictions.sentiment, })
    #                 fig = px.scatter_3d(tsne_df, x="x", y="y", z='z',color=feature_to_color_by, hover_data=['text'])
    #                 subh = f"""Word-Embeddings =`{embedder_name}`, Manifold-Algo =`{algo}` for `D=3`"""
    #                 cols[col_index].markdown(subh)
    #                 cols[col_index].write(fig,key=key)
    #
    #                 # st.write(fig)
    #                 col_index+=1
    #                 if are_cols_full() :
    #                     cols = st.beta_columns(num_cols)
    #                     col_index = 0
    #         # Todo fancy embed infos etc
    #         # if display_embed_information: display_embed_vetor_information(e_com,mat)
    #
    #
