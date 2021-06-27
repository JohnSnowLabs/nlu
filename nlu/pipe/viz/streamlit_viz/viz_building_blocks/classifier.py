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
class ClassifierStreamlitBlock():

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
    )->None:
        if show_logo :StreamlitVizTracker.show_logo()
        if set_wide_layout_CSS : _set_block_container_style()
        if title:st.header(title)
        if sub_title:st.subheader(sub_title)

        # if generate_code_sample: st.code(get_code_for_viz('CLASSES',StreamlitUtilsOS.extract_name(pipe),text))
        if not isinstance(text, (pd.DataFrame, pd.Series)):
            text = st.text_area('Enter N texts, seperated by new lines to view classification results for','\n'.join(text) if isinstance(text,list) else text, key=key)
            text = text.split("\n")
            while '' in text : text.remove('')
        classifier_pipes = [pipe]
        classifier_components_usable = [e for e in Discoverer.get_components('classify',True, include_aliases=True)]
        classifier_components = StreamlitUtilsOS.find_all_classifier_components(pipe)
        loaded_classifier_nlu_refs = [c.info.nlu_ref for c in classifier_components ]

        for l in loaded_classifier_nlu_refs:
            if 'converter' in l :
                loaded_classifier_nlu_refs.remove(l)
                continue
            if l not in classifier_components_usable : classifier_components_usable.append(l)

        classifier_components_usable.sort()
        loaded_classifier_nlu_refs.sort()
        for r in loaded_classifier_nlu_refs:
            if r not in  classifier_components_usable : loaded_classifier_nlu_refs.remove(r)
        if show_model_selector :
            if model_select_position =='side':classifier_components_selection   = st.sidebar.multiselect("Pick additional Classifiers",options=classifier_components_usable,default=loaded_classifier_nlu_refs,key = key)
            else:classifier_components_selection   = st.multiselect("Pick additional Classifiers",options=classifier_components_usable,default=loaded_classifier_nlu_refs,key = key)
        # else : ValueError("Please define model_select_position as main or side")
        classifier_algos_to_load = list(set(classifier_components_selection) - set(loaded_classifier_nlu_refs))
        for classifier in classifier_algos_to_load:classifier_pipes.append(nlu.load(classifier))
        StreamlitVizTracker.loaded_document_classifier_pipes+= classifier_pipes
        if generate_code_sample:st.code(get_code_for_viz('CLASSES',[StreamlitUtilsOS.extract_name(p) for p  in classifier_pipes],text))

        dfs = []
        all_classifier_cols=[]
        for p in classifier_pipes :
            df = p.predict(text, output_level=output_level, metadata=metadata, positions=positions)
            classifier_cols = StreamlitUtilsOS.get_classifier_cols(p)
            for c in classifier_cols :
                if c not in df.columns : classifier_cols.remove(c)

            if 'text' in df.columns: classifier_cols += ['text']
            elif 'document' in df.columns: classifier_cols += ['document']
            all_classifier_cols+= classifier_cols
            dfs.append(df)
        df = pd.concat(dfs, axis=1)
        df = df.loc[:,~df.columns.duplicated()]
        for c in all_classifier_cols :
            if c not in df.columns : all_classifier_cols.remove(c)
        all_classifier_cols = list(set(all_classifier_cols))

        if len(all_classifier_cols) == 0: st.error('No classes detected')
        else :st.write(df[all_classifier_cols],key=key)
        if show_infos :
            # VizUtilsStreamlitOS.display_infos()
            StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
            StreamlitVizTracker.display_footer()


