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
class TokenFeaturesStreamlitBlock():
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
    ) -> None:
        """Visualizer for token features."""
        StreamlitVizTracker.footer_displayed=False
        if show_logo :StreamlitVizTracker.show_logo()
        if set_wide_layout_CSS : _set_block_container_style()
        if title:st.header(title)
        # if generate_code_sample: st.code(get_code_for_viz('TOKEN',StreamlitUtilsOS.extract_name(pipe),text))
        if sub_title:st.subheader(sub_title)
        token_pipes = [pipe]
        if show_text_input : text = st.text_area("Enter text you want to view token features for", text, key=key)
        if show_model_select :
            token_pipes_components_usable = [e for e in Discoverer.get_components(get_all=True)]
            loaded_nlu_refs = [c.info.nlu_ref for c in pipe.components]

            for l in loaded_nlu_refs:
                if 'converter' in l :
                    loaded_nlu_refs.remove(l)
                    continue
                if l not in token_pipes_components_usable : token_pipes_components_usable.append(l)
            token_pipes_components_usable = list(set(token_pipes_components_usable))
            loaded_nlu_refs = list(set(loaded_nlu_refs))
            if '' in loaded_nlu_refs : loaded_nlu_refs.remove('')
            if ' ' in loaded_nlu_refs : loaded_nlu_refs.remove(' ')
            token_pipes_components_usable.sort()
            loaded_nlu_refs.sort()
            if model_select_position =='side':model_selection   = st.sidebar.multiselect("Pick any additional models for token features",options=token_pipes_components_usable,default=loaded_nlu_refs,key = key)
            else:model_selection   = st.multiselect("Pick any additional models for token features",options=token_pipes_components_usable,default=loaded_nlu_refs,key = key)
            # else : ValueError("Please define model_select_position as main or side")
            models_to_load = list(set(model_selection) - set(loaded_nlu_refs))
            for model in models_to_load:token_pipes.append(nlu.load(model))
            StreamlitVizTracker.loaded_token_pipes+= token_pipes
        if generate_code_sample:st.code(get_code_for_viz('TOKEN',[StreamlitUtilsOS.extract_name(p) for p  in token_pipes],text))
        dfs = []
        for p in token_pipes:
            df = p.predict(text, output_level=output_level, metadata=full_metadata,positions=positions)
            dfs.append(df)


        df = pd.concat(dfs,axis=1)
        df = df.loc[:,~df.columns.duplicated()]
        if show_feature_select :
            exp = st.beta_expander("Select token features to display")
            features = exp.multiselect(
                "Token features",
                options=list(df.columns),
                default=list(df.columns)
            )
        st.dataframe(df[features])
        if show_infos :
            # VizUtilsStreamlitOS.display_infos()
            StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
            StreamlitVizTracker.display_footer()

