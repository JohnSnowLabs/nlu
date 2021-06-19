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
class NERStreamlitBlock():
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

    ):
        StreamlitVizTracker.footer_displayed=False
        if set_wide_layout_CSS : _set_block_container_style()
        if show_logo :StreamlitVizTracker.show_logo()
        if show_model_select :
            model_selection = Discoverer.get_components('ner',include_pipes=True)
            model_selection.sort()
            if model_select_position == 'side':ner_model_2_viz = st.sidebar.selectbox("Select a NER model",model_selection,index=model_selection.index(pipe.nlu_ref.split(' ')[0]))
            else : ner_model_2_viz = st.selectbox("Select a NER model",model_selection,index=model_selection.index(pipe.nlu_ref.split(' ')[0]))
            pipe = pipe if pipe.nlu_ref == ner_model_2_viz else StreamlitUtilsOS.get_pipe(ner_model_2_viz)
        if title: st.header(title)
        if show_text_input : text = st.text_area("Enter text you want to visualize NER classes for below", text, key=key)
        if sub_title : st.subheader(sub_title)
        if generate_code_sample: st.code(get_code_for_viz('NER',StreamlitUtilsOS.extract_name(pipe),text))
        if ner_tags is None: ner_tags = StreamlitUtilsOS.get_NER_tags_in_pipe(pipe)

        if not show_color_selector :
            if show_label_select:
                exp = st.beta_expander("Select entity labels to highlight")
                label_select = exp.multiselect(
                    "These labels are predicted by the NER model. Select which ones you want to display",
                    options=ner_tags,default=list(ner_tags))
            else : label_select = ner_tags
            pipe.viz(text,write_to_streamlit=True, viz_type='ner',labels_to_viz=label_select,viz_colors=colors, streamlit_key=key)
        else : # TODO WIP color select
            cols = st.beta_columns(3)
            exp = cols[0].beta_expander("Select entity labels to display")
            color = st.color_picker('Pick A Color', '#00f900',key = key)
            color = cols[2].color_picker('Pick A Color for a specific entity label', '#00f900',key = key)
            tag2color = cols[1].selectbox('Pick a ner tag to color', ner_tags,key = key)
            colors[tag2color]=color
        if show_table : st.write(pipe.predict(text, output_level='chunk'),key = key)

        if show_infos :
            # VizUtilsStreamlitOS.display_infos()
            StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes = [pipe])
            StreamlitVizTracker.display_footer()

