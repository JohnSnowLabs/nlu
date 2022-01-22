from typing import Optional
import streamlit as st
from nlu.pipe.viz.streamlit_viz.streamlit_utils_OS import StreamlitUtilsOS
from nlu.pipe.viz.streamlit_viz.gen_streamlit_code import get_code_for_viz
from nlu.pipe.viz.streamlit_viz.styles import _set_block_container_style
from nlu.pipe.viz.streamlit_viz.streamlit_viz_tracker import StreamlitVizTracker


class DepTreeStreamlitBlock():

    @staticmethod
    def visualize_dep_tree(
            pipe,  # nlu component_list
            text: str = 'Billy likes to swim',
            title: Optional[str] = "Dependency Parse & Part-of-speech tags",
            sub_title: Optional[
                str] = 'POS tags define a `grammatical label` for `each token` and the `Dependency Tree` classifies `Relations between the tokens` ',
            set_wide_layout_CSS: bool = True,
            generate_code_sample: bool = False,
            key="NLU_streamlit_dep_tree",
            show_infos: bool = True,
            show_logo: bool = True,
            show_text_input: bool = True,
    ):
        StreamlitVizTracker.footer_displayed = False
        if show_logo: StreamlitVizTracker.show_logo()
        if set_wide_layout_CSS: _set_block_container_style()
        if title: st.header(title)
        if show_text_input:
            text = st.text_area("Enter text you want to visualize dependency tree for ", text, key=key)
        if sub_title: st.subheader(sub_title)
        if generate_code_sample: st.code(get_code_for_viz('TREE', StreamlitUtilsOS.extract_name(pipe), text))
        if isinstance(text, str) and '\n' in text: text = text.split('\n')
        pipe.viz(text, write_to_streamlit=True, viz_type='dep', streamlit_key=key)
        if show_infos:
            # VizUtilsStreamlitOS.display_infos()
            StreamlitVizTracker.display_model_info(pipe.nlu_ref, pipes=[pipe])
            StreamlitVizTracker.display_footer()
