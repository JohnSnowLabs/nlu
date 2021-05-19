from sparknlp_display import NerVisualizer,DependencyParserVisualizer
from sparknlp.annotator import NerConverter,DependencyParserModel, TypedDependencyParserModel, PerceptronModel
from sparknlp.base import  DocumentAssembler
from typing import List, Sequence, Tuple, Optional, Dict, Union, Callable
import streamlit as st
class VizUtilsStreamlitOS():
    """Utils for displaying various NLU viz to streamlit"""
    @staticmethod
    def infer_viz_open_source(pipe)->str:
        """For a given NLUPipeline with only open source components, infers which visualizations are applicable. """
        for c in pipe.components:
            if isinstance(c.model, NerConverter) : return 'ner'
            if isinstance(c.model, DependencyParserModel) : return 'dep'
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