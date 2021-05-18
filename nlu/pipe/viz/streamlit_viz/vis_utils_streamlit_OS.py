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