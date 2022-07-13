from sparknlp_display import NerVisualizer, DependencyParserVisualizer
from sparknlp.annotator import NerConverter, DependencyParserModel, TypedDependencyParserModel, PerceptronModel
from sparknlp.base import DocumentAssembler
from nlu.universe.feature_node_ids import NLP_NODE_IDS

class VizUtilsOS():
    """Utils for interfacing with the Spark-NLP-Display lib and vizzing Open Source Components - Open source"""
    HTML_WRAPPER = """<div class="scroll entities" style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem; white-space:pre-wrap">{}</div>"""

    @staticmethod
    def infer_viz_open_source(pipe) -> str:
        """For a given NLUPipeline with only open source components, infers which visualizations are applicable. """
        for c in pipe.components:
            if isinstance(c.model, NerConverter): return 'ner'
            if isinstance(c.model, DependencyParserModel): return 'dep'

    @staticmethod
    def viz_ner(anno_res, pipe, labels=None, viz_colors={}, is_databricks_env=False, write_to_streamlit=False,
                streamlit_key='RANDOM',  user_ner_col=None):
        """Infer columns required for ner viz and then viz it.
        viz_colors :  set label colors by specifying hex codes , i.e. viz_colors =  {'LOC':'#800080', 'PER':'#77b5fe'}
        labels : only allow these labels to be displayed. (default: [] - all labels will be displayed)
        """
        document_col, entities_col = VizUtilsOS.infer_ner_dependencies(pipe)
        ner_vis = NerVisualizer()
        ner_vis.set_label_colors(viz_colors)
        if user_ner_col :
            entities_col = user_ner_col
        if write_to_streamlit:
            import streamlit as st
            HTML = ner_vis.display(anno_res, label_col=entities_col, document_col=document_col, labels=labels,
                                   return_html=True)
            CSS, HTML = HTML.split('</style>')
            CSS = CSS + '</style>'
            HTML = f'<div> {HTML} '
            st.markdown(CSS, unsafe_allow_html=True)
            st.markdown(VizUtilsOS.HTML_WRAPPER.format(HTML), unsafe_allow_html=True)

        elif not is_databricks_env:
            ner_vis.display(anno_res, label_col=entities_col, document_col=document_col, labels=labels)
        else:
            return ner_vis.display(anno_res, label_col=entities_col, document_col=document_col, labels=labels,
                                   return_html=True)

    @staticmethod
    def infer_ner_dependencies(pipe):
        """Finds entities and doc cols for ner viz"""
        # TODO FIX
        doc_component = None
        entities_component = None
        for c in pipe.components:
            if isinstance(c.model, NerConverter):        entities_component = c
            if isinstance(c.model, DocumentAssembler):   doc_component = c

        document_col = doc_component.spark_output_column_names[0]
        entities_col = entities_component.spark_output_column_names[0]
        return document_col, entities_col

    @staticmethod
    def viz_dep(anno_res, pipe, is_databricks_env, write_to_streamlit, streamlit_key='RANDOM',
                user_pos_col=None, user_dep_untyped_col=None, user_dep_typed_col=None
                ):
        """Viz dep result"""
        pos_col, dep_typ_col, dep_untyp_col = VizUtilsOS.infer_dep_dependencies(pipe)
        dependency_vis = DependencyParserVisualizer()
        if user_pos_col :
            pos_col = user_pos_col

        if user_dep_typed_col:
            dep_typ_col = user_dep_typed_col

        if user_dep_untyped_col:
            dep_untyp_col = dep_untyp_col

        if write_to_streamlit:
            import streamlit as st
            SVG = dependency_vis.display(anno_res, pos_col=pos_col, dependency_col=dep_untyp_col,
                                         dependency_type_col=dep_typ_col, return_html=True)
            # st.markdown(SVG, unsafe_allow_html=True)
            st.markdown(VizUtilsOS.HTML_WRAPPER.format(SVG), unsafe_allow_html=True)

        elif not is_databricks_env:
            dependency_vis.display(anno_res, pos_col=pos_col, dependency_col=dep_untyp_col,
                                   dependency_type_col=dep_typ_col)
        else:
            return dependency_vis.display(anno_res, pos_col=pos_col, dependency_col=dep_untyp_col,
                                          dependency_type_col=dep_typ_col, return_html=True)

    @staticmethod
    def infer_dep_dependencies(pipe):
        """Finds entities,pos,dep_typed,dep_untyped and  doc cols for dep viz viz"""
        # doc_component      = None
        pos_component = None
        dep_untyped_component = None
        dep_typed_component = None
        for c in pipe.components:
            if c.name == NLP_NODE_IDS.POS :
                pos_component = c
            if c.name == NLP_NODE_IDS.TYPED_DEPENDENCY_PARSER :
                dep_typed_component = c
            if c.name == NLP_NODE_IDS.UNTYPED_DEPENDENCY_PARSER :
                dep_untyped_component = c

        pos_col = pos_component.spark_output_column_names[0]
        dep_typ_col = dep_typed_component.spark_output_column_names[0]
        dep_untyp_col = dep_untyped_component.spark_output_column_names[0]
        return pos_col, dep_typ_col, dep_untyp_col
