from sparknlp_jsl.annotator import *
from sparknlp.base import *
from sparknlp_display import *
class VizUtilsHC():
    """Utils for interfacing with the Spark-NLP-Display lib - licensed Viz"""
    HTML_WRAPPER = """<div class="scroll entities" style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem; white-space:pre-wrap">{}</div>"""
    @staticmethod
    def infer_viz_licensed(pipe)->str:
        """For a given NLUPipeline with licensed components, infers which visualizations are applicable. """
        # we go in reverse, which makes NER always take lowest priority and NER feeder annotators have higher priority
        for c in pipe.components[::-1]:
            if isinstance(c.model, TypedDependencyParserModel) : return 'dep'
            if isinstance(c.model, (SentenceEntityResolverModel,ChunkEntityResolverModel)) : return 'resolution'
            if isinstance(c.model, (RelationExtractionDLModel,RelationExtractionDLModel)) : return 'relation'
            if isinstance(c.model, (AssertionDLModel,AssertionLogRegModel)) : return 'assert'
            if isinstance(c.model, (NerConverter,NerConverterInternal)) : return 'ner'


    @staticmethod
    def viz_ner(anno_res, pipe,labels = [] ,  viz_colors={},is_databricks_env =False,write_to_streamlit=False):
        """Infer columns required for ner viz and then viz it.
        viz_colors :  set label colors by specifying hex codes , i.e. viz_colors =  {'LOC':'#800080', 'PER':'#77b5fe'}
        labels : only allow these labels to be displayed. (default: [] - all labels will be displayed)
        """
        document_col,entities_col =  VizUtilsHC.infer_ner_dependencies(pipe)
        ner_vis = NerVisualizer()
        if len(viz_colors) > 0 : ner_vis.set_label_colors(viz_colors)
        if write_to_streamlit :
            import streamlit as st
            HTML = ner_vis.display(anno_res,label_col=entities_col,document_col=document_col, labels=labels, return_html=True )
            CSS,HTML = HTML.split('</style>')
            CSS = CSS + '</style>'
            HTML = f'<div> {HTML} '
            st.markdown(CSS, unsafe_allow_html=True)
            # st.markdown(HTML, unsafe_allow_html=True)
            st.markdown(VizUtilsHC.HTML_WRAPPER.format(HTML), unsafe_allow_html=True)

        elif not is_databricks_env:
            ner_vis.display(anno_res,label_col=entities_col,document_col=document_col, labels=labels )
        else :
            return ner_vis.display(anno_res,label_col=entities_col,document_col=document_col, labels=labels,return_html=True )


    @staticmethod
    def infer_ner_dependencies(pipe):
        """Finds entities and doc cols for ner viz"""
        doc_component      = None
        entities_component = None
        for c in pipe.components:
            if isinstance(c.model, (NerConverter,NerConverterInternal)) :        entities_component  = c
            if isinstance(c.model, DocumentAssembler) :   doc_component = c

        document_col     = doc_component.info.outputs[0]
        entities_col = entities_component.info.outputs[0]
        return document_col, entities_col


    @staticmethod
    def viz_dep(anno_res,pipe,is_databricks_env,write_to_streamlit=False):
        """Viz dep result"""
        pos_col,dep_typ_col,dep_untyp_col  = VizUtilsHC.infer_dep_dependencies(pipe)
        dependency_vis = DependencyParserVisualizer()
        if write_to_streamlit :
            import streamlit as st
            SVG = dependency_vis.display(anno_res,pos_col =pos_col,dependency_col =  dep_untyp_col ,dependency_type_col = dep_typ_col,return_html=True)
            # st.markdown(SVG, unsafe_allow_html=True)
            st.markdown(VizUtilsHC.HTML_WRAPPER.format(SVG), unsafe_allow_html=True)
        elif not is_databricks_env:
            dependency_vis.display(anno_res,pos_col =pos_col,dependency_col =  dep_untyp_col ,dependency_type_col = dep_typ_col)
        else:
            return dependency_vis.display(anno_res,pos_col =pos_col,dependency_col =  dep_untyp_col ,dependency_type_col = dep_typ_col,return_html=True)

    @staticmethod
    def infer_dep_dependencies(pipe):
        """Finds entities,pos,dep_typed,dep_untyped and  doc cols for dep viz viz"""
        # doc_component      = None
        pos_component = None
        dep_untyped_component = None
        dep_typed_component = None
        for c in pipe.components:
            if isinstance(c.model, PerceptronModel) :              pos_component  = c
            if isinstance(c.model, TypedDependencyParserModel) :   dep_typed_component  = c
            if isinstance(c.model, DependencyParserModel) :        dep_untyped_component  = c

        pos_col       = pos_component.info.outputs[0]
        dep_typ_col   = dep_typed_component.info.outputs[0]
        dep_untyp_col = dep_untyped_component.info.outputs[0]
        return pos_col,dep_typ_col,dep_untyp_col




    @staticmethod
    def viz_resolution(anno_res,pipe,viz_colors={},is_databricks_env=False,write_to_streamlit=False):
        """Viz dep result. Set label colors by specifying hex codes, i.e. viz_colors={'TREATMENT':'#800080', 'PROBLEM':'#77b5fe'} """
        entities_col,resolution_col,doc_col  = VizUtilsHC.infer_resolution_dependencies(pipe)
        er_vis = EntityResolverVisualizer()
        if len(viz_colors) > 0 : er_vis.set_label_colors(viz_colors)

        if write_to_streamlit :
            import streamlit as st
            HTML = er_vis.display(anno_res,label_col=entities_col, resolution_col = resolution_col,document_col=doc_col,return_html=True)
            CSS,HTML = HTML.split('</style>')
            CSS = CSS + '</style>'
            HTML = f'<div> {HTML} '
            st.markdown(CSS, unsafe_allow_html=True)
            # st.markdown(HTML, unsafe_allow_html=True)
            st.markdown(VizUtilsHC.HTML_WRAPPER.format(HTML), unsafe_allow_html=True)


        elif not is_databricks_env:
            er_vis.display(anno_res,label_col=entities_col, resolution_col = resolution_col,document_col=doc_col)
        else:
            return  er_vis.display(anno_res,label_col=entities_col, resolution_col = resolution_col,document_col=doc_col,return_html=True)


    @staticmethod
    def infer_resolution_dependencies(pipe):
        """Finds entities_col,resolution_col,doc_col cols for resolution viz viz"""
        entities_component,resolution_component,doc_component = None, None, None
        for c in pipe.components:
            if isinstance(c.model, DocumentAssembler) :              doc_component  = c
            if isinstance(c.model, (NerConverter,NerConverterInternal)) :   entities_component  = c
            if isinstance(c.model, (SentenceEntityResolverModel,ChunkEntityResolverModel)) :        resolution_component  = c

        entities_col       = entities_component.info.outputs[0]
        resolution_col   = resolution_component.info.outputs[0]
        doc_col = doc_component.info.outputs[0]
        return entities_col,resolution_col,doc_col

    @staticmethod
    def viz_relation(anno_res,pipe,is_databricks_env,write_to_streamlit=False):
        """Viz relation result. Set label colors by specifying hex codes, i.e. viz_colors={'TREATMENT':'#800080', 'PROBLEM':'#77b5fe'} """
        relation_col,document_col = VizUtilsHC.infer_relation_dependencies(pipe)
        re_vis = RelationExtractionVisualizer()
        if write_to_streamlit :
            import streamlit as st
            HTML = re_vis.display(anno_res,relation_col = relation_col,document_col = document_col, show_relations=True, return_html=True)
            # st.markdown(HTML, unsafe_allow_html=True)
            st.markdown(VizUtilsHC.HTML_WRAPPER.format(HTML), unsafe_allow_html=True)



        if not is_databricks_env:
            re_vis.display(anno_res,relation_col = relation_col,document_col = document_col, show_relations=True)
        else:
            return  re_vis.display(anno_res,relation_col = relation_col,document_col = document_col, show_relations=True, return_html=True)

    @staticmethod
    def infer_relation_dependencies(pipe):
        """Finds relation_col,document_col  cols for relation viz viz"""
        relation_component,doc_component = None, None
        for c in pipe.components:
            if isinstance(c.model, DocumentAssembler) :              doc_component  = c
            if isinstance(c.model, (RelationExtractionDLModel,RelationExtractionModel)) :   relation_component  = c
        relation_col       = relation_component.info.outputs[0]
        document_col = doc_component.info.outputs[0]
        return relation_col,document_col


    @staticmethod
    def viz_assertion(anno_res,pipe,viz_colors={},is_databricks_env=False,write_to_streamlit=False):
        """Viz relation result. Set label colors by specifying hex codes, i.e. viz_colors={'TREATMENT':'#008080', 'problem':'#800080'} """
        entities_col,assertion_col, doc_col = VizUtilsHC.infer_assertion_dependencies(pipe)
        assertion_vis = AssertionVisualizer()
        if len(viz_colors) > 0 : assertion_vis.set_label_colors(viz_colors)
        if write_to_streamlit :
            import streamlit as st
            HTML = assertion_vis.display(anno_res,label_col = entities_col,assertion_col = assertion_col ,document_col = doc_col,return_html=True)
            # st.markdown(HTML, unsafe_allow_html=True)
            CSS,HTML = HTML.split('</style>')
            CSS = CSS + '</style>'
            HTML = f'<div> {HTML} '
            st.markdown(CSS, unsafe_allow_html=True)
            # st.markdown(HTML, unsafe_allow_html=True)
            st.markdown(VizUtilsHC.HTML_WRAPPER.format(HTML), unsafe_allow_html=True)
        elif not is_databricks_env:
            assertion_vis.display(anno_res,label_col = entities_col,assertion_col = assertion_col ,document_col = doc_col)
        else:
            return  assertion_vis.display(anno_res,label_col = entities_col,assertion_col = assertion_col ,document_col = doc_col,return_html=True)

    @staticmethod
    def infer_assertion_dependencies(pipe):
        """Finds relation_col,document_col  cols for relation viz viz"""
        entities_component,assert_component, doc_component = None, None,None
        for c in pipe.components:
            if isinstance(c.model, DocumentAssembler) :              doc_component  = c
            if isinstance(c.model, (AssertionDLModel,AssertionLogRegModel)) :   assert_component  = c
            if isinstance(c.model, (NerConverter,NerConverterInternal)) :   entities_component  = c

        entities_col       = entities_component.info.outputs[0]
        assertion_col       = assert_component.info.outputs[0]
        doc_col = doc_component.info.outputs[0]
        return entities_col,assertion_col, doc_col


