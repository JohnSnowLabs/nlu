from sparknlp.annotator import *

class ValidateVizPipe():
    """Verify for various visualizatins of dependenciesa re satisfied by component_list """

    @staticmethod
    def viz_tree_satisfied(pipe):
        un_typed_dep  = False
        typed_dep  = False
        for c in pipe.components :
            if isinstance(c.model, (DependencyParserModel)):      un_typed_dep = True
            if isinstance(c.model, (TypedDependencyParserModel)): typed_dep    = True
        return un_typed_dep and typed_dep

    @staticmethod
    def viz_NER_satisfied(pipe):
        NER  = False
        NER_CONV  = False
        for c in pipe.components :
            if isinstance(c.model, (NerDLModel)):      NER = True
            if isinstance(c.model, (NerConverter)):    NER_CONV    = True
        return NER and NER_CONV

