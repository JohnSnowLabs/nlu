from sparknlp.annotator import *

class StreamlitUtils():
    """Verify for various visualizatins of dependenciesa re satisfied by component_list """

    @staticmethod
    def viz_tree_satisfied(pipe):
        un_typed_dep  = False
        typed_dep  = False
        for c in pipe.components :
            if isinstance(c.model, (DependencyParserModel)):      un_typed_dep = True
            if isinstance(c.model, (TypedDependencyParserModel)): typed_dep    = True
        return un_typed_dep and typed_dep
