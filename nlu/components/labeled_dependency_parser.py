from nlu.pipe_components import SparkNLUComponent

class LabeledDependencyParser(SparkNLUComponent):
    def __init__(self, component_name='labeled_dependency_parser', language = 'en' , component_type='dependency_typed', get_default=True,sparknlp_reference=''):
        SparkNLUComponent.__init__(self, component_name, component_type)
        if 'dep' in component_name:
            from nlu.components.dependency_typeds.labeled_dependency_parser.labeled_dependency_parser import \
                LabeledDependencyParser
            if get_default : self.model = LabeledDependencyParser.get_default_model()
            else :self.model = LabeledDependencyParser.get_pretrained_model(sparknlp_reference,language)