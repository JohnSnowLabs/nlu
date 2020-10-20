from nlu.pipe_components import SparkNLUComponent

class LabeledDependencyParser(SparkNLUComponent):
    def __init__(self, annotator_class='labeled_dependency_parser', language ='en', component_type='dependency_typed', get_default=True, nlp_ref='',  nlu_ref=''):
        SparkNLUComponent.__init__(self, annotator_class, component_type)
        if 'dep' in annotator_class:
            from nlu.components.dependency_typeds.labeled_dependency_parser.labeled_dependency_parser import \
                LabeledDependencyParser
            if get_default : self.model = LabeledDependencyParser.get_default_model()
            else :self.model = LabeledDependencyParser.get_pretrained_model(nlp_ref, language)