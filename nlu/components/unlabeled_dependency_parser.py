from nlu.pipe_components import SparkNLUComponent, NLUComponent


class UnlabeledDependencyParser(SparkNLUComponent):

    def __init__(self, component_name='unlabeled_dependency_parser', language='en', component_type='dependency_untyped', get_default = True,sparknlp_reference=''):
        # super(Tokenizer,self).__init__(component_name = component_name, component_type = component_type)
        SparkNLUComponent.__init__(self, component_name, component_type)
        if 'dep' in component_name or 'dep.untyped' in component_name or component_name=='unlabeled_dependency_parser':
            from nlu.components.dependency_untypeds.unlabeled_dependency_parser.unlabeled_dependency_parser import UnlabeledDependencyParser
            if get_default : self.model = UnlabeledDependencyParser.get_default_model()
            else : self.model = UnlabeledDependencyParser.get_pretrained_model(sparknlp_reference,language)