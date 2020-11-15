from nlu.pipe_components import SparkNLUComponent
class Classifier(SparkNLUComponent):
    def __init__(self, annotator_class='sentiment_dl', language='en', component_type='classifier', get_default=True, model = None, nlp_ref ='', nlu_ref='',trainable=False):
        if 'e2e' in nlu_ref or 'toxic' in nlu_ref : annotator_class= 'multi_classifier'
        elif 'e2e' in nlp_ref or 'toxic' in nlp_ref : annotator_class= 'multi_classifier'

        elif 'multiclassifierdl' in nlp_ref : annotator_class= 'multi_classifier'
        elif 'classifierdl' in nlp_ref: annotator_class= 'classifier_dl'

        elif 'yake' in nlu_ref: annotator_class= 'yake'
        elif 'yake' in nlp_ref: annotator_class= 'yake'

        elif 'sentimentdl' in nlp_ref : annotator_class= 'sentiment_dl'

        elif 'vivekn' in nlp_ref or 'vivekn' in nlp_ref : annotator_class= 'vivekn_sentiment'

        elif 'wiki_' in nlu_ref or 'wiki_' in nlp_ref : annotator_class= 'language_detector'
        elif 'pos' in nlu_ref: annotator_class= 'pos'
        elif 'pos' in nlp_ref: annotator_class= 'pos'

        elif 'ner' in nlu_ref: annotator_class= 'ner'
        elif 'ner' in nlp_ref: annotator_class= 'ner'

        
        if model != None : self.model = model
        else :
            if 'sentiment' in annotator_class and 'vivekn' not in annotator_class:
                from nlu import SentimentDl
                if get_default : self.model = SentimentDl.get_default_model()
                else : self.model = SentimentDl.get_pretrained_model(nlp_ref, language)
            elif 'vivekn' in annotator_class:
                from nlu import ViveknSentiment
                if get_default : self.model = ViveknSentiment.get_default_model()
                else : self.model = ViveknSentiment.get_pretrained_model(nlp_ref, language)
            elif 'ner' in annotator_class or 'ner.dl' in annotator_class:
                from nlu import NERDL
                if get_default : self.model = NERDL.get_default_model()
                else : self.model = NERDL.get_pretrained_model(nlp_ref, language)
            elif 'ner.crf' in annotator_class:
                from nlu import NERDLCRF
                if get_default : self.model = NERDLCRF.get_default_model()
                else : self.model = NERDLCRF.get_pretrained_model(nlp_ref, language)
            elif 'multi_classifier_dl' in annotator_class:
                from nlu import MultiClassifierDl
                if get_default : self.model = MultiClassifierDl.get_default_model()
                else : self.model = MultiClassifierDl.get_pretrained_model(nlp_ref, language)
            elif ('classifier_dl' in annotator_class or annotator_class == 'toxic') and not 'multi' in annotator_class:
                from nlu import ClassifierDl
                if trainable: self.model = ClassifierDl.get_trainable_model()
                elif get_default : self.model = ClassifierDl.get_default_model()
                else : self.model = ClassifierDl.get_pretrained_model(nlp_ref, language)
            elif 'language_detector' in annotator_class:
                from nlu import LanguageDetector
                if get_default : self.model = LanguageDetector.get_default_model()
                else: self.model = LanguageDetector.get_pretrained_model(nlp_ref, language)
            elif 'pos' in annotator_class:
                from nlu import PartOfSpeechJsl
                if get_default : self.model = PartOfSpeechJsl.get_default_model()
                else : self.model = PartOfSpeechJsl.get_pretrained_model(nlp_ref, language)
            elif 'yake' in annotator_class:
                from nlu import Yake
                self.model  = Yake.get_default_model()
            elif 'multi_classifier' in annotator_class :
                from nlu import MultiClassifier
                if get_default : self.model = MultiClassifier.get_default_model()
                else : self.model = MultiClassifier.get_pretrained_model(nlp_ref, language)
        SparkNLUComponent.__init__(self, annotator_class, component_type)
