from nlu.pipe.pipe_components import SparkNLUComponent
class Classifier(SparkNLUComponent):
    def __init__(self, annotator_class='sentiment_dl', language='en', component_type='classifier', get_default=True, model = None, nlp_ref ='', nlu_ref='',trainable=False, is_licensed=False, do_ref_checks=True, lang='en',loaded_from_pretrained_pipe=False):
        if do_ref_checks:
            if 'e2e' in nlu_ref or 'toxic' in nlu_ref : annotator_class= 'multi_classifier'
            elif 'e2e' in nlp_ref or 'toxic' in nlp_ref : annotator_class= 'multi_classifier'

            elif 'multiclassifierdl' in nlp_ref : annotator_class= 'multi_classifier'
            elif 'classifierdl' in nlp_ref: annotator_class= 'classifier_dl'

            elif 'yake' in nlu_ref: annotator_class= 'yake'
            elif 'yake' in nlp_ref: annotator_class= 'yake'

            elif 'sentimentdl' in nlp_ref : annotator_class= 'sentiment_dl'

            elif 'vivekn' in nlp_ref or 'vivekn' in nlp_ref : annotator_class= 'vivekn_sentiment'

            elif 'wiki_' in nlu_ref or 'wiki_' in nlp_ref : annotator_class= 'language_detector'
            elif 'pos' in nlu_ref and 'ner' not in nlu_ref:  annotator_class= 'pos'
            elif 'pos' in nlp_ref and 'ner' not in nlp_ref:  annotator_class= 'pos'

            elif 'icd' in nlu_ref: annotator_class= 'classifier_dl'
            elif 'med_ner' in nlu_ref: annotator_class= 'ner_healthcare'
            elif 'generic_classifier' in nlu_ref: annotator_class= 'generic_classifier'
            elif 'ner' in nlu_ref and 'generic' not in nlu_ref : annotator_class= 'ner'
            elif 'ner' in nlp_ref and 'generic' not in nlp_ref : annotator_class= 'ner'

        if model != None :
            self.model = model
            from sparknlp.annotator import NerDLModel,NerCrfModel
            if isinstance(self.model, (NerDLModel,NerCrfModel)): self.model.setIncludeConfidence(True)
            elif is_licensed :
                from sparknlp_jsl.annotator import MedicalNerModel
                if isinstance(self.model, MedicalNerModel): self.model.setIncludeConfidence(True)

        else :
            if 'sentiment' in annotator_class and 'vivekn' not in annotator_class:
                from nlu import SentimentDl
                if trainable : self.model = SentimentDl.get_default_trainable_model()
                elif is_licensed : self.model = SentimentDl.get_pretrained_model(nlp_ref, language, bucket='clinical/models')
                elif get_default : self.model = SentimentDl.get_default_model()
                else : self.model = SentimentDl.get_pretrained_model(nlp_ref, language)
            elif 'generic_classifier' in annotator_class:
                from nlu.components.classifiers.generic_classifier.generic_classifier import GenericClassifier
                if trainable : self.model     = GenericClassifier.get_default_trainable_model()
                else : self.model = GenericClassifier.get_pretrained_model(nlp_ref, language, bucket='clinical/models')
            elif 'vivekn' in annotator_class:
                from nlu import ViveknSentiment
                if get_default : self.model = ViveknSentiment.get_default_model()
                else : self.model = ViveknSentiment.get_pretrained_model(nlp_ref, language)
            elif 'ner' in annotator_class  and 'ner_healthcare' not in annotator_class:
                from nlu import NERDL
                if trainable : self.model = NERDL.get_default_trainable_model()
                elif is_licensed : self.model = NERDL.get_pretrained_model(nlp_ref, language, bucket='clinical/models')
                elif get_default : self.model = NERDL.get_default_model()
                else : self.model = NERDL.get_pretrained_model(nlp_ref, language)
                if hasattr(self, 'model'): self.model.setIncludeConfidence(True)

            elif 'ner.crf' in annotator_class:
                from nlu import NERDLCRF
                if get_default : self.model = NERDLCRF.get_default_model()
                else : self.model = NERDLCRF.get_pretrained_model(nlp_ref, language)
                if hasattr(self, 'model'): self.model.setIncludeConfidence(True)
            elif ('classifier_dl' in annotator_class or annotator_class == 'toxic') and not 'multi' in annotator_class:
                from nlu import ClassifierDl
                if trainable: self.model = ClassifierDl.get_trainable_model()
                elif is_licensed : self.model = ClassifierDl.get_pretrained_model(nlp_ref, language, bucket='clinical/models')
                elif get_default : self.model = ClassifierDl.get_default_model()
                else : self.model = ClassifierDl.get_pretrained_model(nlp_ref, language)
                if hasattr(self.model,'setIncludeConfidence'): self.model.setIncludeConfidence(True)

            elif 'language_detector' in annotator_class:
                from nlu import LanguageDetector
                if get_default : self.model = LanguageDetector.get_default_model()
                else: self.model = LanguageDetector.get_pretrained_model(nlp_ref, language)
            elif 'pos' in annotator_class:
                from nlu import PartOfSpeechJsl
                if trainable : self.model = PartOfSpeechJsl.get_default_trainable_model()
                elif get_default : self.model = PartOfSpeechJsl.get_default_model()
                elif is_licensed : self.model = PartOfSpeechJsl.get_pretrained_model(nlp_ref, language, bucket='clinical/models')
                else : self.model = PartOfSpeechJsl.get_pretrained_model(nlp_ref, language)

            elif 'yake' in annotator_class:
                from nlu import Yake
                self.model  = Yake.get_default_model()
            elif 'multi_classifier' in annotator_class :
                from nlu import MultiClassifier
                if trainable : self.model = MultiClassifier.get_default_trainable_model()
                elif get_default : self.model = MultiClassifier.get_default_model()
                else : self.model = MultiClassifier.get_pretrained_model(nlp_ref, language)
            elif 'ner_healthcare' in annotator_class:
                from nlu.components.classifiers.ner_healthcare.ner_dl_healthcare import NERDLHealthcare
                if trainable : self.model     = NERDLHealthcare.get_default_trainable_model()
                else : self.model = NERDLHealthcare.get_pretrained_model(nlp_ref, language, bucket='clinical/models')



        SparkNLUComponent.__init__(self, annotator_class, component_type, nlu_ref, nlp_ref, lang,loaded_from_pretrained_pipe , is_licensed)
