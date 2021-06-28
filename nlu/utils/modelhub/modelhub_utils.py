import logging
logger = logging.getLogger('nlu')
import nlu
import requests
class ModelHubUtils():
    modelhub_json_url = 'https://nlp.johnsnowlabs.com/models.json'
    data = requests.get(modelhub_json_url).json()
    """Modelhub utils"""
    @staticmethod
    def NLU_ref_to_NLP_ref(nlu_ref: str,lang: str = None) -> str:
        """Resolve a Spark NLU reference to q NLP reference.
        Args :
        NLU_ref : which nlu model's nlp refrence to return.
        lang : what language is the model in.
        """
        nlu_namespaces_to_check = [nlu.Spellbook.pretrained_pipe_references, nlu.Spellbook.pretrained_models_references, nlu.Spellbook.pretrained_healthcare_model_references, nlu.Spellbook.licensed_storage_ref_2_nlu_ref , nlu.Spellbook.storage_ref_2_nlu_ref]#]
        for dict_ in nlu_namespaces_to_check:
            if lang:
                if lang in dict_.keys():
                    for reference in dict_[lang]:
                        if reference ==nlu_ref:
                            return  dict_[lang][reference]
            else :
                for dict_ in nlu_namespaces_to_check:
                    for lang in dict_:
                        for reference in dict_[lang]:
                            if reference ==nlu_ref:
                                return  dict_[lang][reference]
        for _nlp_ref, nlp_ref_type in nlu.Spellbook.component_alias_references.items():
            if _nlp_ref == nlu_ref : return  nlp_ref_type[0]

        return ''


    @staticmethod
    def get_url_by_nlu_refrence(nlu_refrence: str ) -> str:
        """Rsolves  a  URL for an NLU refrence.

        Args :
            nlu_refrence : Which nlu refrence's url to return.

        """

        # getting spark refrence for given nlu refrence
        if nlu_refrence =='' : return  'https://nlp.johnsnowlabs.com/models'
        if nlu_refrence.split(".")[0] not in  nlu.AllComponentsInfo().all_languages:
            nlu_refrence = "en."+nlu_refrence
        nlp_refrence= ModelHubUtils.NLU_ref_to_NLP_ref(nlu_refrence)
        if nlp_refrence == None :
            print(f"{nlp_refrence}         {nlu_refrence}")
            return 'https://nlp.johnsnowlabs.com/models'
        else :
            for model in ModelHubUtils.data :
                if (model['language'] in nlu_refrence.split(".") or model['language'] in nlp_refrence.split('_')) and model['name'] == nlp_refrence:
                    return f"https://nlp.johnsnowlabs.com/{model['url']}"
        return 'https://nlp.johnsnowlabs.com/models'

    @staticmethod
    def return_json_entry(nlu_refrence:str) -> dict:
        """Resolves a Json entry for an nlp_refrence.

        Args:
            nlp_refrence: What nlp_refrence to resolve.
        """

        if nlu_refrence.split(".")[0] not in  nlu.AllComponentsInfo().all_languages:
            nlu_refrence = "en."+nlu_refrence
        nlp_refrence= ModelHubUtils.NLU_ref_to_NLP_ref(nlu_refrence)

        language = nlu_refrence.split(".")[0]
        for model in ModelHubUtils.data :
            if model['language']== language and model["name"] == nlp_refrence:
                return model

