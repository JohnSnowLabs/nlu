import logging
import nlu
import requests

logger = logging.getLogger('nlu')


class ModelHubUtils():
    """Modelhub utils"""
    modelhub_json_url = 'https://nlp.johnsnowlabs.com/models.json'
    data = requests.get(modelhub_json_url).json()

    @staticmethod
    def NLU_ref_to_NLP_ref(nlu_ref: str, lang: str = None) -> str:
        """Resolve a Spark NLU reference to a NLP reference.
        :param nlu_ref: which nlu model_anno_obj's nlp refrence to return.
        :param lang: what language is the model_anno_obj in.
        :return: Spark nlp model_anno_obj name
        """
        nlu_namespaces_to_check = [nlu.Spellbook.pretrained_pipe_references, nlu.Spellbook.pretrained_models_references,
                                   nlu.Spellbook.pretrained_healthcare_model_references,
                                   nlu.Spellbook.licensed_storage_ref_2_nlu_ref,
                                   nlu.Spellbook.storage_ref_2_nlu_ref]  # ]
        for dict_ in nlu_namespaces_to_check:
            if lang:
                if lang in dict_.keys():
                    for reference in dict_[lang]:
                        if reference == nlu_ref:
                            return dict_[lang][reference]
            else:
                for dict_ in nlu_namespaces_to_check:
                    for lang in dict_:
                        for reference in dict_[lang]:
                            if reference == nlu_ref:
                                return dict_[lang][reference]
        for _nlp_ref, nlp_ref_type in nlu.Spellbook.component_alias_references.items():
            if _nlp_ref == nlu_ref: return nlp_ref_type[0]
        return ''

    @staticmethod
    def get_url_by_nlu_refrence(nlu_ref: str) -> str:
        """Resolves a URL for an NLU reference.
        :param nlu_ref: Which nlu refrence's url to return.
        :return: url to modelhub
        """

        # getting spark refrence for given nlu refrence
        if not nlu_ref: return 'https://nlp.johnsnowlabs.com/models'
        if nlu_ref.split(".")[0] not in nlu.Spellbook.pretrained_models_references.keys():
            nlu_ref = "en." + nlu_ref
        nlp_refrence = ModelHubUtils.NLU_ref_to_NLP_ref(nlu_ref)
        if nlp_refrence == None:
            print(f"{nlp_refrence}         {nlu_ref}")
            return 'https://nlp.johnsnowlabs.com/models'
        else:
            for model in ModelHubUtils.data:
                if (model['language'] in nlu_ref.split(".") or model['language'] in nlp_refrence.split('_')) and \
                        model['name'] == nlp_refrence:
                    return f"https://nlp.johnsnowlabs.com/{model['url']}"
            for model in ModelHubUtils.data:
                # Retry, but no respect to lang
                if model['name'] == nlp_refrence:
                    return f"https://nlp.johnsnowlabs.com/{model['url']}"
        return 'https://nlp.johnsnowlabs.com/models'

    @staticmethod
    def return_json_entry(nlu_ref: str) -> dict:
        """Resolves a Json entry for annlu_refrence.
        :param nlu_ref:  What nlp_refrence to resolve
        :return: Json entry of that nlu reference
        """
        if nlu_ref.split(".")[0] not in nlu.Spellbook.pretrained_models_references.keys():
            nlu_ref = "en." + nlu_ref
        nlp_refrence = ModelHubUtils.NLU_ref_to_NLP_ref(nlu_ref)

        language = nlu_ref.split(".")[0]
        for model in ModelHubUtils.data:
            if model['language'] == language and model["name"] == nlp_refrence:
                return model
