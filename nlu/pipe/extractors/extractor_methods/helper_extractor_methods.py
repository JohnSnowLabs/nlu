"""
This module contains methods that will be applied inside of the apply call on the freshly convertet
Spark DF that is now as list of dicts.

These methods are meant to be applied in the internal calls of the metadata extraction.
They expect dictionaries which represent the metadata field extracted from Spark NLP annotators.


"""
import numpy as np


def meta_extract_language_classifier_max_confidence(row, configs):
    ''' Extract the language classificationw ith highest confidence and drop the others '''
    # # todo Get the best, but what about TOP K!  conditional sentence extraction and mroe docs
    # unpack all confidences to float and set 'sentence' key value to -1 so it does not affect finding the highest cnfidence
    unpack_dict_values = lambda x: -1 if 'sentence' in x[0] else float(x[1][0])
    l = list(map(unpack_dict_values, row.items()))
    m = np.argmax(l)
    k = list(row.keys())[m]

    return {k + '_confidence': row[k][0]}  # remoe [0] for list return


def zipp(l): return zip(*l)  # unpack during list comprehension not supported in Python, need this workaround for now


def extract_maximum_confidence(row, configs):
    ''' Extract the maximum confidence from any classifier with N classes.
    A classifier with N classes, has N confidences in it's metadata by default, which is too much data usually.
    This extractor gets the highest confidence from the array of confidences.
    This method assumes all keys in metadata corrospond to confidences, except the `sentence` key, which maps to a sentence ID
    key schema is 'meta_' + configs.output_col_prefix + '_confidence'

    Parameters
    -------------
    configs : SparkNLPExtractorConfig
    if configs.get_sentence_origin is True, the sentence origin column will be kept, otherwise dropped.
    row : dict
        i.e. looks like{'meta_category_sentence': ['0'],'meta_category_surprise': ['0.0050183665'],'meta_category_sadness': ['8.706827E-5'],'meta_category_joy': ['0.9947379'],'meta_category_fear': ['1.5667251E-4']}
    Returns
    ------------
    dict
      if configs.get_sentence_origin True  {'meta_sentiment_dl_sentence': ['0', '1'], 'meta_sentiment_dl_confidence': [0.9366506, 0.9366506]}
      else {'meta_sentiment_dl_confidence': [0.9366506, 0.9366506]}
    '''
    meta_sent_key = 'meta_' + configs.output_col_prefix + '_sentence'
    fl = lambda \
            k: False if 'sentence' in k else True  # every key  that has not the sub string sentence in it is considerd a confidence key
    confidences_keys = list(filter(fl, row.keys()))

    if configs.pop_meta_list:
        return {
            **{'meta_' + configs.output_col_prefix + '_confidence': max([float(row[k][0]) for k in confidences_keys])},
            **({'meta_' + configs.output_col_prefix + '_sentence': row[
                meta_sent_key]} if configs.get_sentence_origin else {})
        }
    else:
        if len(confidences_keys) == 1:
            return {
                **{'meta_' + configs.output_col_prefix + '_confidence': max([float(row[k]) for k in confidences_keys])},
                **({'meta_' + configs.output_col_prefix + '_sentence': row[
                    meta_sent_key]} if configs.get_sentence_origin else {})
            }
        else:
            return {
                **{'meta_' + configs.output_col_prefix + '_confidence': [max(z) for z in zipp(
                    list(map(float, row[k])) for k in confidences_keys)]},
                **({'meta_' + configs.output_col_prefix + '_sentence': row[
                    meta_sent_key]} if configs.get_sentence_origin else {})
            }


def unpack_HPO_codes(row, k):
    # Case: HPO
    UMLS_codes = []
    ORPHA_CODES = []
    MSH_CODES = []
    SNOMED_CODES = []
    OMIM_CODES = []

    for resolution in row[k]:
        k_candidates = resolution.split(':::')
        for candidate in k_candidates:
            # There is 0 to 5 alt terminologies, 1 per HPO_codes. If one is missing, we have to append None
            alt_terminologies = candidate.split('||')
            UMLS_ok, ORPHA_ok, MSH_ok, SNOMED_ok, OMIM_ok = False, False, False, False, False
            for alt in alt_terminologies:
                if 'UMLS' in alt:
                    UMLS_codes.append(alt)
                    UMLS_ok = True
                elif 'ORPHA' in alt:
                    ORPHA_CODES.append(alt)
                    ORPHA_ok = True
                elif 'MSH' in alt:
                    MSH_CODES.append(alt)
                    MSH_ok = True
                elif 'SNOMED' in alt:
                    SNOMED_CODES.append(alt)
                    SNOMED_ok = True
                elif 'OMIM' in alt:
                    OMIM_CODES.append(alt)
                    OMIM_ok = True
            # Detect which of the 0 to 5 alt terminologies are missing and add None for them
            if not UMLS_ok:
                UMLS_codes.append(None)
            if not ORPHA_ok:
                ORPHA_CODES.append(None)
            if not MSH_ok:
                MSH_CODES.append(None)
            if not SNOMED_ok:
                SNOMED_CODES.append(None)
            if not OMIM_ok:
                OMIM_CODES.append(None)
    return UMLS_codes, ORPHA_CODES, MSH_CODES, SNOMED_CODES, OMIM_CODES,
    # Write into dict


def extract_resolver_all_k_subfields_splitted(row, configs):
    ''' Extract all metadata fields for sentence resolver annotators and splits all _k_ fields on ::: , relevant for all_k_result, all_k_resolutions, al_k_distances, all_k_cosine_distances
    pop_meta_list should be true, if outputlevel of pipe is the same as the resolver component we are extracting here for
    ::: for icd
    || in HCC splits billable/status/code, always aligns. in HPO splits resolutions in alterantive terminologies, not always aligns
    ::: splits Mappings of resolutions, i.e. len(split(:::)) == number_of_entities  == number of resolutions
    Special case HPO :MeSH/SNOMED/UMLS/ORPHA/OMIM


unpack_resolutions = lambda x : x.split(':::')
unpack_k_aux_label = lambda x : x.split('||')
# each of the k resolutions as K aux label.
# for HPO, each aux label, can have 0 to 5 extra terminolgies
unpack_terms = lambda x: x.split('')

unpacked = list(map(lambda x: list(map(unpack_k_aux_label, unpack_resolutions(x))), row[k] ))

for key, g in itertools.groupby(unpacked, lambda x : x.split(':')[0]):
    print(key,list(g))

unpacked[2]

    :UMLS
    '''
    # todo del AUX label col for CODES
    HPO_CODES = ['UMLS', 'ORPHA', 'MSH', 'SNOMED', 'OMIM']
    prefix = 'meta_' + configs.output_col_prefix + '_'
    res = {}
    for k in row.keys():
        if '_k_' in k:
            if '||' in row[k][0]:

                if any(x in row[k][0] for x in HPO_CODES):
                    # Case : HPO
                    res[prefix + 'k_UMLS_codes'], \
                    res[prefix + 'k_ORPHA_codes'], \
                    res[prefix + 'k_MESH_codes'], \
                    res[prefix + 'k_SNOMED_codes'], \
                    res[prefix + 'k_OMIM_codes'] = unpack_HPO_codes(row, k)

                else:
                    # CASE : billable code handling
                    f = lambda s: list(map(lambda x: x.split("||"), s.split(':::')))
                    # Triple assignment so we unpack properly
                    res[prefix + 'billable'], \
                    res[prefix + 'hcc_status'], \
                    res[prefix + 'hcc_code'] = zip(*map(lambda x: zip(*x), map(f, row[k])))
                    # Casting from tuple to list or we get problems during pd explode
                    h = lambda z: list(map(lambda r: list(r), z))# [0]
                    res[prefix + 'billable'] = h(res[prefix + 'billable'])
                    res[prefix + 'hcc_status'] = h(res[prefix + 'hcc_status'])
                    res[prefix + 'hcc_code'] = h(res[prefix + 'hcc_code'])

            elif ':::' in row[k][0]:
                # CASE : General code handling
                res[prefix+k.replace('results', 'codes')] = list(map(lambda x: x.split(':::'), row[k]))# [0]
        else:
            # Any other metadata field hadling
            res[k] = row[k]
    return res


def extract_chunk_mapper_relation_data(row, configs):
    ''' Splits all_relations field on ::: to create an array ,


    uses row.relation as prefix


    '''

    prefix = 'meta_' + configs.output_col_prefix + '_'
    for k in row.keys():
        if 'chunk_all_relations' in k:
            row[k] = [s.split(':::') for s in row[k]]
    return row



def extract_coreference_data(row_metadata,row_results, configs):
    ''' Splits all_relations field on ::: to create an array ,
    | Text                                                                    |heads         | Co-References    | Heads_sentence| Coref_sentence| coref_head_begin | coref_head_end|
    |John told Mary he would like to borrow a book from her, after his lunch  |[John, Marry]  | [he,his], [her] | [0,0]         | [0,0],[0]     | [0,0], [10]      | [3,3], [13]


    |Text     | heads  | Co-Refernces
    | John    | ROOT   |   [he,his]
    | told    | /      |  /
    | Marry   |  ROOT | [her]
    | he      | JOHN  | /
    | likes   | /     |  /
    | her     | MARRY |/



    |ORIGIN_REFERENCE | CO_REFERENCES|
    | Peter           | he , him, that dude |
    | Maria           | her, she, the lady |


    '''
    head_to_coref = {}
    prefix = 'meta_' + configs.output_col_prefix + '_'

    # for (k_meta,v_meta), (k_result,v_result) in zip(row_metadata.items(), row_results.items()):
    #     if

    raise NotImplemented('Not implemented')
