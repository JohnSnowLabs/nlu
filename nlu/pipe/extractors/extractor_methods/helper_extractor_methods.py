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


def extract_resolver_all_k_subfields_splitted(row, configs):
    ''' Extract all metadata fields for sentence resolver annotators and splits all _k_ fields on ::: , relevant for all_k_result, all_k_resolutions, al_k_distances, all_k_cosine_distances
    pop_meta_list should be true, if outputlevel of pipe is the same as the resolver component we are extracting here for
    ::: for icd
    || for HCC
    '''
    prefix = 'meta_' + configs.output_col_prefix + '_'
    res = {}
    for k in row.keys():
        if '_k_' in k:
            if '||' in row[k][0]:
                # billable code handling
                f = lambda s: list(map(lambda x: x.split("||"), s.split(':::')))
                # Triple assignment so we unpkack properly
                res[prefix + 'billable'], \
                res[prefix + 'hcc_status'], \
                res[prefix + 'hcc_code'] = zip(*map(lambda x: zip(*x), map(f, row[k])))
                # Casting from tuple to list or we get problems during pd explode
                h = lambda z: list(map(lambda r: list(r), z))
                res[prefix + 'billable'] = h(res[prefix + 'billable'])
                res[prefix + 'hcc_status'] = h(res[prefix + 'hcc_status'])
                res[prefix + 'hcc_code'] = h(res[prefix + 'hcc_code'])

            elif ':::' in row[k][0]:
                # General code handling
                res[k.replace('results', 'codes')] = list(map(lambda x: x.split(':::'), row[k]))
        else:
            # Any other metadata field hadling
            res[k] = row[k]
    return res
    # if we pop, this means we extract array elements into single elements, i.e. ['hello'] will be 'hello'. We can do this if field is on same level as ppipe
    #
    # if configs.pop_meta_list:
    #     if is_code:
    #         return {k.replace('results', 'codes'):
    #                     row[k][0].split(":::") if '_k_' in k else
    #                     row[k][0] for k in row.keys()}
    #     elif is_billable:
    #         # billable
    #         res = {}
    #         for k in row.keys():
    #             if '_k_' in k:
    #                 billable, hcc_status, hcc_code = row[k].split('||')
    #                 res['billable'] = billable
    #                 res['hcc_status'] = hcc_status
    #                 res['hcc_code'] = hcc_code
    #             else:
    #                 res[k] = row[k][0]
    #     else:
    #         return {k: row[k] for k in row.keys()}
    #
    # else:
    #     if is_code:
    #         return {k.replace('results', 'codes'):
    #                     row[k][0].split(":::") if '_k_' in k else
    #                     row[k] for k in row.keys()}
    #     elif is_billable:
    #         # billable
    #         res = {}
    #         for k in row.keys():
    #             if '_k_' in k:
    #                 billable, hcc_status, hcc_code = row[k].split('||')
    #                 res['billable'] = billable
    #                 res['hcc_status'] = hcc_status
    #                 res['hcc_code'] = hcc_code
    #             else:
    #                 res[k] = row[k]
    #     else:
    #         return {k: row[k][0] for k in row.keys()}
# z[['meta_icd10cm_code_billable', 'meta_icd10cm_code_hcc_status','meta_icd10cm_code_hcc_code']]
# df['meta_icd10cm_code_billable'].apply(lambda r : list(map(list, r))).values
