"""
This module contains methods that will be applied inside of the apply call on the freshly convertet
Spark DF that is now as list of dicts.

These methods are meant to be applied in the internal calls of the metadata extraction.
They expect dictionaries which represent the metadata field extracted from Spark NLP annotators.


"""

def meta_extract_language_classifier_max_confidence(row,configs):
    ''' Extract the language classificationw ith highest confidence and drop the others '''
    # Get the best, but what about TOP K! todo
    # TODO conditional sentence extraction and mroe docs
    #unpack all confidences to float and set 'sentence' key value to -1 so it does not affect finding the highest cnfidence
    unpack_dict_values = lambda x : -1 if 'sentence' in x[0]  else float(x[1][0])
    l = list(map(unpack_dict_values,row.items()))
    m = np.argmax(l)
    k = list(row.keys())[m]

    return {k+'_confidence' : row[k][0]} # remoe [0] for list return


def meta_extract_maximum_binary_confidence(row,configs):
    ''' Extract the maximum confidence for a binary classifier that returns 2 confidences.
    key schema is 'meta_' + configs.output_col_prefix + '_confidence'

    Parameters
    -------------
    configs : SparkNLPExtractorConfig
    if configs.get_sentence_origin is True, the sentence origin column will be kept, otherwise dropped.


    row : dict
        i.e. looks like {'meta_sentiment_dl_sentence': ['0', '1', '2'], 'meta_sentiment_dl_pos': ['1.0', '1.0', '1.0'], 'meta_sentiment_dl_neg': ['5.5978343E-11', '5.5978343E-11', '5.5978343E-11']}

    Returns
    ------------
    dict
      if configs.get_sentence_origin True  {'meta_sentiment_dl_sentence': ['0', '1'], 'meta_sentiment_dl_confidence': [0.9366506, 0.9366506]}
      else {'meta_sentiment_dl_confidence': [0.9366506, 0.9366506]}
    '''
    # TODO inline these variables
    meta_sent_key = 'meta_' + configs.output_col_prefix + '_sentence'
    meta_conf_key_neg = 'meta_' + configs.output_col_prefix + '_neg'
    meta_conf_key_pos = 'meta_' + configs.output_col_prefix + '_pos'
    # Zip Pos/Neg conf column and keep max
    keep_max = lambda x: max(float(x[0]), float(x[1]))
    return {
        **{'meta_' + configs.output_col_prefix + '_confidence':list(map(keep_max,zip(row[meta_conf_key_pos],row[meta_conf_key_neg])))},
        **({'meta_' + configs.output_col_prefix + '_sentence' : row[meta_sent_key]} if configs.get_sentence_origin else {})
    }


