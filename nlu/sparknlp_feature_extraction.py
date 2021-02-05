from functools import reduce
"""
Contains methods to apply on pandas Dataframes, so called Extractors which are meant to run on the result of a SparkNLP pipeline .to_pandas() result`

There is one extractor for every Annotatornnotatyor class in Spark NLP. 
Each extractor is meant to applied on a pandas df, running df.apply(extractor)

Analogus Spark based API <<<TO BE IMPLEMENTED>>>

All extractors leverage  extract_base_sparknlp_features 
Most extractors provide configurable metadata and all of them provide a full_metadata which gets every key/value pair frrom the metadata field


"""

def extract_base_sparknlp_features(row, key_prefix, extract_positions = True, extract_result = True, extract_annotator_type=True, extract_embeddings = True):
    """
    Extract base features common in all saprk NLP annotators
    Begin/End/Embedding/Metadata/Result, except for the blacklisted features

    row = pyspark.row
    returns a DICT
    """
    unpack_dict_list = lambda d, k : d[k]
    unpack_begin = lambda x : unpack_dict_list(x,'begin')
    unpack_end = lambda x : unpack_dict_list(x,'end')
    unpack_annotator_type = lambda x : unpack_dict_list(x,'annotatorType')
    unpack_result = lambda x : unpack_dict_list(x,'result')
    unpack_embeddings = lambda x : unpack_dict_list(x,'embeddings')



    # Either extract list of anno results and put them in a dict with corrosponding key name or return {} for easy merge in return
    beginnings      = { key_prefix+'_beginnings' : list(map(unpack_begin,row))} if extract_positions else {}
    endings         = { key_prefix+'_endings'    : list(map(unpack_end,row))} if extract_positions else {}
    results         = { key_prefix+'_results'    : list(map(unpack_result,row))} if extract_result else {}
    embeddings         = { key_prefix+'_embeddings'    : list(map(unpack_embeddings,row))} if extract_embeddings else {}
    annotator_types = { key_prefix+'_types'    : list(map(unpack_annotator_type,row))} if extract_annotator_type else {}


    return {**beginnings,**endings,**results,**annotator_types, **embeddings} # Merge dicts


def extract_sparknlp_full_metadata(row, meta_key_prefix, meta_key_blacklist=[]):
    """
    Extract base features common in all saprk NLP annotators
    Begin/End/Embedding/Metadata/Result, except for the blacklisted features

    returns a DICT which will be merged into pd.Serise by the extractor calling this exctractor for .apply() in pythonify
    """

    unpack_dict_list = lambda d, k : d[k]
    unpack_metadata_to_dict_list = lambda x : unpack_dict_list(x,'metadata')

    # extract list of metadata dictionaries (all dict should have same keys)
    metadatas_dict_list = list(map(unpack_metadata_to_dict_list,row))
    # extract keys, which should all be equal in all rows
    keys_in_metadata = list(metadatas_dict_list[0].keys()) if len(metadatas_dict_list) > 0 else []
    # dectorate lambda with key to extract, equalt to def decorate_f(key): return lambda x,y :  x+ [y[key]]
    # For a list of dicts which all have the same keys, will return a lit of all the values for one key in all the dicts
    extract_val_from_dic_list_to_list = lambda key : lambda x,y :  x+ [y[key]]
    # List of lambda expression, on for each Key to be extracted. (TODO balcklisting?)
    dict_value_extractors = list(map(extract_val_from_dic_list_to_list,keys_in_metadata))
    # reduce list of dicts with same struct and a common key to a list of values for thay key. Leveraging closuer for meta_dict_list
    reduce_dict_list_to_values = lambda t : reduce(t,metadatas_dict_list,[])
    # list of lists, where each list is corrosponding to all values in the previous dict list
    meta_values_list = list(map(reduce_dict_list_to_values, dict_value_extractors))#, metadatas_dict_list,[] ))
    # add prefix to key and zip with values for final dict result
    result = dict(zip(list(map(lambda x : 'full_meta_'+ meta_key_prefix + '_' + x, keys_in_metadata)),meta_values_list))
    return result


res.token.apply(extract_sparknlp_full_metadata,meta_key_prefix='kek')
