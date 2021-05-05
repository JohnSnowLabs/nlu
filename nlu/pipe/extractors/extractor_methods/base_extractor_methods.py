"""
This module contains methods that will be applied inside of the apply call on the freshly convertet
Spark DF that is now as list of dicts.

These methods are meant to be applied in the internal calls of the metadata extraction.
They expect dictionaries which represent the metadata field extracted from Spark NLP annotators.


"""
import numpy as np
from pyspark.sql import Row as PysparkRow
from nlu.pipe.extractors.extractor_base_data_classes import *
from functools import reduce, partial
import pandas as pd

def extract_pyspark_rows(r:pd.Series,)-> pd.Series:
    """ Convert pyspark.sql.Row[Annotation] to List(Dict[str,str]) objects. Except for key=metadata in dict, this element in the Dict which is [str,Dict[str,str]]

    Checks if elements are of type list and wether they contain Pyspark Rows.
     If PysparkRow, call .asDict() on every row element to generate the dicts
    First method that runs after toPandas() call
    """
    if    isinstance(r,str) : return r
    elif  isinstance(r,list):
        if len(r) == 0 : return r
        # SHOULD THIS REALLY BE LIST???? WHY NOT JUST DICTS?!?! ISNT IT JUST ONE DICT PER ROW?!?!?
        # THEN WE CAN LEAVE OUT THIS OUTER LIST!
        elif isinstance(r[0], PysparkRow):
            pyspark_row_to_list = lambda l : l.asDict()
            # return next(map(pyspark_row_to_list,r))
            # WE MUST MAKE THIS LIST! A aNNOTATOR MAY RETURN MULTIPLE aNNOTATIONS per Row! Cannnot use next()
            return list(map(pyspark_row_to_list,r))
    return r


def extract_base_sparknlp_features(row:pd.Series, configs:SparkNLPExtractorConfig)->dict:
    """
    Extract base features common in all saprk NLP annotators
    Begin/End/Embedding/Metadata/Result, except for the blacklisted features
    Expects a list with Token Annotator Outputs from extract_pyspark_rows() , i.e
    Setting pop to true for a certain field will return only the first element of that fields list of elements. Useful if that field always has exactly 1 result, like many classifirs
    [{'annotatorType': 'token',
    'begin': 0,
    'embeddings': [],
    'end': 4,
    'metadata': {'sentence': '0'},
    'result': 'Hello'
    }]
    row = pyspark.row

    or

    [
      {'annotatorType': 'language',
  'begin': 0,
  'embeddings': [],
  'end': 57,
  'metadata': {'bg': '0.0',
   'sentence': '0',
   'sl': '5.2462015E-24',
   'sv': '2.5977007E-25'},
  'result': 'en'}
  ]

    returns a DICT
    """
    unpack_dict_list      = lambda d, k : d[k]
    unpack_begin          = lambda x : unpack_dict_list(x,'begin')
    unpack_end            = lambda x : unpack_dict_list(x,'end')
    unpack_annotator_type = lambda x : unpack_dict_list(x,'annotatorType')
    unpack_result         = lambda x : unpack_dict_list(x,'result')
    unpack_embeddings     = lambda x : unpack_dict_list(x,'embeddings')
    # Either extract list of anno results and put them in a dict with corrosponding key name or return empty dict {} for easy merge in return
    annotator_types = { configs.output_col_prefix+'_types'      : list(map(unpack_annotator_type,row))} if configs.get_annotator_type else {}
    # Same logic as above, but we check wether to pop or not and either evaluate the map result with list() or just next()
    if configs.pop_result_list:
        results         = { configs.output_col_prefix+'_results'  : next(map(unpack_result,row))} if configs.get_result else {}
    else:
        results         = { configs.output_col_prefix+'_results'  : list(map(unpack_result,row))} if configs.get_result else {}
    if configs.pop_begin_list:
        beginnings      = { configs.output_col_prefix+'_beginnings' : next(map(unpack_begin,row))} if configs.get_begin or configs.get_positions else {}
    else:
        beginnings      = { configs.output_col_prefix+'_beginnings' : list(map(unpack_begin,row))} if configs.get_begin or configs.get_positions else {}

    if configs.pop_end_list:
        endings         = { configs.output_col_prefix+'_endings'    : next(map(unpack_end,row))} if configs.get_end or configs.get_positions else {}
    else:
        endings         = { configs.output_col_prefix+'_endings'    : list(map(unpack_end,row))} if configs.get_end or configs.get_positions else {}

    if configs.pop_embeds_list:
        embeddings      = { configs.output_col_prefix+'_embeddings' : next(map(unpack_embeddings,row))} if configs.get_embeds else {}
    else:
        embeddings      = { configs.output_col_prefix+'_embeddings' : list(map(unpack_embeddings,row))} if configs.get_embeds else {}

    return {**beginnings,**endings,**results,**annotator_types, **embeddings} # Merge dicts




def extract_sparknlp_metadata(row : pd.Series, configs:SparkNLPExtractorConfig)-> dict:
    """
    Extract base features common in all saprk NLP annotators
    Begin/End/Embedding/Metadata/Result, except for the blacklisted features
    Expects a list with Token Annotator Outputs, i.e.
    Can either use a WHITE_LISTE or BLACK_LIST or get ALL metadata
    For WHITE_LIST != [], only metadata keys/values will be kepts, for which the keys are contained in the white list
    For WHITE_LIST == [] AND BLACK_LIST !=, all metadata key/values will be returned, which are not on the black list.
    If  WHITE_LIST is not [] the BLACK_LIST will be ignored.

    returns one DICT which will be merged into pd.Serise by the extractor calling this exctractor for .apply() in pythonify
    """
    if len(row) == 0: return {}
    unpack_dict_list = lambda d, k : d[k]
    # extract list of metadata dictionaries (all dict should have same keys)
    unpack_metadata_to_dict_list = lambda x : unpack_dict_list(x,'metadata')

    metadatas_dict_list = list(map(unpack_metadata_to_dict_list,row))
    # extract keys, which should all be equal in all rows

    if configs.get_full_meta:
        keys_in_metadata = list(metadatas_dict_list[0].keys()) if len(metadatas_dict_list) > 0 else []
    elif len(configs.meta_white_list) != 0 :
        keys_in_metadata = [k for k in metadatas_dict_list[0].keys() if k in configs.meta_white_list ]
    elif len(configs.meta_black_list) !=0 :
        keys_in_metadata = [k for k in metadatas_dict_list[0].keys() if k not in configs.meta_black_list ]
    else :
        keys_in_metadata = []

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
    result = dict(zip(list(map(lambda x : 'meta_'+ configs.output_col_prefix + '_' + x, keys_in_metadata)),meta_values_list))
    return result


def extract_master(row:pd.Series ,configs:SparkNLPExtractorConfig ) -> pd.Series:
    """
    Re-Usable base extractor for simple Annotators like Document/Token/etc..?
    extract_universal/?/Better name?
    row = a list or Spark-NLP annotations as dictionary
    """
    if len(row) == 0 : return pd.Series({})
    # Get base annotations
    base_annos = extract_base_sparknlp_features(row,configs)
    # Get Metadata
    all_metas = extract_sparknlp_metadata(row, configs) if configs.get_meta or configs.get_full_meta else {}

    # Apply custom extractor methods
    if configs.meta_data_extractor.name != '':
        all_metas = configs.meta_data_extractor.extractor_method(all_metas,configs)

    # Apply Finishers on metadata/additional fields
    return pd.Series(
        {
            ** base_annos,
            ** all_metas
        })




def apply_extractors_and_merge(df, column_to_extractor_map,  keep_stranger_features,stranger_features):
    """ apply extract_master on all fields with corrosponding configs after converting Pyspark Rows to List[Dict]
    and merge them to a final DF (1 to 1 mapping still)
    df  The Df we want to apply the extractors on
    columns_to_extractor_map Map column names to extractor configs. Columns which are not in these keys will be ignored These configs will be passed to master_extractor for every column
    """
    # keep df and ex_resolver in closure and apply base extractor with configs for each col
    extractor      = lambda c : df[c].apply(extract_master, configs = column_to_extractor_map[c])
    keep_strangers = lambda c : df[c]

    # merged_extraction_df
    # apply the extract_master together with it's configs to every column and geenrate a list of output DF's, one per Spark NLP COL
    return pd.concat(
        list(map(extractor,column_to_extractor_map.keys())) +
        list(map(keep_strangers,stranger_features)) if keep_stranger_features else [],
        axis=1)
    # return zip_and_explode(
    #         pd.concat(
    #             list(map(extractor,column_to_extractor_map.keys())) +
    #             list(map(keep_stragers,stranger_features)) if keep_stranger_features else [],
    #             axis=1))

# !wget https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/scripts/colab_setup.sh -O - | bash
# ! pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple peanut_butter_data_time==3.0.1rc56 > /dev/null
# ! pip install spark-nlp-display

# !wget https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/scripts/colab_setup.sh -O - | bash
# ! pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple peanut_butter_data_time==3.0.1rc71 > /dev/null
# import nlu
# import nlu


# def pad_series_for_multi_pad(s):
#     """Sub token output level """
#     # index_df.columns=cols_to_explode
#     # df.explode
#     # index_sizes = [d.groupby(d.index).size() for d in exploded_df_list]
#     # index_df = pd.concat(index_sizes, axis=1)
#
# def store_idx_and_explode(df):
#     # index rest and re-merging for concat
#     df = df.explode(next(filter(lambda c : 'origin' not in c, list(df.columns))))
#     df.index.name = "old_idx"
#     df["num"] = range(0, len(df.index))
#     df = df.set_index("num", append=True)
#     return df

# def zip_and_explode(df:pd.DataFrame, cols_to_explode:List[str], output_level, lower_output_level, higher_output_level, same_output_level):
def zip_and_explode(df:pd.DataFrame,origin_cols_to_explode, origin_cols_not_to_explode, output_level):
    """ returns a NEW dataframe where cols_to_explode are all exploded together

    Used to extract SAME OUTPUT LEVEL annotator outputs.
    if cols_to_explode  are passed which are not at same output level, this will crash!
    Takes in a Dataframe and a list of columns to explode. Basically works like exploding multiple columns should work.
    Returns a new DataFrame, with cols_to_explode ziped and exploeded and the other column concatet back and left untouched otherwise

    This method needs to know
    1. What are columns at zip (same) level . They will be zip/exploded
    2. What are  columns higher than those at zip level. They will be unpacked from list
    3. What are columns below zip level? They will be left unotuched
    """


    # Some queries will result in index duplication
    same_level_cols_filter      = lambda c : any( og_c == c  for og_c in origin_cols_to_explode)
    cols_to_explode             = list(filter(same_level_cols_filter,df.columns))
    # not_same_level_cols_filter  = lambda c : any( og_c in c  for og_c in origin_cols_not_to_explode)
    # not_same_level_cols         = list(filter(not_same_level_cols_filter,df.columns))


    # if NUM component at same output level >1 and outputlevel is CHUNK we need the following padding logick
    if output_level == 'chunk' and len(cols_to_explode)>2:
        df[cols_to_explode] = df[cols_to_explode].apply(pad_same_level_cols,axis=1)

    pd_col_extractor_generator = lambda col : lambda x :  df[col]
    explode_series  = lambda s : s.explode()#.reset_index(drop=True)#.rename({'index':'origin_index'})#(drop=True)
    pd_col_extractors = list(map(pd_col_extractor_generator,cols_to_explode))
    # We call the pd series generator that needs a dummy call
    call = lambda x : x(0)
    list_of_pd_series_to_explod = list(map(call,pd_col_extractors))
    # Call explode on every series object, returns a list of pd.Series objects, which have been exploded
    exploded_series_list = list(map(explode_series,list_of_pd_series_to_explod))
    # Create pd.Dataframes from the pd.Series
    exploded_df_list = list(map(pd.DataFrame,exploded_series_list))
    # merge results into final pd.DataFrame
    try :
        merged_explosions = pd.concat([df.drop(cols_to_explode,axis=1)] +  exploded_df_list,axis=1)
    except:
        # if fails, try again but with  padding
        df[cols_to_explode] = df[cols_to_explode].apply(pad_same_level_cols,axis=1)
        pd_col_extractor_generator = lambda col : lambda x :  df[col]
        explode_series  = lambda s : s.explode()#.reset_index(drop=True)#.rename({'index':'origin_index'})#(drop=True)
        pd_col_extractors = list(map(pd_col_extractor_generator,cols_to_explode))
        # We call the pd series generator that needs a dummy call
        call = lambda x : x(0)
        list_of_pd_series_to_explod = list(map(call,pd_col_extractors))
        # Call explode on every series object, returns a list of pd.Series objects, which have been exploded
        exploded_series_list = list(map(explode_series,list_of_pd_series_to_explod))
        # Create pd.Dataframes from the pd.Series
        exploded_df_list = list(map(pd.DataFrame,exploded_series_list))
        # merge results into final pd.DataFrame
        merged_explosions = pd.concat([df.drop(cols_to_explode,axis=1)] +  exploded_df_list,axis=1)

    return merged_explosions



def pad_same_level_cols(row ):
    """We must ensure that the cols which are going to be exploded have all the same amount of elements.
    To ensure this,w e aply this methods on the cols we wish to explode. It ensures, they have all the same
    length and can be exploded eronous free
    """
    max_len = 0
    lens = {}
    for c in row.index :
        if isinstance(row[c],list):
            lens[c] = len(row[c])
            if lens[c] > max_len : max_len = lens[c]
        if isinstance(row[c],float):
            lens[c] = 1
            row[c]  = [row[c]]

    for c, lenght in lens.items():
        if lenght < max_len :
            row[c] += [np.nan] * (max_len-lenght)
    return row

"""
 We basically need to know the longest series for each Column and Index in the exploded serieses
1. get longest series for each element
2. Pad all series that are shortet than the longest for each index
2.1 Exploded fields that are too short paddedw ith NONE
2.2 Non-Exploded fields padded with repititoon of it self to fill the gap for index
"""