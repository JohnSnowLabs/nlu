"""
This module contains methods that will be applied inside of the apply call on the freshly convertet
Spark DF that is now as list of dicts.

These methods are meant to be applied in the internal calls of the metadata extraction.
They expect dictionaries which represent the metadata field extracted from Spark NLP annotators.


"""
import numpy as np
import pyspark
from pyspark.sql import Row as PysparkRow
from nlu.pipe.extractors.extractor_base_data_classes import *
from functools import reduce, partial
import pandas as pd

from sparknlp.annotation import Annotation


def extract_light_pipe_rows(df):
    """Extract Annotations from Light Pipeline into same represenation as other extractors in thos module"""
    ff = lambda row: list(map(f, row)) if isinstance(row, List) else row
    f = lambda anno: dict(annotatorType=anno.annotator_type,
                          begin=anno.begin, end=anno.end,
                          result=anno.result,
                          metadata=anno.metadata,
                          embeddings=anno.embeddings if isinstance(anno, List) else []) \
        if isinstance(anno, Annotation) else anno
    return df.applymap(ff)


def extract_pyspark_rows(r: pd.Series, ) -> pd.Series:
    """ Convert pyspark.sql.Row[Annotation] to List(Dict[str,str]) objects. Except for key=metadata in dict,
    this element in the Dict which is [str,Dict[str,str]] Checks if elements are of type list and whether they contain
    Pyspark Rows. If PysparkRow, call .asDict() on every row element to generate the dicts First method that runs
    after toPandas() call
    """
    if isinstance(r, str):
        return r
    elif isinstance(r, list):
        if len(r) == 0:
            return r
        elif isinstance(r[0], PysparkRow):
            pyspark_row_to_list = lambda l: l.asDict()
            return list(map(pyspark_row_to_list, r))
    return r


def extract_pyarrow_rows(r: pd.Series, ) -> pd.Series:
    """ Convert pyspark.sql.Row[Annotation] to List(Dict[str,str]) objects. Except for key=metadata in dict,
    this element in the Dict which is [str,Dict[str,str]]
    Checks if elements are of type list and whether they contain Pyspark Rows.
     If PysparkRow, call .asDict() on every row element to generate the dicts
    First method that runs after toPandas() call
    """
    if isinstance(r, str):
        return r
    elif isinstance(r, np.ndarray):
        if len(r) == 0:
            return r
        elif isinstance(r[0], dict) and 'annotatorType' in r[0].keys():
            r[0]['metadata'] = dict(r[0]['metadata'])
            return r
    return r


def extract_base_sparkocr_features(row: pd.Series, configs: SparkOCRExtractorConfig) -> dict:
    ###### OCR EXTRACTOR
    # for now only text recognizer outputs fetched
    if configs.name == 'default text recognizer config':
        if configs.get_text:
            return {'text': row}
    # Check for primitive type here and return
    if 'visual_classifier' in configs.name:
        # Either Label or Confidence
        if isinstance(row, str):
            return {'visual_classifier_label': row}
        else:
            return {'visual_classifier_confidence': row}

    else:
        # # OCR unpackers (TODO WIP)
        # unpack_text = lambda x: unpack_dict_list(x, 'text')
        # # unpack_image = lambda x : unpack_dict_list(x, 'TODO') # is data?
        # unpack_image_origin = lambda x: unpack_dict_list(x, 'origin')
        # unpack_image_height = lambda x: unpack_dict_list(x, 'height')
        # unpack_image_width = lambda x: unpack_dict_list(x, 'width')
        # unpack_image_n_channels = lambda x: unpack_dict_list(x, 'nChannels')
        # unpack_image_mode = lambda x: unpack_dict_list(x, 'mode')
        # unpack_image_resolution = lambda x: unpack_dict_list(x, 'resolution')
        # unpack_image_data = lambda x: unpack_dict_list(x, 'data')
        # # unpack_path = lambda x : unpack_dict_list(x, 'TODO')
        # unpack_modification_time = lambda x: unpack_dict_list(x, 'TODO')
        # unpack_length = lambda x: unpack_dict_list(x, 'TODO')
        # unpack_page_num = lambda x: unpack_dict_list(x, 'TODO')
        # unpack_confidence = lambda x: unpack_dict_list(x, 'TODO')
        # unpack_exception = lambda x: unpack_dict_list(x, 'TODO')
        # unpack_img_positions = lambda x: unpack_dict_list(x, 'TODO')
        # if configs.get_image:
        #     pass
        # if configs.get_image_origin:
        #     pass
        # if configs.get_image_height:
        #     pass
        # if configs.get_image_width:
        #     pass
        # if configs.get_image_n_channels:
        #     pass
        # if configs.get_image_mode:
        #     pass
        # if configs.get_image_resolution:
        #     pass
        # if configs.get_image_data:
        #     pass
        # if configs.get_path:
        #     pass
        # if configs.get_modification_time:
        #     pass
        # if configs.get_length:
        #     pass
        # if configs.get_page_num:
        #     pass
        # if configs.get_confidence:
        #     pass
        # if configs.get_exception:
        #     pass
        # if configs.get_img_positions:
        #     pass
        # return {**beginnings, **endings, **results, **annotator_types, **embeddings}  # Merge dicts OCR output

        return {}


def extract_base_sparknlp_features(row: pd.Series, configs: SparkNLPExtractorConfig) -> dict:
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

    unpack_dict_list = lambda d, k: d[k]
    unpack_begin = lambda x: unpack_dict_list(x, 'begin')
    unpack_end = lambda x: unpack_dict_list(x, 'end')
    unpack_annotator_type = lambda x: unpack_dict_list(x, 'annotatorType')
    unpack_result = lambda x: unpack_dict_list(x, 'result')
    unpack_embeddings = lambda x: unpack_dict_list(x, 'embeddings')

    # Either extract list of anno results and put them in a dict with corrosponding key name or return empty dict {} for easy merge in return
    annotator_types = {configs.output_col_prefix + '_types': list(
        map(unpack_annotator_type, row))} if configs.get_annotator_type else {}
    # Same logic as above, but we check wether to pop or not and either evaluate the map result with list() or just next()
    if configs.pop_result_list:
        results = {configs.output_col_prefix + '_results': next(map(unpack_result, row))} if configs.get_result else {}
    else:
        results = {configs.output_col_prefix + '_results': list(map(unpack_result, row))} if configs.get_result else {}
    if configs.pop_begin_list:
        beginnings = {configs.output_col_prefix + '_beginnings': next(
            map(unpack_begin, row))} if configs.get_begin or configs.get_positions else {}
    else:
        beginnings = {configs.output_col_prefix + '_beginnings': list(
            map(unpack_begin, row))} if configs.get_begin or configs.get_positions else {}

    if configs.pop_end_list:
        endings = {configs.output_col_prefix + '_endings': next(
            map(unpack_end, row))} if configs.get_end or configs.get_positions else {}
    else:
        endings = {configs.output_col_prefix + '_endings': list(
            map(unpack_end, row))} if configs.get_end or configs.get_positions else {}

    if configs.pop_embeds_list:
        embeddings = {
            configs.output_col_prefix + '_embeddings': next(map(unpack_embeddings, row))} if configs.get_embeds else {}
    else:
        embeddings = {
            configs.output_col_prefix + '_embeddings': list(map(unpack_embeddings, row))} if configs.get_embeds else {}

    return {**beginnings, **endings, **results, **annotator_types, **embeddings}  # Merge dicts NLP output


def extract_sparknlp_metadata(row: pd.Series, configs: SparkNLPExtractorConfig) -> dict:
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
    unpack_dict_list = lambda d, k: d[k]
    # extract list of metadata dictionaries (all dict should have same keys)
    unpack_metadata_to_dict_list = lambda x: unpack_dict_list(x, 'metadata')

    metadatas_dict_list = list(map(unpack_metadata_to_dict_list, row))
    # extract keys, which should all be equal in all rows

    if configs.get_full_meta:
        keys_in_metadata = list(metadatas_dict_list[0].keys()) if len(metadatas_dict_list) > 0 else []
    elif len(configs.meta_white_list) != 0:
        keys_in_metadata = [k for k in metadatas_dict_list[0].keys() if k in configs.meta_white_list]
    elif len(configs.meta_black_list) != 0:
        keys_in_metadata = [k for k in metadatas_dict_list[0].keys() if k not in configs.meta_black_list]
    else:
        keys_in_metadata = []

    # dectorate lambda with key to extract, equalt to def decorate_f(key): return lambda x,y :  x+ [y[key]]
    # For a list of dicts which all have the same keys, will return a list of all the values for one key in all the dicts
    if configs.pop_meta_list:
        f = lambda key: metadatas_dict_list[0][key]
        metadata_scalars = list(map(f, keys_in_metadata))
        result = dict(
            zip(map(lambda x: 'meta_' + configs.output_col_prefix + '_' + x, keys_in_metadata), metadata_scalars))
        return result
    extract_val_from_dic_list_to_list = lambda key: lambda x, y: x + [y[key]]
    # List of lambda expression, on for each Key to be extracted. (TODO balcklisting?)
    dict_value_extractors = list(map(extract_val_from_dic_list_to_list, keys_in_metadata))
    # reduce list of dicts with same struct and a common key to a list of values for thay key. Leveraging closuer for meta_dict_list
    reduce_dict_list_to_values = lambda t: reduce(t, metadatas_dict_list, [])
    # list of lists, where each list is corrosponding to all values in the previous dict list
    meta_values_list = list(map(reduce_dict_list_to_values, dict_value_extractors))
    # add prefix to key and zip with values for final dict result
    result = dict(
        zip(list(map(lambda x: 'meta_' + configs.output_col_prefix + '_' + x, keys_in_metadata)), meta_values_list))
    return result


def extract_master(row: pd.Series, configs: SparkNLPExtractorConfig) -> pd.Series:
    """
    Re-Usable base extractor for simple Annotators like Document/Token/etc..?
    extract_universal/?/Better name?
    row = a list or Spark-NLP annotations as dictionary
    """
    if isinstance(row, pyspark.sql.Row) and len(row) == 0:
        return pd.Series({})
    if isinstance(configs, SparkOCRExtractorConfig):
        base_annos = extract_base_sparkocr_features(row, configs)
    else:
        base_annos = extract_base_sparknlp_features(row, configs)
    # Get Metadata
    all_metas = extract_sparknlp_metadata(row, configs) if configs.get_meta or configs.get_full_meta else {}

    # Apply custom extractor methods
    if configs.meta_data_extractor.name != '':
        if configs.meta_data_extractor.extractor_with_result_method:
            all_metas = configs.meta_data_extractor.extractor_with_result_method(all_metas, base_annos, configs)
        else:
            all_metas = configs.meta_data_extractor.extractor_method(all_metas, configs)

    # Apply Finishers on metadata/additional fields
    return pd.Series(
        {
            **base_annos,
            **all_metas
        })


def apply_extractors_and_merge(df, anno_2_ex_config, keep_stranger_features, stranger_features):
    """ apply extract_master on all fields with corrosponding configs after converting Pyspark Rows to List[Dict]
    and merge them to a final DF (1 to 1 mapping still)
    df  The Df we want to apply the extractors on
    columns_to_extractor_map Map column names to extractor configs. Columns which are not in these keys will be ignored These configs will be passed to master_extractor for every column
    """
    # keep df and ex_resolver in closure and apply base extractor with configs for each col
    extractor = lambda c: df[c].apply(extract_master, configs=anno_2_ex_config[c])
    keep_strangers = lambda c: df[c]

    # merged_extraction_df
    # apply the extract_master together with it's configs to every column and geenrate a list of output DF's, one per Spark NLP COL
    # TODO handle MULTI-COL-OUTPUT. If Anno has multi cols, then we either needs multiple keys in anno_2_ex or use something besides
    # anno_2_ex_config.keys() here because it will only apply to one of the extracted rows..(?)

    # Apply each Anno Extractor to the corrosponding generated col.
    # If no Extractor defined for a col and it is not a stranger feature, it will be dropped here
    return pd.concat(
        list(map(extractor, anno_2_ex_config.keys())) +
        list(map(keep_strangers, stranger_features)) if keep_stranger_features else [],
        axis=1)


def pad_same_level_cols(row):
    """We must ensure that the cols which are going to be exploded have all the same amount of elements.
    To ensure this, we apply this methods on the cols we wish to explode. It ensures, they have all the same
    length and can be exploded eronous free
    """
    max_len = 0
    lens = {}
    for c in row.index:
        if isinstance(row[c], list):
            lens[c] = len(row[c])
            if lens[c] > max_len: max_len = lens[c]
        else:
            lens[c] = 1
            row[c] = [row[c]]

    for c, length in lens.items():
        if length < max_len:
            row[c] += [np.nan] * (max_len - length)
    return row


def zip_and_explode(df: pd.DataFrame, cols_to_explode: List[str]) -> pd.DataFrame:
    """
    Returns a new dataframe, where columns in cols_to_explode should have all array elements.
    :param df: Dataframe to explode columns on. Each column in cols_to_explode should be of type array
    :param cols_to_explode: list of columns to explode
    :return: new dataframe, where each array element of a row in cols_to_explode is in a new row.
            For exploding Rows where the lists are same length,
            lists will be padded to length of the longest list in that row (sub-levels)
            Elements of columns which are not in cols_to_explode, will be in lists
    """
    # Check cols we want to explode actually exist, if no data extracted cols can be missing
    missing = []
    for col in cols_to_explode:
        if col not in df.columns:
            missing.append(col)
    for miss in missing:
        cols_to_explode.remove(miss)
    # Drop duplicate cols
    df = df.loc[:, ~df.columns.duplicated()]
    if len(cols_to_explode) > 0:
        # We must pad all cols we want to explode to the same length because pandas limitation.
        # Spark API does not require this since it handles cols with not same length by creating nan. We do it ourselves here manually
        df[cols_to_explode] = df[cols_to_explode].apply(pad_same_level_cols, axis=1)
        return pd.concat([df.explode(c)[c] for c in cols_to_explode] + [df.drop(cols_to_explode, axis=1)], axis=1)
    else:
        # No padding
        return pd.concat([df.drop(cols_to_explode, axis=1)], axis=1)
