from typing import List, Dict, Set, Dict, Tuple, Optional, Any, Callable, Dict
from dataclasses import dataclass, field

"""
Contains dataclases which represent Extractors and their configuration

"""

from typing import List, Dict, Set, Dict, Tuple, Optional, Any
from typing import Callable
from typing import List, Dict, Set, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SparkNLPExtractor:
    """
    Define and describe a **extractor_method** that will be called on every annotator's output row.
    The extractor method will receive a list of dictionaries of type Dict[Str,Any] and must return a List[Any]
    It will receive a list of annotation results of a particular field for a given row of spark nlp pipeline transformation results.
    Wrap a extractor method, description and name in a dataclass that is used as part of a SparkNLPExtractorConfig

    Parameters
    ----------

    extractor_method    : Callable[[List[Dict[str,Any]]], List[Any]]  :
          An extractor is a method that takes in [Dict[str,Any]] and returns [Any].
          Will be applied to every output row's metadata for that particular
    description         : str
          Description
    name                :str
          Description
    """
    extractor_method: Callable[[List[Dict[str, Any]]], List[Any]] = field(repr=False, default=lambda x: x)
    description: str = field(default='')
    name: str = field(default='')
    extractor_with_result_method: \
        Optional[Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], List[Any]]] = field(repr=False, default=None)


## TODO extra config for getting "to which sentence did chunk/x/y/z belong to?"
@dataclass
class SparkNLPExtractorConfig:
    """
    Universal Configuration class for defining what data to extract from a Spark NLP annotator.
    These extractor configs can be passed to any extractor NLU defined for Spark-NLP.
    Setting a boolean config to false, results in the extractor NOT returning that field from the Annotator outputs
    Metadata Extractor methods will be applied to metadata after white/black list filtering has been applied to the fields
    Extractor methods for the fields

    output_col_prefix : str
          Prefix used for naming output columns from this annotator this config is applied to/
    description : str
          Describes how this configs affects te outputs
    name : str
          A name for this config
    get_positions : bool
          Get Annotation ending and beginning. If this is set to true, get_begin and get_end will be ignore and both positons will be outputted
    get_begin : bool
          Get Annotation beginnings
    get_end : bool
          Get Annotation ends
    get_embeds : bool
          Get Annotation Embeds
    get_result : bool
          Get Annotation results
    get_meta : bool
          Should get meta any meta at all? IF THIS FALSE, get_full_meta, whitelist and extractors will be ignored.
    get_sentence_origin : bool.
          Should extract from which sentence a prediction was generated from. If output level is Document, this field is irrelevant and should be set to False.
    get_full_meta : bool
          Get all keys and vals from base meta map. If this is true, white/blacklist will be ignored
    get_annotator_type : bool
          Get Annotator Type.
    pop_result_list : bool
          Should unpack the result field. Only set true for annotators that return exactly one element in their result, like Document classifier! This will convert list with just 1 element into just their element in the final pandas representation
    pop_result_list : bool
          Should unpack the result field. Only set true for annotators that return exactly one element in their result, like Document classifier! This will convert list with just 1 element into just their element in the final pandas representation
    pop_begin_list : bool
          Should unpack the begin field. Only set true for annotators that return exactly one element in their result, like Document classifier! This will convert list with just 1 element into just their element in the final pandas representation
    pop_end_list : bool
          Should unpack the end field. Only set true for annotators that return exactly one element in their result, like Document classifier! This will convert list with just 1 element into just their element in the final pandas representation
    pop_embed_list : bool
          Should unpack the embed field. Only set true for annotators that return exactly one element in their result, like Document classifier! This will convert list with just 1 element into just their element in the final pandas representation
    pop_meta_list : bool
          Should unpack the meta field. Only set true for annotators that return exactly one element in their result, like Document classifier! This will convert list with just 1 element into just their element in the final pandas representation
    meta_white_list : List[str]
          Whitelist some keys which should be fetched from meta map. If this is not [], meta_black_list will be ignored
    meta_black_list : List[str]
          black_list some keys which should not be fetched from meta map
    meta_data_extractor : List[str]
          An extractor is a method that takes in [Dict[str,Any]] and returs [Any]
    begin_extractor : SparkNLPExtractor
          DOCS
    end_extractor : SparkNLPExtractor
          DOCS
    result_extractor : SparkNLPExtractor
          DOCS
    embedding_extractor : SparkNLPExtractor
          DOCS

    """
    ## TODO pretty __repr__ or __to__string() method! Leverage  SparkNLPExtractor fields
    output_col_prefix: str
    get_positions: bool = field(default=False)
    get_begin: bool = field(default=False)
    get_end: bool = field(default=False)
    get_embeds: bool = field(default=False)
    get_result: bool = field(default=False)
    get_meta: bool = field(default=False)
    get_sentence_origin: bool = field(
        default=False)  # Should extract from which sentence a prediction was generated from. If output level is Document, this field is irrelevant and should be set to false
    get_full_meta: bool = field(default=False)
    get_annotator_type: bool = field(default=False)
    pop_result_list: bool = field(default=False)  # TODO implement in ex
    pop_begin_list: bool = field(default=False)  # TODO implement in ex
    pop_end_list: bool = field(default=False)  # TODO implement in ex
    pop_embeds_list: bool = field(default=False)  # TODO implement in ex
    pop_meta_list: bool = field(default=False)  # TODO implement in ex
    pop_never: bool = field(default=False)  # never ever pop
    meta_black_list: List[str] = field(default=list)
    meta_white_list: List[str] = field(default=list)
    meta_data_extractor: SparkNLPExtractor = field(default=SparkNLPExtractor())
    begin_extractor: SparkNLPExtractor = field(default=SparkNLPExtractor())
    end_extractor: SparkNLPExtractor = field(default=SparkNLPExtractor())
    result_extractor: SparkNLPExtractor = field(default=SparkNLPExtractor())
    embedding_extractor: SparkNLPExtractor = field(default=SparkNLPExtractor())
    description: str = field(default='')
    name: str = field(default='')


class SparkOCRExtractorConfig(SparkNLPExtractorConfig):
    get_text: bool = field(default=False)
    # Image struct fields
    get_image: bool = field(default=False)
    get_image_origin: bool = field(default=False)
    get_image_height: bool = field(default=False)
    get_image_width: bool = field(default=False)
    get_image_n_channels: bool = field(default=False)
    get_image_mode: bool = field(default=False)
    get_image_resolution: bool = field(default=False)
    get_image_data: bool = field(default=False)
    # General OCR fields
    # get_path          :bool              = field(default = False)# origin is path
    get_modification_time: bool = field(default=False)
    get_length: bool = field(default=False)
    get_page_num: bool = field(default=False)
    get_confidence: bool = field(default=False)
    get_exception: bool = field(default=False)
    # Position struct fields
    get_img_positions: bool = field(default=False)
