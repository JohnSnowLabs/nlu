import glob
import logging
from typing import List
import os
import numpy as np
import pyspark
import sparknlp
from pyspark.sql.functions import monotonically_increasing_id

from nlu.pipe.utils.ocr_data_conversion_utils import OcrDataConversionUtils

logger = logging.getLogger('nlu')
from nlu.pipe.pipe_logic import PipeUtils
import pandas as pd
from nlu.pipe.utils.data_conversion_utils import DataConversionUtils


def __predict_standard_spark(pipe, data, output_level, positions, keep_stranger_features, metadata,
                             drop_irrelevant_cols, return_spark_df, get_embeddings):
    # 1. Convert data to Spark DF
    data, stranger_features, output_datatype = DataConversionUtils.to_spark_df(data, pipe.spark, pipe.raw_text_column, pipe.has_span_classifiers)

    # 3. Apply Spark Pipeline
    data = pipe.vanilla_transformer_pipe.transform(data)

    # 4. Convert resulting spark DF into nicer format and by default into pandas.
    if return_spark_df: return data   # Returns RAW  Spark Dataframe result of component_list prediction
    return pipe.pythonify_spark_dataframe(data,
                                          keep_stranger_features=keep_stranger_features,
                                          stranger_features=stranger_features,
                                          output_metadata=metadata,
                                          drop_irrelevant_cols=drop_irrelevant_cols,
                                          positions=positions,
                                          output_level=output_level,
                                          get_embeddings=get_embeddings
                                          )


def predict_multi_threaded_light_pipe(pipe, data, output_level, positions, keep_stranger_features, metadata,
                                      drop_irrelevant_cols, get_embeddings):
    # 1. Try light component_list predcit
    # 2. if vanilla Fails use Vanilla
    # 3.  if vanilla fails raise error
    data, stranger_features, output_datatype = DataConversionUtils.to_pandas_df(data, pipe.raw_text_column)

    # Predict -> Cast to PDF -> Join with original inputs. It does NOT yield EMBEDDINGS.
    data = data.join(pd.DataFrame(pipe.light_transformer_pipe.fullAnnotate(data.text.values)))

    return pipe.pythonify_spark_dataframe(data,
                                          keep_stranger_features=keep_stranger_features,
                                          stranger_features=stranger_features,
                                          output_metadata=metadata,
                                          drop_irrelevant_cols=drop_irrelevant_cols,
                                          positions=positions,
                                          output_level=output_level,
                                          get_embeddings=get_embeddings
                                          )


def __predict_ocr_spark(pipe, data, output_level, positions, keep_stranger_features, metadata,
                        drop_irrelevant_cols, get_embeddings):
    """
        Check if there are any OCR components in the Pipe.
        If yes, we verify data contains pointer to jsl_folder or image files.
        If yes, df = spark.read.format("binaryFile").load(imagePath)
        Run OCR pipe on df and pythonify procedure afterwards

    """
    pipe.fit()

    OcrDataConversionUtils.validate_OCR_compatible_inputs(data)
    paths = OcrDataConversionUtils.extract_iterable_paths_from_data(data)
    accepted_file_types = OcrDataConversionUtils.get_accepted_ocr_file_types(pipe)
    file_paths = OcrDataConversionUtils.glob_files_of_accepted_type(paths, accepted_file_types)
    spark = sparknlp.start()  # Fetches Spark Session that has already been licensed
    data = pipe.vanilla_transformer_pipe.transform(spark.read.format("binaryFile").load(file_paths)).withColumn(
        'origin_index', monotonically_increasing_id().alias('origin_index'))
    return pipe.pythonify_spark_dataframe(data,
                                          keep_stranger_features=keep_stranger_features,
                                          output_metadata=metadata,
                                          drop_irrelevant_cols=drop_irrelevant_cols,
                                          positions=positions,
                                          output_level=output_level,
                                          get_embeddings=get_embeddings
                                          )


def __predict__(pipe, data, output_level, positions, keep_stranger_features, metadata, multithread,
                drop_irrelevant_cols, return_spark_df, get_embeddings):
    '''
    Annotates a Pandas Dataframe/Pandas Series/Numpy Array/Spark DataFrame/Python List strings /Python String
    :param data: Data to predict on
    :param output_level: output level, either document/sentence/chunk/token
    :param positions: whether to output indexes that map predictions back to position in origin string
    :param keep_stranger_features: whether to keep columns in the dataframe that are not generated by pandas. I.e. when you s a dataframe with 10 columns and only one of them is named text, the returned dataframe will only contain the text column when set to false
    :param metadata: whether to keep additional metadata in final df or not like confidences of every possible class for preidctions.
    :param multithread: Whether to use multithreading based light pipeline. In some cases, this may cause errors.
    :param drop_irellevant_cols: Whether to drop cols of different output levels, i.e. when predicting token level and dro_irrelevant_cols = True then chunk, sentence and Doc will be dropped
    :param return_spark_df: Prediction results will be returned right after transforming with the Spark NLP pipeline
    :return:
    '''

    if output_level == '':
        # Default sentence level for all components
        if pipe.has_nlp_components and not PipeUtils.contains_T5_or_GPT_transformer(pipe) and not pipe.has_span_classifiers:
            pipe.component_output_level = 'sentence'
            pipe.components = PipeUtils.configure_component_output_levels(pipe, 'sentence')
    else:
        if pipe.has_nlp_components and output_level in ['document', 'sentence']:
            # Pipe must be re-configured for document/sentence level
            pipe.component_output_level = output_level
            pipe.components = PipeUtils.configure_component_output_levels(pipe, output_level)

        elif pipe.has_nlp_components and output_level in ['token']:
            # Add tokenizer if not in pipe, default its inputs to sentence
            pipe.component_output_level = 'sentence'
            pipe.components = PipeUtils.configure_component_output_levels(pipe, 'sentence')
            pipe = PipeUtils.add_tokenizer_to_pipe_if_missing(pipe)

    if get_embeddings is None:
        # Grab embeds if nlu ref is of type embed
        get_embeddings = True if 'embed' in pipe.nlu_ref else False

    if not pipe.is_fitted:
        if pipe.has_trainable_components:
            pipe.fit(data)
        else:
            pipe.fit()

    # configure Lightpipline usage
    pipe.configure_light_pipe_usage(DataConversionUtils.size_of(data), multithread)

    if pipe.contains_ocr_components:
        # Ocr processing
        try:
            return __predict_ocr_spark(pipe, data, output_level, positions, keep_stranger_features,
                                       metadata, drop_irrelevant_cols, get_embeddings=get_embeddings)
        except Exception as err:
            logger.warning(f"Predictions Failed={err}")
            pipe.print_exception_err(err)
            raise Exception("Failure to process data with NLU OCR pipe")
    if return_spark_df:
        try:
            return __predict_standard_spark(pipe, data, output_level, positions, keep_stranger_features, metadata,
                                            drop_irrelevant_cols, return_spark_df, get_embeddings)
        except Exception as err:
            logger.warning(f"Predictions Failed={err}")
            pipe.print_exception_err(err)
            raise Exception("Failure to process data with NLU")
    elif not get_embeddings and multithread or pipe.prefer_light:
        # In Some scenarios we prefer light, because Bugs in ChunkMapper...
        # Try Multithreaded with Fallback vanilla as option. No Embeddings in this mode
        try:
            return predict_multi_threaded_light_pipe(pipe, data, output_level, positions, keep_stranger_features,
                                                     metadata, drop_irrelevant_cols, get_embeddings=get_embeddings)


        except Exception as err:
            logger.warning(
                f"Multithreaded mode with Light pipeline failed. trying to predict again with non multithreaded mode, "
                f"err={err}")
            try:
                return __predict_standard_spark(pipe, data, output_level, positions, keep_stranger_features, metadata,
                                                drop_irrelevant_cols, return_spark_df, get_embeddings)
            except Exception as err:
                logger.warning(f"Predictions Failed={err}")
                pipe.print_exception_err(err)
                raise Exception("Failure to process data with NLU")
    else:
        # Standard predict with no fallback
        try:
            return __predict_standard_spark(pipe, data, output_level, positions, keep_stranger_features, metadata,
                                            drop_irrelevant_cols, return_spark_df, get_embeddings)
        except Exception as err:
            logger.warning(f"Predictions Failed={err}")
            pipe.print_exception_err(err)
            raise Exception("Failure to process data with NLU")


def debug_print_pipe_cols(pipe):
    for c in pipe.components:
        print(f'{c.spark_input_column_names}->{c.name}->{c.spark_output_column_names}')

