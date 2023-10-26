import logging
import os
from typing import Optional
from typing import Optional
import os
import sparknlp
from pyspark.sql.functions import monotonically_increasing_id
from sparknlp.common import AnnotatorType

from nlu.pipe.utils.audio_data_conversion_utils import AudioDataConversionUtils
from nlu.pipe.utils.data_conversion_utils import DataConversionUtils
from nlu.pipe.utils.ocr_data_conversion_utils import OcrDataConversionUtils

logger = logging.getLogger('nlu')
from nlu.pipe.pipe_logic import PipeUtils

import pandas as pd
from pydantic import BaseModel


def get_first_anno_with_output_type(pipe, out_type):
    for s in pipe.vanilla_transformer_pipe.stages:
        if hasattr(s, 'outputAnnotatorType') and s.outputAnnotatorType == out_type:
            return s
    return None


def serialize(img_path):
    with open(img_path, 'rb') as img_file:
        return img_file.read()


def deserialize(binary_image, path):
    with open(path, 'wb') as img_file:
        img_file.write(binary_image)


class PredictParams(BaseModel):
    output_level: Optional[str] = ''
    positions: Optional[bool] = False
    keep_stranger_features: Optional[bool] = True
    metadata: Optional[bool] = False
    multithread: Optional[bool] = True
    drop_irrelevant_cols: Optional[bool] = True
    return_spark_df: Optional[bool] = False
    get_embeddings: Optional[bool] = True

    @staticmethod
    def has_param_cols(df: pd.DataFrame):
        return all([c not in df.columns for c in PredictParams.__fields__.keys()])

    @staticmethod
    def maybe_from_pandas_df(df: pd.DataFrame):
        # only first row is used
        if df.shape[0] == 0:
            return None
        if PredictParams.has_param_cols(df):
            # no params in df
            return None
        param_row = df.iloc[0].to_dict()
        try:
            return PredictParams(**param_row)
        except Exception as e:
            print(f'Exception trying to parse prediction parameters for param row:'
                  f' \n{param_row} \n', e)
            return None


def get_first_anno_with_output_type(pipe, out_type):
    for s in pipe.vanilla_transformer_pipe.stages:
        if hasattr(s, 'outputAnnotatorType') and s.outputAnnotatorType == out_type:
            return s
    return None


def __predict_standard_spark(pipe, data, output_level, positions, keep_stranger_features, metadata,
                             drop_irrelevant_cols, return_spark_df, get_embeddings):
    # 1. Convert data to Spark DF
    data, stranger_features, output_datatype = DataConversionUtils.to_spark_df(data, pipe.spark, pipe.raw_text_column,
                                                                               is_span_data=pipe.has_span_classifiers,
                                                                               is_tabular_qa_data=pipe.has_table_qa_models,
                                                                               )

    # 3. Apply Spark Pipeline
    data = pipe.vanilla_transformer_pipe.transform(data)

    # 4. Convert resulting spark DF into nicer format and by default into pandas.
    if return_spark_df: return data  # Returns RAW  Spark Dataframe result of component_list prediction
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

    # Some annos require `image` format, some will require `binary` format. We need to figure out which one is needed possible provide both
    if pipe.requires_image_format and pipe.requires_binary_format:
        from pyspark.sql.functions import regexp_replace
        # Image & Binary formats required. We read as both and join the dfs
        img_df = spark.read.format("image").load(file_paths).withColumn("modified_origin",
                                                                        regexp_replace("image.origin", ":/{1,}", ":"))

        # Read the files in binaryFile format
        binary_df = spark.read.format("binaryFile").load(file_paths).withColumn("modified_path",
                                                                                regexp_replace("path", ":/{1,}", ":"))

        data = img_df.join(binary_df, img_df["modified_origin"] == binary_df["modified_path"]).drop('modified_path')

    elif pipe.requires_image_format:
        # only image format required
        data = spark.read.format("image").load(file_paths)
    elif pipe.requires_binary_format:
        # only binary required
        data = spark.read.format("binaryFile").load(file_paths)
    else:
        # fallback default
        data = spark.read.format("binaryFile").load(file_paths)
    data = data.withColumn('origin_index', monotonically_increasing_id().alias('origin_index'))

    data = pipe.vanilla_transformer_pipe.transform(data)
    return pipe.pythonify_spark_dataframe(data,
                                          keep_stranger_features=keep_stranger_features,
                                          output_metadata=metadata,
                                          drop_irrelevant_cols=drop_irrelevant_cols,
                                          positions=positions,
                                          output_level=output_level,
                                          get_embeddings=get_embeddings
                                          )


def __predict_audio_spark(pipe, data, output_level, positions, keep_stranger_features, metadata,
                          drop_irrelevant_cols, get_embeddings):
    """
        Check if there are any OCR components in the Pipe.
        If yes, we verify data contains pointer to jsl_folder or image files.
        If yes, df = spark.read.format("binaryFile").load(imagePath)
        Run OCR pipe on df and pythonify procedure afterwards

    """
    pipe.fit()

    try:
        import librosa
    except:
        raise ImportError('The librosa library is not installed and required for audio features! '
                          'Run pip install librosa ')
    sample_rate = 16000
    AudioDataConversionUtils.validate_paths(data)
    paths = AudioDataConversionUtils.extract_iterable_paths_from_data(data)
    accepted_file_types = AudioDataConversionUtils.get_accepted_audio_file_types(pipe)
    file_paths = AudioDataConversionUtils.glob_files_of_accepted_type(paths, accepted_file_types)
    data = AudioDataConversionUtils.data_to_spark_audio_df(data=file_paths, sample_rate=sample_rate,
                                                           spark=sparknlp.start())
    data = pipe.vanilla_transformer_pipe.transform(data).withColumn(
        'origin_index', monotonically_increasing_id().alias('origin_index'))

    return pipe.pythonify_spark_dataframe(data,
                                          keep_stranger_features=keep_stranger_features,
                                          output_metadata=metadata,
                                          drop_irrelevant_cols=drop_irrelevant_cols,
                                          positions=positions,
                                          output_level=output_level,
                                          get_embeddings=get_embeddings
                                          )


def __db_endpoint_predict__(pipe, data):
    """
    1) parse pred params from first row maybe
    2) serialize/deserialize img
    """
    print("CUSOTM NLU MODE!")
    print(data.columns)
    params = PredictParams.maybe_from_pandas_df(data)
    if params:
        params = params.dict()
    else:
        params = {}
    files = []
    if 'file' in data.columns and 'file_type' in data.columns:
        print("DETECTED FILE COLS")
        skip_first = PredictParams.has_param_cols(data)
        for i, row in data.iterrows():
            print(f"DESERIALIZING {row.file_type} file {row.file}")
            if i == 0 and skip_first:
                continue
            file_name = f'file{i}.{row.file_type}'
            files.append(file_name)
            deserialize(row.file, file_name)
        data = files

    if params:
        return __predict__(pipe, data, **params, normal_pred_on_db=True)
    else:
        # no params detect, we call again with default params
        return __predict__(pipe, data, **PredictParams().dict(), normal_pred_on_db=True)


def __predict_standard_spark_only_embed(pipe, data, return_spark_df):
    # 1. Convert data to Spark DF
    data, stranger_features, output_datatype = DataConversionUtils.to_spark_df(data, pipe.spark, pipe.raw_text_column,
                                                                               is_span_data=pipe.has_span_classifiers,
                                                                               is_tabular_qa_data=pipe.has_table_qa_models,
                                                                               )

    # 2. Apply Spark Pipeline
    data = pipe.vanilla_transformer_pipe.transform(data)

    # 3. Validate data
    sent_embedder = get_first_anno_with_output_type(pipe, AnnotatorType.SENTENCE_EMBEDDINGS)
    if not sent_embedder or not hasattr(sent_embedder, 'getOutputCol', ):
        raise Exception('No Sentence Embedder found in pipeline')

    # 4. return embeds
    emb_col = sent_embedder.getOutputCol()
    if return_spark_df:
        return data.select(f'{emb_col}.embeddings')

    # Note, this is only document output level. If pipe has sentence detector, we will only keep first embed of every document.
    return [r.embeddings[0] for r in data.select(f'{emb_col}.embeddings').collect()]


def __predict__(pipe, data, output_level, positions, keep_stranger_features, metadata, multithread,
                drop_irrelevant_cols, return_spark_df, get_embeddings, embed_only=False,normal_pred_on_db=False):
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
    if embed_only:
        pipe.fit()
        return __predict_standard_spark_only_embed(pipe, data, return_spark_df)

    if 'DB_ENDPOINT_ENV' in os.environ and not normal_pred_on_db:
        return __db_endpoint_predict__(pipe,data)

    if output_level == '' and not pipe.has_table_qa_models:
        # Default sentence level for all components
        if pipe.has_nlp_components and not PipeUtils.contains_t5_or_gpt(
                pipe) and not pipe.has_span_classifiers:
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

        pipe.__configure_light_pipe_usage__(DataConversionUtils.size_of(data), multithread)

    if pipe.contains_ocr_components and pipe.contains_audio_components:
        """ Idea:
        Expect Array of Paths 
        For every path classify file ending and use it to correctly handle Img or Audio stuff 
        """
        raise Exception('Cannot mix Audio and OCR components in a Pipe?')

    if pipe.contains_audio_components:
        return __predict_audio_spark(pipe, data, output_level, positions, keep_stranger_features,
                                     metadata, drop_irrelevant_cols, get_embeddings=get_embeddings)

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
