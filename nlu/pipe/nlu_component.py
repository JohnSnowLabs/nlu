from dataclasses import dataclass
from typing import Dict, Callable, Optional, List, Any, Union

from sparknlp.common import AnnotatorApproach, AnnotatorModel
from sparknlp.internal import AnnotatorTransformer

from nlu.universe.logic_universes import AnnoTypes
from nlu.universe.universes import ComponentBackends
from nlu.universe.atoms import NlpLevel, LicenseType, JslAnnoId, JslAnnoPyClass, JslAnnoJavaClass, LanguageIso, \
    JslFeature
from nlu.universe.feature_node_universes import NlpFeatureNode


def debug_print_pipe_cols(pipe):
    for c in pipe.components:
        print(f'{c.spark_input_column_names}->{c.name}->{c.spark_output_column_names}')


@dataclass
class NluComponent:
    """Contains various metadata about the loaded component"""

    name: str  # Name for this anno
    type: str  # this tells us which kind of component this is
    # Extractor method applicable to Pandas DF for getting pretty outputs
    pdf_extractor_methods: Dict[str, Callable[[], any]]
    pdf_col_name_substitutor: Callable[[], any]  # substitution method for renaming final cols to somthing redable
    # sdf_extractor_methods : Dict[str,Callable[[],any]] # Extractor method applicable to Spark  DF for getting pretty outputs # TODO NOT IN BUILD
    # sdf_col_name_substitutor : Optional[Callable[[],any]] # substitution method for renaming final cols to somthing redable # TODO NOT IN BUILD
    output_level: NlpLevel  # Output level of the component for data transformation logic or call it putput mapping??
    node: NlpFeatureNode  # Graph node
    description: str  # general annotator/model/component/pipeline info
    provider: ComponentBackends  # Who provides the implementation of this annotator, Spark-NLP for base. Would be
    license: LicenseType  # open source or private
    computation_context: str  # Will this component do its computation in Spark land (like all of Spark NLP annotators do) or does it require some other computation engine or library like Tensorflow, Numpy, HuggingFace, etc..
    output_context: str  # Will this components final result
    jsl_anno_class_id: JslAnnoId  # JSL Annotator Class this belongs to
    jsl_anno_py_class: JslAnnoPyClass  # JSL Annotator Class this belongs to
    get_default_model: Optional[Callable[[], AnnotatorTransformer]] = None  # Returns Concrete JSL Annotator object.
    # Returns Concrete JSL Annotator object. May by None lang,name, bucket
    get_pretrained_model: Optional[Callable[[str, str, str], AnnotatorTransformer]] = None
    # Returns Concrete JSL Annotator object. May by None
    get_trainable_model: Optional[Callable[[], AnnotatorTransformer]] = None
    trainable: bool = False
    language: [LanguageIso] = None
    # constructor_args: ComponentConstructorArgs = None  # Args used to originally create this component
    nlu_ref: str = None
    nlp_ref: str = None
    in_types: List[JslFeature] = None
    out_types: List[JslFeature] = None
    in_types_default: List[JslFeature] = None
    out_types_default: List[JslFeature] = None
    spark_input_column_names: List[str] = None
    spark_output_column_names: List[str] = None
    paramMap: Dict[Any, Any] = None
    paramSetterMap: Dict[Any, Any] = None
    paramGetterMap: Dict[Any, Any] = None
    # Any anno class. Needs to be Any, so we can cover unimported HC models
    model: Union[AnnotatorApproach, AnnotatorModel] = None
    storage_ref: Optional[str] = None
    storage_ref_nlu_ref_resolution: Optional[str] = None  # nlu_ref corresponding to storage_ref
    loaded_from_pretrained_pipe: bool = False  # If this component was derived from a pre-build SparkNLP pipeline or from NLU
    has_storage_ref: bool = False
    is_storage_ref_consumer: bool = False  # # Whether this anno takes in some features that are storage ref based
    is_storage_ref_producer: bool = False  # Whether this anno generates some features that are storage ref based
    # Reference to trainable version of this anno, if this is a non-trainable anno otherwise None
    trainable_mirror_anno: Optional[JslAnnoId] = None
    # Reference to trained version of this anno, if this is a trainable anno otherwise None
    trained_mirror_anno: Optional[JslAnnoId] = None
    applicable_file_types : List[str]  = None # Used for OCR annotators to deduct applicable file types
    is_trained : bool = True # Set to true for trainable annotators

    def set_metadata(self, jsl_anno_object: Union[AnnotatorApproach, AnnotatorModel],
                     nlu_ref: str,
                     nlp_ref: str,
                     language: LanguageIso,
                     loaded_from_pretrained_pipe: bool,
                     license_type: LicenseType,
                     storage_ref: Optional[str] = None):
        """Write metadata to nlu component after constructing it """
        self.model = jsl_anno_object
        self.nlu_ref = nlu_ref
        self.nlp_ref = nlp_ref
        self.language = language
        self.loaded_from_pretrained_pipe = loaded_from_pretrained_pipe
        self.license = license_type
        self.in_types = self.node.ins.copy()
        self.out_types = self.node.outs.copy()
        self.in_types_default = self.node.ins.copy()
        self.out_types_default = self.node.outs.copy()
        self.spark_input_column_names = self.in_types.copy()
        self.spark_output_column_names = self.out_types.copy()
        if storage_ref:
            self.storage_ref = storage_ref
        if nlp_ref == 'glove_840B_300' or nlp_ref == 'glove_6B_300':
            self.lang = 'xx'
        if hasattr(self.model, 'setIncludeConfidence'):
            self.model.setIncludeConfidence(True)
        # if self.has_storage_ref and 'converter' in self.name:
        from nlu.universe.feature_node_ids import NLP_NODE_IDS
        if self.name in [NLP_NODE_IDS.SENTENCE_EMBEDDINGS_CONVERTER,NLP_NODE_IDS.CHUNK_EMBEDDINGS_CONVERTER]:
            # Embedding converters initially have
            self.storage_ref = ''
        if self.trainable:
            self.is_trained = False
        return self

    def __str__(self):
        return f'Component(ID={self.name}, NLU_REF={self.nlu_ref} NLP_REF={self.nlp_ref})'

    def __hash__(self):
        return hash((self.name, self.nlu_ref, self.nlp_ref, self.jsl_anno_class_id, self.language))
