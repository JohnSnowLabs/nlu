import unittest
from sparknlp.base import *
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
import nlu
import tests.secrets as sct

from sparknlp.pretrained import PretrainedPipeline


class PartiallyImplementedComponentsCase(unittest.TestCase):
    class SentenceResolutionTests(unittest.TestCase):
        SPARK_NLP_LICENSE = sct.SPARK_NLP_LICENSE
        AWS_ACCESS_KEY_ID = sct.AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY = sct.AWS_SECRET_ACCESS_KEY
        JSL_SECRET = sct.JSL_SECRET
        nlu.auth(
            SPARK_NLP_LICENSE, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, JSL_SECRET
        )

    def test_partially_implemented_handling(self):
        documentAssembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentenceDetector = SentenceDetector() \
            .setInputCols("document") \
            .setOutputCol("sentence")

        tokenizer = Tokenizer() \
            .setInputCols("sentence") \
            .setOutputCol("token")

        word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
            .setInputCols("sentence", "token") \
            .setOutputCol("word_embeddings")

        # to get PROBLEM entitis
        clinical_ner = MedicalNerModel().pretrained("ner_clinical", "en", "clinical/models") \
            .setInputCols(["sentence", "token", "word_embeddings"]) \
            .setOutputCol("clinical_ner")

        clinical_ner_chunk = NerConverter() \
            .setInputCols("sentence", "token", "clinical_ner") \
            .setOutputCol("clinical_ner_chunk") \
            .setWhiteList(["PROBLEM"])

        # to get DRUG entities
        posology_ner = MedicalNerModel().pretrained("ner_posology", "en", "clinical/models") \
            .setInputCols(["sentence", "token", "word_embeddings"]) \
            .setOutputCol("posology_ner")

        posology_ner_chunk = NerConverter() \
            .setInputCols("sentence", "token", "posology_ner") \
            .setOutputCol("posology_ner_chunk") \
            .setWhiteList(["DRUG"])

        # merge the chunks into a single ner_chunk
        chunk_merger = ChunkMergeApproach() \
            .setInputCols("clinical_ner_chunk", "posology_ner_chunk") \
            .setOutputCol("final_ner_chunk") \
            .setMergeOverlapping(False)

        # convert chunks to doc to get sentence embeddings of them
        chunk2doc = Chunk2Doc().setInputCols("final_ner_chunk").setOutputCol("final_chunk_doc")

        sbiobert_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli", "en", "clinical/models") \
            .setInputCols(["final_chunk_doc"]) \
            .setOutputCol("sbert_embeddings") \
            .setCaseSensitive(False)

        # filter PROBLEM entity embeddings
        router_sentence_icd10 = Router() \
            .setInputCols("sbert_embeddings") \
            .setFilterFieldsElements(["PROBLEM"]) \
            .setOutputCol("problem_embeddings")

        # filter DRUG entity embeddings
        router_sentence_rxnorm = Router() \
            .setInputCols("sbert_embeddings") \
            .setFilterFieldsElements(["DRUG"]) \
            .setOutputCol("drug_embeddings")

        # use problem_embeddings only
        icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_slim_billable_hcc", "en",
                                                              "clinical/models") \
            .setInputCols(["clinical_ner_chunk", "problem_embeddings"]) \
            .setOutputCol("icd10cm_code") \
            .setDistanceFunction("EUCLIDEAN")

        # use drug_embeddings only
        rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented", "en",
                                                                 "clinical/models") \
            .setInputCols(["posology_ner_chunk", "drug_embeddings"]) \
            .setOutputCol("rxnorm_code") \
            .setDistanceFunction("EUCLIDEAN")

        pipeline = Pipeline(stages=[
            documentAssembler,
            sentenceDetector,
            tokenizer,
            word_embeddings,
            clinical_ner,
            clinical_ner_chunk,
            posology_ner,
            posology_ner_chunk,
            chunk_merger,
            chunk2doc,
            sbiobert_embeddings,
            router_sentence_icd10,
            router_sentence_rxnorm,
            icd_resolver,
            rxnorm_resolver
        ])

        clinical_note = """The patient is a 41-year-old Vietnamese female with a cough that started last week. 
        She has had right-sided chest pain radiating to her back with fever starting yesterday. 
        She has a history of pericarditis in May 2006 and developed cough with right-sided chest pain. 
        MEDICATIONS
        1. Coumadin 1 mg daily. Last INR was on Tuesday, August 14, 2007, and her INR was 2.3.
        2. Amiodarone 100 mg p.o. daily.
        """

        p = nlu.to_nlu_pipe(pipeline)
        df = p.predict(clinical_note)
        print(df)
        for c in df:
            print(df[c])

    def test_small_example(self):

        documentAssembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("ner_chunk")

        sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en', 'clinical/models') \
            .setInputCols(["ner_chunk"]) \
            .setOutputCol("sentence_embeddings") \
            .setCaseSensitive(False)

        rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented", "en",
                                                                 "clinical/models") \
            .setInputCols(["ner_chunk", "sentence_embeddings"]) \
            .setOutputCol("rxnorm_code") \
            .setDistanceFunction("EUCLIDEAN")

        rxnorm_pipelineModel = PipelineModel(
            stages=[
                documentAssembler,
                sbert_embedder,
                rxnorm_resolver])
        nlu.enable_verbose()
        p = nlu.to_nlu_pipe(rxnorm_pipelineModel)
        text = 'metformin 100 mg'
        df = p.predict(text)
        print(df)
        for c in df:
            print(df[c])

    def test_re_pipe(self):

        documenter = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentencer = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentences")

        tokenizer = sparknlp.annotators.Tokenizer() \
            .setInputCols(["sentences"]) \
            .setOutputCol("tokens")

        words_embedder = WordEmbeddingsModel() \
            .pretrained("embeddings_clinical", "en", "clinical/models") \
            .setInputCols(["sentences", "tokens"]) \
            .setOutputCol("embeddings")

        pos_tagger = PerceptronModel() \
            .pretrained("pos_clinical", "en", "clinical/models") \
            .setInputCols(["sentences", "tokens"]) \
            .setOutputCol("pos_tags")

        ner_tagger = MedicalNerModel() \
            .pretrained("ner_posology", "en", "clinical/models") \
            .setInputCols("sentences", "tokens", "embeddings") \
            .setOutputCol("ner_tags")

        ner_chunker = NerConverterInternal() \
            .setInputCols(["sentences", "tokens", "ner_tags"]) \
            .setOutputCol("ner_chunks")

        dependency_parser = DependencyParserModel() \
            .pretrained("dependency_conllu", "en") \
            .setInputCols(["sentences", "pos_tags", "tokens"]) \
            .setOutputCol("dependencies")

        reModel = RelationExtractionModel() \
            .pretrained("posology_re") \
            .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"]) \
            .setOutputCol("relations") \
            .setMaxSyntacticDistance(4)

        pipeline = Pipeline(stages=[
            documenter,
            sentencer,
            tokenizer,
            words_embedder,
            pos_tagger,
            ner_tagger,
            ner_chunker,
            dependency_parser,
            reModel
        ])
        text = """
        The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also 
        given 1 unit of Metformin daily.
        He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 
        12 units of insulin lispro with meals , and metformin 1000 mg two times a day.
        """

        ner_tagger = MedicalNerModel.pretrained("ner_ade_clinical", "en", "clinical/models") \
            .setInputCols("sentences", "tokens", "embeddings") \
            .setOutputCol("ner_tags")

        reModel = RelationExtractionModel() \
            .pretrained("re_ade_clinical", "en", 'clinical/models') \
            .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"]) \
            .setOutputCol("relations") \
            .setMaxSyntacticDistance(10) \
            .setRelationPairs(["drug-ade, ade-drug"]) \
            .setRelationPairsCaseSensitive(
            False)  # it will return any "ade-drug" or "ADE-DRUG" relationship. By default it’s set to False
        # True, then the pairs of entities in the dataset should match the pairs in setRelationPairs in their specific case (case sensitive).
        # False, meaning that the match of those relation names is case insensitive.
        ade_pipeline = Pipeline(stages=[
            documenter,
            sentencer,
            tokenizer,
            words_embedder,
            pos_tagger,
            ner_tagger,
            ner_chunker,
            dependency_parser,
            reModel
        ])
        ade_text = "I experienced fatigue, muscle cramps, anxiety, agression and sadness after taking Lipitor but no more adverse after passing Zocor."

        # nlu.to_nlu_pipe(ade_pipeline).viz(ade_text)
        nlu.to_pretty_df(ade_pipeline, ade_text, positions=True)

        # nlu.to_pretty_df(pipeline,text,positions=True)

    def test_ner_pipe(self):

        from sparknlp.pretrained import PretrainedPipeline
        pipeline = PretrainedPipeline('explain_clinical_doc_carp', 'en', 'clinical/models')
        text = """A 28-year-old female with a history of gestational diabetes mellitus, used to take metformin 1000 mg two times a day, presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting .
        She was seen by the endocrinology service and discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals.
        """
        df = nlu.to_pretty_df(pipeline.model, text, output_level='chunk')
        c = ['token',
             'pos',
             'entities_clinical_ner_chunks',
             'unlabeled_dependency',
             ]
        df[c].head(20)

    def test_deid_finisher_pipe(self):

        pipe = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")
        text = "Record date : 2093-01-13 , David Hale , M.D .  Name : Hendrickson , Ora MR 25 years-old . # 719435 Date : 01/13/93 . Signed by Oliveira Sander . Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street. Phone 302-786-5227."
        df = nlu.to_pretty_df(pipe, text, output_level='chunk')

    def test_ade_assert(self):
        ade_pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')
        text = """I have an allergic reaction to vancomycin. 
            My skin has be itchy, sore throat/burning/itchy, and numbness in tongue and gums. 
            I would not recommend this drug to anyone, especially since I have never had such an adverse reaction to any other medication."""
        # df = nlu.to_pretty_df(ade_pipeline,  text, output_level='chunk')
        df = nlu.viz(ade_pipeline, text)

    def test_redl_bug(self):
        text = 'She is diagnosed as cancer in 1991. Then she was admitted to Mayo Clinic in May 2000 and discharged in October 2001'
        events_ner_tagger = MedicalNerModel.pretrained("ner_events_clinical", "en", "clinical/models") \
            .setInputCols("sentences", "tokens", "embeddings") \
            .setOutputCol("ner_tags")

        events_re_ner_chunk_filter = RENerChunksFilter() \
            .setInputCols(["ner_chunks", "dependencies"]) \
            .setOutputCol("re_ner_chunks")

        events_re_Model = RelationExtractionDLModel() \
            .pretrained('redl_temporal_events_biobert', "en", "clinical/models") \
            .setPredictionThreshold(0.5) \
            .setInputCols(["re_ner_chunks", "sentences"]) \
            .setOutputCol("relations")
        documenter = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentencer = SentenceDetector() \
            .setInputCols(["document"]) \
            .setOutputCol("sentences")
        words_embedder = WordEmbeddingsModel() \
            .pretrained("embeddings_clinical", "en", "clinical/models") \
            .setInputCols(["sentences", "tokens"]) \
            .setOutputCol("embeddings")

        pos_tagger = PerceptronModel() \
            .pretrained("pos_clinical", "en", "clinical/models") \
            .setInputCols(["sentences", "tokens"]) \
            .setOutputCol("pos_tags")

        dependency_parser = DependencyParserModel() \
            .pretrained("dependency_conllu", "en") \
            .setInputCols(["sentences", "pos_tags", "tokens"]) \
            .setOutputCol("dependencies")
        tokenizer = Tokenizer() \
            .setInputCols(["sentences"]) \
            .setOutputCol("tokens")

        ner_chunker = NerConverterInternal() \
            .setInputCols(["sentences", "tokens", "ner_tags"]) \
            .setOutputCol("ner_chunks")
        pipeline = Pipeline(stages=[
            documenter,
            sentencer,
            tokenizer,
            words_embedder,
            pos_tagger,
            events_ner_tagger,
            ner_chunker,
            dependency_parser,
            events_re_ner_chunk_filter,
            events_re_Model
        ])

        nlu.to_pretty_df(pipeline, 'Hlel world')

    def test_multi_ner_viz(self):
        text = 'She is diagnosed as cancer in 1991. Then she was admitted to Mayo Clinic in May 2000 and discharged in October 2001'
        documentAssembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en",
                                                              "clinical/models") \
            .setInputCols(["document"]) \
            .setOutputCol("sentence")

        tokenizer = Tokenizer() \
            .setInputCols(["sentence"]) \
            .setOutputCol("token")

        word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models") \
            .setInputCols(["sentence", "token"]) \
            .setOutputCol("embeddings")

        jsl_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
            .setInputCols(["sentence", "token", "embeddings"]) \
            .setOutputCol("jsl_ner")

        jsl_ner_converter = NerConverter() \
            .setInputCols(["sentence", "token", "jsl_ner"]) \
            .setOutputCol("jsl_ner_chunk")

        jsl_ner_pipeline = Pipeline(stages=[
            documentAssembler,
            sentenceDetector,
            tokenizer,
            word_embeddings,
            jsl_ner,
            jsl_ner_converter])

        nlu.viz(jsl_ner_pipeline, text,
                ner_col = '',

                )
    def test_viz_bug(self):
        nlu.load('med_ner.jsl.wip.clinical resolve_chunk.rxnorm.in').viz("He took 2 pills of Aspirin daily")

if __name__ == "__main__":
    unittest.main()
