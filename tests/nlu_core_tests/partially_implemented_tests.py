import unittest
from sparknlp.base import *
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
import nlu
import tests.secrets as sct

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
            .setInputCols("sentence","token","clinical_ner") \
            .setOutputCol("clinical_ner_chunk") \
            .setWhiteList(["PROBLEM"])

        # to get DRUG entities
        posology_ner = MedicalNerModel().pretrained("ner_posology", "en", "clinical/models") \
            .setInputCols(["sentence", "token", "word_embeddings"]) \
            .setOutputCol("posology_ner")

        posology_ner_chunk = NerConverter() \
            .setInputCols("sentence","token","posology_ner") \
            .setOutputCol("posology_ner_chunk") \
            .setWhiteList(["DRUG"])

        # merge the chunks into a single ner_chunk
        chunk_merger = ChunkMergeApproach() \
            .setInputCols("clinical_ner_chunk","posology_ner_chunk") \
            .setOutputCol("final_ner_chunk") \
            .setMergeOverlapping(False)


        # convert chunks to doc to get sentence embeddings of them
        chunk2doc = Chunk2Doc().setInputCols("final_ner_chunk").setOutputCol("final_chunk_doc")


        sbiobert_embeddings = BertSentenceEmbeddings.pretrained("sbiobert_base_cased_mli","en","clinical/models") \
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
        icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_slim_billable_hcc","en", "clinical/models") \
            .setInputCols(["clinical_ner_chunk", "problem_embeddings"]) \
            .setOutputCol("icd10cm_code") \
            .setDistanceFunction("EUCLIDEAN")


        # use drug_embeddings only
        rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models") \
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
        print (df)
        for c in df:
            print(df[c])

    def test_small_example(self):

        documentAssembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("ner_chunk")

        sbert_embedder = BertSentenceEmbeddings.pretrained('sbiobert_base_cased_mli', 'en','clinical/models') \
            .setInputCols(["ner_chunk"]) \
            .setOutputCol("sentence_embeddings") \
            .setCaseSensitive(False)

        rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_augmented","en", "clinical/models") \
            .setInputCols(["ner_chunk", "sentence_embeddings"]) \
            .setOutputCol("rxnorm_code") \
            .setDistanceFunction("EUCLIDEAN")

        rxnorm_pipelineModel = PipelineModel(
            stages = [
                documentAssembler,
                sbert_embedder,
                rxnorm_resolver])
        nlu.enable_verbose()
        p = nlu.to_nlu_pipe(rxnorm_pipelineModel)
        text = 'metformin 100 mg'
        df = p.predict(text)
        print (df)
        for c in df:
            print(df[c])

if __name__ == "__main__":
    unittest.main()
