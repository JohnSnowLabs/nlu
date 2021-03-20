d = {

# DeIdentificationModel  |||| EXPORTED
'en.de_identify':'deidentify_dl',
'en.de_identify.rules':'deid_rules',
'en.de_identify.clinical':'deidentify_enriched_clinical',
'en.de_identify.large':'deidentify_large',
'en.de_identify.rb':'deidentify_rb',
'en.de_identify.rb_no_regex':'deidentify_rb_no_regex',

# ner DEID  |||| EXPORTED
'en.ner.deid.augmented':'ner_deid_augmented',
'en.ner.deid.biobert':'ner_deid_biobert',
'en.ner.deid.enriched':'ner_deid_enriched',
'en.ner.deid.enriched_biobert':'ner_deid_enriched_biobert',
'en.ner.deid.large':'ner_deid_large',
'en.ner.deid.sd':'ner_deid_sd',
'en.ner.deid.sd_large':'ner_deid_sd_large',
'en.ner.deid.synthetic':'ner_deid_synthetic',
'en.ner.deid.dl':'ner_deidentify_dl',
'en.ner.deid.':'nerdl_deid',

# WordEmbeddings ||||||||||||||EXPORTET
'en.':'embeddings_biovec',
'en.':'embeddings_clinical',
'en.':'embeddings_healthcare',
'en.':'embeddings_healthcare_100d',
'en.':'embeddings_icd10_base',
'en.':'embeddings_icdoem',
'en.':'embeddings_icdoem_2ng',


# pos clinical || EXPORTED
'en.pos.clinical':'pos_clinical',
'en.pos.fast':'pos_fast_med',
ss

# RelationExtraction ||| EXPORTED BUT DEPRECRATED?!??!
'en.extract_relation':'re_bodypart_directions',
'en.extract_relation.bodypart.direction':'re_bodypart_directions',
'en.extract_relation.bodypart.problem':'re_bodypart_problem',
'en.extract_relation.bodypart.proceduretest':'re_bodypart_proceduretest',
'en.extract_relation.chemprot':'re_chemprot_clinical',
'en.extract_relation.clinical':'re_clinical',
'en.extract_relation.date_clinical':'re_date_clinical',
'en.extract_relation.drug_drug_interaction':'re_drug_drug_interaction_clinical',
'en.extract_relation.human_phenotype_gene':'re_human_phenotype_gene_clinical',
'en.extract_relation.temporal_events':'re_temporal_events_clinical',
'en.extract_relation.temporal_events.enriched':'re_temporal_events_enriched_clinical',

# RelationExtractionDL |||| EXOIRTED
'en.extract_relation':'redl_bodypart_direction_biobert',
'en.extract_relation.bodypart.direction':'redl_bodypart_direction_biobert',
'en.extract_relation.bodypart.problem':'redl_bodypart_problem_biobert',
'en.extract_relation.bodypart.procedure':'redl_bodypart_procedure_test_biobert',
'en.extract_relation.chemprot':'redl_chemprot_biobert',
'en.extract_relation.clinical':'redl_clinical_biobert',
'en.extract_relation.date':'redl_date_clinical_biobert',
'en.extract_relation.drug_drug_interaction':'redl_drug_drug_interaction_biobert',
'en.extract_relation.humen_phenotype_gene':'redl_human_phenotype_gene_biobert',
'en.extract_relation.temporal_events':'redl_temporal_events_biobert',


#NERDL Models  ||||  EXPORTED!!!
'en.ner.ade.biobert':'ner_ade_biobert',
'en.ner.ade.clinical':'ner_ade_clinical',
'en.ner.ade.clinical_bert':'ner_ade_clinicalbert',
'en.ner.ade.ade_healthcare':'ner_ade_healthcare',
'en.ner.anatomy':'ner_anatomy',
'en.ner.anatomy.biobert':'ner_anatomy_biobert',
'en.ner.anatomy.coarse':'ner_anatomy_coarse',
'en.ner.anatomy.coarse_biobert':'ner_anatomy_coarse_biobert',
'en.ner.aspect_sentiment':'ner_aspect_based_sentiment',
'en.ner.bacterial_species':'ner_bacterial_species',
'en.ner.bionlp':'ner_bionlp',
'en.ner.bionlp.biobert':'ner_bionlp_biobert',
'en.ner.bionlp.biobert.noncontrib':'ner_bionlp_noncontrib', # todo whats noncontrib?
'en.ner.cancer':'ner_cancer_genetics',
'en.ner.cellular':'ner_cellular',
'en.ner.cellular.biobert':'ner_cellular_biobert',
'en.ner.chemicals':'ner_chemicals',
'en.ner.chemprot':'ner_chemprot_biobert',
'en.ner.chemprot.clinical':'ner_chemprot_clinical',
'en.ner.clinical':'ner_clinical',
'en.ner.clinical.biobert':'ner_clinical_biobert',
'en.ner.clinical.icdem':'ner_clinical_icdem',
'en.ner.clinical.large':'ner_clinical_large',
'en.ner.clinical.noncontrib':'ner_clinical_noncontrib',

### NERDL |||||| EXPORTED (PART2)
'en.ner.diseases.':'ner_diseases',
'en.ner.diseases.biobert':'ner_diseases_biobert',
'en.ner.diseases.large':'ner_diseases_large',
'en.ner.drugs':'ner_drugs',
'en.ner.drugsgreedy':'ner_drugs_greedy',
'en.ner.drugs.large':'ner_drugs_large',
'en.ner.events_biobert':'ner_events_biobert',
'en.ner.events_clinical':'ner_events_clinical',
'en.ner.events_healthcre':'ner_events_healthcare',
'en.ner.financial_contract':'ner_financial_contract',
'en.ner.healthcare':'ner_healthcare',
'en.ner.human_phenotype.gene_biobert':'ner_human_phenotype_gene_biobert',
'en.ner.human_phenotype.gene_clinical':'ner_human_phenotype_gene_clinical',
'en.ner.human_phenotype.go_biobert':'ner_human_phenotype_go_biobert',
'en.ner.human_phenotype.go_clinical':'ner_human_phenotype_go_clinical',
'en.ner.jsl':'ner_jsl',
'en.ner.jsl.biobert':'ner_jsl_biobert',
'en.ner.jsl.enriched':'ner_jsl_enriched',
'en.ner.jsl.enriched_biobert':'ner_jsl_enriched_biobert',
'en.ner.measurements':'ner_measurements_clinical',
'en.ner.medmentions':'ner_medmentions_coarse',
'en.ner.posology':'ner_posology',
'en.ner.posology.biobert':'ner_posology_biobert',
'en.ner.posology.greedy':'ner_posology_greedy',
'en.ner.posology.healthcare':'ner_posology_healthcare',
'en.ner.posology.large':'ner_posology_large',
'en.ner.posology.large_biobert':'ner_posology_large_biobert',
'en.ner.posology.small':'ner_posology_small',
'en.ner.radiology':'ner_radiology',
'en.ner.radiology.wip_clinical':'ner_radiology_wip_clinical',
'en.ner.risk_factors':'ner_risk_factors',
'en.ner.risk_factors.biobert':'ner_risk_factors_biobert',
'en.ner.i2b2':'nerdl_i2b2',
'en.ner.tumour':'nerdl_tumour_demo',
'en.ner.jsl.wip.clinical':'jsl_ner_wip_clinical',
'en.ner.jsl.wip.clinical.greedy':'jsl_ner_wip_greedy_clinical',
'en.ner.jsl.wip.clinical.modifier':'jsl_ner_wip_modifier_clinical',
'en.ner.jsl.wip.clinical.rd':'jsl_rd_ner_wip_greedy_clinical',




# SentenceBertEmbedding |||||||||||||| EXPORTERD
'en.':'sent_biobert_base_uncased_mednli',
'en.':'sent_bluebert_base_uncased_mednli',
'en.resolve_sentence.mli.bluebuert':'sbluebert_base_uncased_mli',


#ASSERT (so smul) AssertionDLModel ||| EXPORTED
'en.assert':'assertion_dl', # clinicalz
'en.assert.biobert':'assertion_dl_biobert', # biobert_pubmed_base_cased
'en.assert.healthcare':'assertion_dl_healthcare', # embeddings_healthcare100
'en.assert.large':'assertion_dl_large', # clinical



#BIOBERT embeddings  |||||||||||| EXPORTED
'en.embed.biobert.clinical':'biobert_clinical_base_cased', # Todo crash NO POOLING LAYER
'en.embed.biobert.discharge':'biobert_discharge_base_cased',
'en.embed.biobert.pmc':'biobert_pmc_base_cased',
'en.embed.biobert.pubmed':'biobert_pubmed_base_cased', # Todo crash NO POOLING LAYER
'en.embed.biobert.pubmed_large':'biobert_pubmed_large_cased',
'en.embed.biobert.pubmed_pmc':'biobert_pubmed_pmc_base_cased',


# sentence Entity resolvers ||||||||EX{PRTED
'en.':'sbiobert_base_cased_mli',
'en.':'sbiobertresolve_cpt',
'en.':'sbiobertresolve_cpt_augmented',
'en.':'sbiobertresolve_cpt_procedures_augmented',
'en.':'sbiobertresolve_hcc_augmented',
'en.':'sbiobertresolve_icd10cm',
'en.':'sbiobertresolve_icd10cm_augmented',
'en.':'sbiobertresolve_icd10cm_augmented_billable_hcc',
'en.':'sbiobertresolve_icd10pcs',
'en.':'sbiobertresolve_icdo',
'en.':'sbiobertresolve_rxcui',
'en.':'sbiobertresolve_rxnorm',
'en.':'sbiobertresolve_snomed_auxConcepts',
'en.':'sbiobertresolve_snomed_auxConcepts_int',
'en.':'sbiobertresolve_snomed_findings',
'en.':'sbiobertresolve_snomed_findings_int',
'en.':'sbluebert_base_uncased_mli',





# ClassifierDL Models       +++++++++}|||||||||||||||||||||||||||||||||||||| EXPORTED `
'en.classify.icd10.clinical':'classifier_icd10cm_hcc_clinical', # WHCIH CLASS? # TODO NOT LAODING
'en.classify.icd10.healthcare':'classifier_icd10cm_hcc_healthcare', # TODO NOT LOADING CORRECt
'en.classify.ade.biobert':'classifierdl_ade_biobert',
'en.classify.ade.clinical':'classifierdl_ade_clinicalbert',
'en.classify.ade.conversational':'classifierdl_ade_conversational_biobert',
'en.classify.gender.biobert':'classifierdl_gender_biobert',
'en.classify.gender.sbert':'classifierdl_gender_sbert', # ok!
'en.classify.pico':'classifierdl_pico_biobert',
'en.classify.icd10.use':'useclassifier_icd10cm_hcc',

    # Chunk resolve ||||||||| EXPORTED
    'en.resolve_chunk.athena_conditions':'chunkresolve_athena_conditions_healthcare',
    'en.resolve_chunk.cpt_clinical':'chunkresolve_cpt_clinical',
    'en.resolve_chunk.cpt_icdoem':'chunkresolve_cpt_icdoem',
    'en.resolve_chunk.icd10cm.clinical':'chunkresolve_icd10cm_clinical',
    'en.resolve_chunk.icd10cm.diseases_clinical':'chunkresolve_icd10cm_diseases_clinical',
    'en.resolve_chunk.icd10cm.hcc_clinical':'chunkresolve_icd10cm_hcc_clinical',
    'en.resolve_chunk.icd10cm.hcc_healthcare':'chunkresolve_icd10cm_hcc_healthcare',
    'en.resolve_chunk.icd10cm.icdoem,':'chunkresolve_icd10cm_icdoem',
    'en.resolve_chunk.icd10cm.icdoem_2ng':'chunkresolve_icd10cm_icdoem_2ng',
    'en.resolve_chunk.icd10cm.injuries':'chunkresolve_icd10cm_injuries_clinical',
    'en.resolve_chunk.icd10cm.musculoskeletal':'chunkresolve_icd10cm_musculoskeletal_clinical',
    'en.resolve_chunk.icd10cm.neoplasms':'chunkresolve_icd10cm_neoplasms_clinical',
    'en.resolve_chunk.icd10cm.poison':'chunkresolve_icd10cm_poison_ext_clinical',
    'en.resolve_chunk.icd10cm.puerile':'chunkresolve_icd10cm_puerile_clinical',
    'en.resolve_chunk.icd10cpt.icdoem':'chunkresolve_icd10cpt_icdoem_2ng',
    'en.resolve_chunk.icd10pcs.clinical':'chunkresolve_icd10pcs_clinical',
    'en.resolve_chunk.icd10pcs.icdoem':'chunkresolve_icd10pcs_icdoem',
    'en.resolve_chunk.icd10pcs.icdoem_2ng':'chunkresolve_icd10pcs_icdoem_2ng',
    'en.resolve_chunk.icdo.clinical':'chunkresolve_icdo_clinical',
    'en.resolve_chunk.icdo.icdoem':'chunkresolve_icdo_icdoem',
    'en.resolve_chunk.loinc':'chunkresolve_loinc_clinical',
    'en.resolve_chunk.rxnorm.cd':'chunkresolve_rxnorm_cd_clinical',
    'en.resolve_chunk.rxnorm.in':'chunkresolve_rxnorm_in_clinical',
    'en.resolve_chunk.rxnorm.in_healthcare':'chunkresolve_rxnorm_in_healthcare',
    'en.resolve_chunk.rxnorm.sbd':'chunkresolve_rxnorm_sbd_clinical',
    'en.resolve_chunk.rxnorm.scd':'chunkresolve_rxnorm_scd_clinical',
    'en.resolve_chunk.rxnorm.scdc':'chunkresolve_rxnorm_scdc_clinical',
    'en.resolve_chunk.rxnorm.scdc_healthcare':'chunkresolve_rxnorm_scdc_healthcare',
    'en.resolve_chunk.rxnorm.xsmall.clinical':'chunkresolve_rxnorm_xsmall_clinical',
    'en.resolve_chunk.snomed.findings':'chunkresolve_snomed_findings_clinical',


#===========================+EXPORT END ==================================

# AssertionLogRegModel
'en.':'assertion_ml',


# SentenceEntityResolverModel biobert
# These all hava analogus models with s prefixed to them. Deprecated?
# Sentence Entity Resolvers Deprecated models
# 'en.':'biobertresolve_cpt', #  TODO Crash?
# 'en.':'biobertresolve_icd10cm', # TODO crash
# 'en.':'biobertresolve_icd10cm_hcc',  # TODO crash
# 'en.':'biobertresolve_icd10pcs', # todo crash
# 'en.':'biobertresolve_icdo', # todo crash
# 'en.':'biobertresolve_loinc', # todo crash
# 'en.':'biobertresolve_rxnorm_bdcd',
# 'en.':'biobertresolve_snomed_findings',
# 'en.':'biobertresolve_snomed_findings_int',


















#?
'en.':'clinical_analysis',
'en.':'clinical_deidentification',
'en.':'clinical_ner_assertion',


# context
'en.':'context_spell_med',


# resolve cpt (SENTENCE OR CHUNK??) or pipe?
'en.':'resolve_cpt_cl_em',
'en.':'resolve_cpt_icdoem',
'en.':'resolve_icd10',
'en.':'resolve_icd10cm_cl_em',
'en.':'resolve_icd10cm_icdem',
'en.':'resolve_icd10cm_icdoem',
'en.':'resolve_icd10pcs_cl_em',
'en.':'resolve_icdo_icdoem',
'en.':'resolve_rxnorm_clinical_l1',
'en.':'resolve_rxnorm_clinical_l2',
'en.':'resolve_rxnorm_healthcare_l1',
'en.':'resolve_rxnorm_healthcare_l2',
'en.':'resolve_rxnorm_l1_idx_icdoem_2ng',
'en.':'resolve_rxnorm_l1_ovrlrc_icdoem_2ng',
'en.':'resolve_rxnorm_l1_tfidf_icdoem_2ng',
'en.':'resolve_rxnorm_l2_icdoem_2ng',

# resolve snomed (SENTENCE OR CHUNK??) or pipe?
'en.resolve_?.snomed':'resolve_snomed_clinical_l1', # default snomed resolver
'en.resolve_?.snomed.clinical.l1':'resolve_snomed_clinical_l1',
'en.resolve_?.snomed.clinical.l2':'resolve_snomed_clinical_l2',
'en.resolve_?.snomed.l1.idx':'resolve_snomed_l1_idx_icdoem_2ng',
'en.resolve_?.snomed.l1.ovrlrc':'resolve_snomed_l1_ovrlrc_icdoem_2ng',
'en.resolve_?.snomed.l1.tfidf':'resolve_snomed_l1_tfidf_icdoem_2ng',
'en.resolve_?.snomed.l2.icdoem':'resolve_snomed_l2_icdoem_2ng',



# ensemble resolve # TODO class?
'en.':'ensembleresolve_icd10cm_clinical',
'en.':'ensembleresolve_rxnorm_small_clinical',
'en.':'ensembleresolve_snomed_small_clinical',
# entity resolve
'en.':'entity_resolver_icd10',

# Explain Pipes
'en.':'explain_clinical_doc_ade',
'en.':'explain_clinical_doc_carp',
'en.':'explain_clinical_doc_era',

# relation extractor pipe posology
'en.':'recognize_entities_posology',


# ? (pipes?)
'en.':'people_disambiguator',
# pipe?
'en.':'ppl_posology_rxnorm',


# sentence detector dl healthcare
'en.':'sentence_detector_dl_healthcare',
# spel check clinical pipe?
'en.':'spellcheck_clinical',
# spellcheck dl
'en.':'spellcheck_dl',
# t5?
'en.':'t5_base_mediqa_mnli',
# text matchcher old deprecated/ client project
# 'en.':'textmatch_cpt_token',
# 'en.':'textmatch_cpt_token_n2c1',
# 'en.':'textmatch_icd10cm',
# 'en.':'textmatch_icd10cm_uncased',
# 'en.':'textmatch_icdo',
# 'en.':'textmatch_icdo_ner',
# 'en.':'textmatch_icdo_ner_n2c4',
#


}



########################## NON EN TODO ##########################
# DE
w2v_cc_300d,de
chunkresolve_ICD10GM,de
chunkresolve_ICD10GM_2021,de
ner_legal,de
ner_healthcare,de
ner_healthcare_slim,de
ner_traffic,de



#ES
embeddings_scielo_150d,es
embeddings_scielo_300d,es
embeddings_scielo_50d,es
embeddings_scielowiki_150d,es
embeddings_scielowiki_300d,es
embeddings_scielowiki_50d,es
embeddings_sciwiki_150d,es
embeddings_sciwiki_300d,es
embeddings_sciwiki_50d,es
'es.':'ner_neoplasms', # todo sure es?
'es.':'ner_diag_proc',

# ner CRF (DEPARACTD)
# 'en.':'ner_crf',

# ner CRF (DEPRACTATED)
# 'en.':'nercrf_deid',z
# TODO JSL STUFF IS ARE UNDOCUMENTED CLIENT PROEJCTS! exclude for now!
# Jsl assert
'en.':'jsl_assertion_wip',
'en.':'jsl_assertion_wip_large',
# JSL embeds
'en.':'jsl_bert_pubmed_cased',





