d = {
#ASSERT (so smul) AssertionDLModel
'en.':'assertion_dl', # clinical
'en.':'assertion_dl_biobert', # biobert_pubmed_base_cased
'en.':'assertion_dl_healthcare', # embeddings_healthcare100
'en.':'assertion_dl_large', # clinical


# PIPE?
# AssertionLogRegModel
'en.':'assertion_ml',

#BIOBERT embeddings
'en.embed.biobert.clinical':'biobert_clinical_base_cased',
'en.embed.biobert.discharge':'biobert_discharge_base_cased',
'en.embed.biobert.pmc':'biobert_pmc_base_cased',
'en.embed.biobert.pubmed':'biobert_pubmed_base_cased',
'en.embed.biobert.pubmed_large':'biobert_pubmed_large_cased',
'en.embed.biobert.pubmed_pmc':'biobert_pubmed_pmc_base_cased',


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


# Chunk resolve

'en.':'chunkresolve_athena_conditions_healthcare',
'en.':'chunkresolve_cpt_clinical',
'en.':'chunkresolve_cpt_icdoem',

# Chunk resolve icd
'en.':'chunkresolve_icd10cm_clinical',
'en.':'chunkresolve_icd10cm_diseases_clinical',
'en.':'chunkresolve_icd10cm_hcc_clinical',
'en.':'chunkresolve_icd10cm_hcc_healthcare',
'en.':'chunkresolve_icd10cm_icdoem',
'en.':'chunkresolve_icd10cm_icdoem_2ng',
'en.':'chunkresolve_icd10cm_injuries_clinical',
'en.':'chunkresolve_icd10cm_musculoskeletal_clinical',
'en.':'chunkresolve_icd10cm_neoplasms_clinical',
'en.':'chunkresolve_icd10cm_poison_ext_clinical',
'en.':'chunkresolve_icd10cm_puerile_clinical',
'en.':'chunkresolve_icd10cpt_icdoem_2ng',
'en.':'chunkresolve_icd10pcs_clinical',
'en.':'chunkresolve_icd10pcs_icdoem',
'en.':'chunkresolve_icd10pcs_icdoem_2ng',
'en.':'chunkresolve_icdo_clinical',
'en.':'chunkresolve_icdo_icdoem',

'en.':'chunkresolve_loinc_clinical',

# chunk resolve rxnorm
'en.':'chunkresolve_rxnorm_cd_clinical',
'en.':'chunkresolve_rxnorm_in_clinical',
'en.':'chunkresolve_rxnorm_in_healthcare',
'en.':'chunkresolve_rxnorm_sbd_clinical',
'en.':'chunkresolve_rxnorm_scd_clinical',
'en.':'chunkresolve_rxnorm_scdc_clinical',
'en.':'chunkresolve_rxnorm_scdc_healthcare',
'en.':'chunkresolve_rxnorm_xsmall_clinical',

#Chunk resolve snomed
'en.':'chunkresolve_snomed_findings_clinical',

# Classify icd
'en.':'classifier_icd10cm_hcc_clinical',
'en.':'classifier_icd10cm_hcc_healthcare',
# cassify ade
'en.':'classifierdl_ade_biobert',
'en.':'classifierdl_ade_clinicalbert',
'en.':'classifierdl_ade_conversational_biobert',
# Classify gender
'en.':'classifierdl_gender_biobert',
'en.':'classifierdl_gender_sbert',

# Classify pico
'en.':'classifierdl_pico_biobert',


#?
'en.':'clinical_analysis',
'en.':'clinical_deidentification',
'en.':'clinical_ner_assertion',


# context
'en.':'context_spell_med',

# deid
'en.':'deid_rules',
'en.':'deidentify_dl',
'en.':'deidentify_enriched_clinical',
'en.':'deidentify_large',
'en.':'deidentify_rb',
'en.':'deidentify_rb_no_regex',


# embeddings
'en.':'embeddings_biovec',
'en.':'embeddings_clinical',
'en.':'embeddings_healthcare',
'en.':'embeddings_healthcare_100d',
'en.':'embeddings_icd10_base',
'en.':'embeddings_icdoem',
'en.':'embeddings_icdoem_2ng',


# ensemble resolve
'en.':'ensembleresolve_icd10cm_clinical',
'en.':'ensembleresolve_rxnorm_small_clinical',
'en.':'ensembleresolve_snomed_small_clinical',

# entity resolve
'en.':'entity_resolver_icd10',

# Explain
'en.':'explain_clinical_doc_ade',
'en.':'explain_clinical_doc_carp',
'en.':'explain_clinical_doc_era',

# Jsl assert
'en.':'jsl_assertion_wip',
'en.':'jsl_assertion_wip_large',

# JSL embeds
'en.':'jsl_bert_pubmed_cased',
'en.':'jsl_ner_wip_clinical',

# jsl ner
'en.':'jsl_ner_wip_greedy_clinical',
'en.':'jsl_ner_wip_modifier_clinical',
'en.':'jsl_rd_ner_wip_greedy_clinical',

# ner ade
'en.':'ner_ade_biobert',
'en.':'ner_ade_clinical',
'en.':'ner_ade_clinicalbert',
'en.':'ner_ade_healthcare',

# ner anatomy
'en.':'ner_anatomy',
'en.':'ner_anatomy_biobert',
'en.':'ner_anatomy_coarse',
'en.':'ner_anatomy_coarse_biobert',

# ner aspect sentiment
'en.':'ner_aspect_based_sentiment',

# ner bacterial species
'en.':'ner_bacterial_species',

# ner bionlp
'en.':'ner_bionlp',
'en.':'ner_bionlp_biobert',
'en.':'ner_bionlp_noncontrib',

# ner cancer
'en.':'ner_cancer_genetics',

# ner cells
'en.':'ner_cellular',
'en.':'ner_cellular_biobert',

# ner chem
'en.':'ner_chemicals',
'en.':'ner_chemprot_biobert',
'en.':'ner_chemprot_clinical',

# ner clinica;
'en.':'ner_clinical',
'en.':'ner_clinical_biobert',
'en.':'ner_clinical_icdem',
'en.':'ner_clinical_large',
'en.':'ner_clinical_noncontrib',


# ner CRF (DEPARACTD)
# 'en.':'ner_crf',
# ner DEID
'en.':'ner_deid_augmented',
'en.':'ner_deid_biobert',
'en.':'ner_deid_enriched',
'en.':'ner_deid_enriched_biobert',
'en.':'ner_deid_large',
'en.':'ner_deid_sd',
'en.':'ner_deid_sd_large',
'en.':'ner_deid_synthetic',
'en.':'ner_deidentify_dl',


# ner diag
'es.':'ner_diag_proc',

# ner disease
'en.':'ner_diseases',
'en.':'ner_diseases_biobert',
'en.':'ner_diseases_large',


# ner drugs
'en.':'ner_drugs',
'en.':'ner_drugs_greedy',
'en.':'ner_drugs_large',

# ner events
'en.':'ner_events_biobert',
'en.':'ner_events_clinical',
'en.':'ner_events_healthcare',

# ner fin
'en.':'ner_financial_contract',


# ner hc
'en.':'ner_healthcare',

# ner phenotype
'en.':'ner_human_phenotype_gene_biobert',
'en.':'ner_human_phenotype_gene_clinical',
'en.':'ner_human_phenotype_go_biobert',
'en.':'ner_human_phenotype_go_clinical',

# ner jsl
'en.':'ner_jsl',
'en.':'ner_jsl_biobert',
'en.':'ner_jsl_enriched',
'en.':'ner_jsl_enriched_biobert',

# ner measurements
'en.':'ner_measurements_clinical',
# ner medmentions
'en.':'ner_medmentions_coarse',

# ner neoplasm
'es.':'ner_neoplasms',
# ner posology
'en.':'ner_posology',
'en.':'ner_posology_biobert',
'en.':'ner_posology_greedy',
'en.':'ner_posology_healthcare',
'en.':'ner_posology_large',
'en.':'ner_posology_large_biobert',
'en.':'ner_posology_small',

# ner radiology
'en.':'ner_radiology',
'en.':'ner_radiology_wip_clinical',
# ner risk factors
'en.':'ner_risk_factors',
'en.':'ner_risk_factors_biobert',

# ner CRF (DEPRACTATED)
# 'en.':'nercrf_deid',
# 'en.':'nercrf_tumour_demo',

# ner deid
'en.':'nerdl_deid',
# ner i2b2
'en.':'nerdl_i2b2',
'en.':'nerdl_tumour_demo',

# ? (pipes?)
'en.':'people_disambiguator',

# pos clinical
'en.':'pos_clinical',
# pos med
'en.':'pos_fast_med',

# pipe?
'en.':'ppl_posology_rxnorm',

# relation extractor body parts
'en.':'re_bodypart_directions',
'en.':'re_bodypart_problem',
'en.':'re_bodypart_proceduretest',
# relation extractor chem
'en.':'re_chemprot_clinical',

# relation clincial

'en.':'re_clinical',
'en.':'re_date_clinical',
# relation extractor  drug interaction
'en.':'re_drug_drug_interaction_clinical',
# relation extractor human phenotype
'en.':'re_human_phenotype_gene_clinical',

# relation extractor temporal events
'en.':'re_temporal_events_clinical',
'en.':'re_temporal_events_enriched_clinical',

# relation extractor pipe posology ?
'en.':'recognize_entities_posology',

# relation extractor bodypart
'en.':'redl_bodypart_direction_biobert',
'en.':'redl_bodypart_problem_biobert',
'en.':'redl_bodypart_procedure_test_biobert',
# relation extractor chem
'en.':'redl_chemprot_biobert',

# relation extractor clinical
'en.':'redl_clinical_biobert',


# relation extractor date clinical
'en.':'redl_date_clinical_biobert',

# relation extractor drug drug
'en.':'redl_drug_drug_interaction_biobert',

# relation extractor human phenotype
'en.':'redl_human_phenotype_gene_biobert',


# relation extractor temporal events
'en.':'redl_temporal_events_biobert',

# resolve cpt
'en.':'resolve_cpt_cl_em',
'en.':'resolve_cpt_icdoem',

# resolve icd10
'en.':'resolve_icd10',
'en.':'resolve_icd10cm_cl_em',
'en.':'resolve_icd10cm_icdem',
'en.':'resolve_icd10cm_icdoem',
'en.':'resolve_icd10pcs_cl_em',
'en.':'resolve_icdo_icdoem',

# resolve rxnorm
'en.':'resolve_rxnorm_clinical_l1',
'en.':'resolve_rxnorm_clinical_l2',
'en.':'resolve_rxnorm_healthcare_l1',
'en.':'resolve_rxnorm_healthcare_l2',
'en.':'resolve_rxnorm_l1_idx_icdoem_2ng',
'en.':'resolve_rxnorm_l1_ovrlrc_icdoem_2ng',
'en.':'resolve_rxnorm_l1_tfidf_icdoem_2ng',
'en.':'resolve_rxnorm_l2_icdoem_2ng',

# resolve snomed
'en.':'resolve_snomed_clinical_l1',
'en.':'resolve_snomed_clinical_l2',
'en.':'resolve_snomed_l1_idx_icdoem_2ng',
'en.':'resolve_snomed_l1_ovrlrc_icdoem_2ng',
'en.':'resolve_snomed_l1_tfidf_icdoem_2ng',
'en.':'resolve_snomed_l2_icdoem_2ng',

# sentence Entity resolvers

# resolve sentence mli
'en.':'sbiobert_base_cased_mli',

# resolve sentence cpt
'en.':'sbiobertresolve_cpt',
'en.':'sbiobertresolve_cpt_augmented',
'en.':'sbiobertresolve_cpt_procedures_augmented',

# resolve sentence hcc
'en.':'sbiobertresolve_hcc_augmented',

# resolve sentence icd
'en.':'sbiobertresolve_icd10cm',
'en.':'sbiobertresolve_icd10cm_augmented',
'en.':'sbiobertresolve_icd10cm_augmented_billable_hcc',
'en.':'sbiobertresolve_icd10pcs',
'en.':'sbiobertresolve_icdo',

# rewsolve sentence rx
'en.':'sbiobertresolve_rxcui',
'en.':'sbiobertresolve_rxnorm',
# resolve sentence snomed s
'en.':'sbiobertresolve_snomed_auxConcepts',
'en.':'sbiobertresolve_snomed_auxConcepts_int',
'en.':'sbiobertresolve_snomed_findings',
'en.':'sbiobertresolve_snomed_findings_int',

# resolve sentence ?????
'en.':'sbluebert_base_uncased_mli',



# ????
'en.':'sent_biobert_base_uncased_mednli',
'en.':'sent_bluebert_base_uncased_mednli',

# sentence detector dl healthcare
'en.':'sentence_detector_dl_healthcare',

# spel check clinical
'en.':'spellcheck_clinical',

# spellcheck dl
'en.':'spellcheck_dl',

# t5?
'en.':'t5_base_mediqa_mnli',

# text 2 sql deprecated
# 'en.':'text2sql_glove',


# text match cpt
'en.':'textmatch_cpt_token',
'en.':'textmatch_cpt_token_n2c1',

# text match
'en.':'textmatch_icd10cm',
'en.':'textmatch_icd10cm_uncased',
'en.':'textmatch_icdo',
'en.':'textmatch_icdo_ner',
'en.':'textmatch_icdo_ner_n2c4',

# classify icd10
'en.':'useclassifier_icd10cm_hcc',
}

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
