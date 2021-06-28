---
layout: docs
header: true
layout: article
title: NLU release notes
permalink: /docs/en/release_notes
key: docs-release-notes
modify_date: "2020-06-12"
---

<div class="main-docs" markdown="1">

# NLU 3.1 Release Notes

# 2600+ New Models for 200+ Languages and 10+ Dimension Reduction Algorithms for Streamlit Word-Embedding visualizations in 3-D

We are extremely excited to announce the release of NLU 3.1 !
This is our biggest release so far and it comes with over `2600+ new models in 200+` languages, including `DistilBERT`, `RoBERTa`, and `XLM-RoBERTa` and Huggingface based Embeddings from the [incredible Spark-NLP 3.1.0 release](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.1.0),
new `Streamlit Visualizations` for visualizing Word Embeddings in `3-D`, `2-D`, and `1-D`,
New Healthcare pipelines for `healthcare code mappings`
and finally `confidence extraction` for open source NER models.
Additionally, the NLU Namespace has been renamed to the NLU Spellbook, to reflect the magicalness of each 1-liners represented by them!


## Streamlit Word Embedding visualization via Manifold and Matrix Decomposition algorithms

### <kbd>function</kbd> `pipe.viz_streamlit_word_embed_manifold`

Visualize Word Embeddings in `1-D`, `2-D`, or `3-D` by `Reducing Dimensionality` via 11 Supported methods from  [Manifold Algorithms](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold)
and [Matrix Decomposition Algorithms](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition).
Additionally, you can color the lower dimensional points with a label that has been previously assigned to the text by specifying a list of nlu references in the `additional_classifiers_for_coloring` parameter.

- Reduces Dimensionality of high dimensional Word Embeddings to `1-D`, `2-D`, or `3-D` and plot the resulting data in an interactive `Plotly` plot
- Applicable with [any of the 100+ Word Embedding models](https://nlp.johnsnowlabs.com/models?task=Embeddings)
- Color points by classifying with any of the 100+ [Parts of Speech Classifiers](https://nlp.johnsnowlabs.com/models?task=Part+of+Speech+Tagging) or [Document Classifiers](https://nlp.johnsnowlabs.com/models?task=Text+Classification)
- Gemerates `NUM-DIMENSIONS` * `NUM-EMBEDDINGS` * `NUM-DIMENSION-REDUCTION-ALGOS` plots



```python
nlu.load('bert',verbose=True).viz_streamlit_word_embed_manifold(default_texts=THE_MATRIX_ARCHITECT_SCRIPT.split('\n'),default_algos_to_apply=['TSNE'],MAX_DISPLAY_NUM=5)
```

<img  src="https://github.com/JohnSnowLabs/nlu/blob/3.1.0rc1/docs/assets/streamlit_docs_assets/gif/word_embed_dimension_reduction/manifold_intro.gif?raw=true">




### <kbd>function parameters</kbd> `pipe.viz_streamlit_word_embed_manifold`
| Argument    | Type        |                                                            Default         |Description |
|--------------------------- | ---------- |-----------------------------------------------------------| ------------------------------------------------------- |
`default_texts`|          `List[str]`  | ("Donald Trump likes to party!", "Angela Merkel likes to party!", 'Peter HATES TO PARTTY!!!! :(') | List of strings to apply classifiers, embeddings, and manifolds to.
| `text`                    | `Optional[str]`   |     `'Billy likes to swim'`                 | Text to predict classes for. | 
`sub_title`|         ` Optional[str]` | "Apply any of the 11 `Manifold` or `Matrix Decomposition` algorithms to reduce the dimensionality of `Word Embeddings` to `1-D`, `2-D` and `3-D` " | Sub title of the Streamlit app
`default_algos_to_apply`|           `List[str]` | `["TSNE", "PCA"]` | A list Manifold and Matrix Decomposition Algorithms to apply. Can be either `'TSNE'`,`'ISOMAP'`,`'LLE'`,`'Spectral Embedding'`, `'MDS'`,`'PCA'`,`'SVD aka LSA'`,`'DictionaryLearning'`,`'FactorAnalysis'`,`'FastICA'` or `'KernelPCA'`,
`target_dimensions`|          `List[int]` | `(1,2,3)` | Defines the target dimension embeddings will be reduced to
`show_algo_select`|          `bool` | `True`  | Show selector for Manifold and Matrix Decomposition Algorithms
`show_embed_select`|          `bool` | `True` | Show selector for Embedding Selection
`show_color_select`|          `bool` | `True` | Show selector for coloring plots
`MAX_DISPLAY_NUM`|         `int`|`100` | Cap maximum number of Tokens displayed
|`display_embed_information`              | `bool`              |  `True`                         | Show additional embedding information like `dimension`, `nlu_reference`, `spark_nlp_reference`, `sotrage_reference`, `modelhub link` and more.
| `set_wide_layout_CSS`      |  `bool`                                                             |  `True`                                                                                   | Whether to inject custom CSS or not.
|`num_cols`                               | `int`               |  `2`                            |  How many columns should for the layout in streamlit when rendering the similarity matrixes.
|     `key`                               |  `str`              | `"NLU_streamlit"`               | Key for the Streamlit elements drawn
`additional_classifiers_for_coloring`|         `List[str]`|`['pos', 'sentiment.imdb']` | List of additional NLU references to load for generting hue colors
| `show_model_select`        |  `bool`                                          | `True`                                                                                 | Show a model selection dropdowns that makes any of the 1000+ models avaiable in 1 click
| `model_select_position`    |  `str`                                                             |   `'side'`            | [Whether to output the positions of predictions or not, see `pipe.predict(positions=true`) for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-positions-parameter)  |
| `show_logo`             |  `bool`                                            | `True`                                                                                   | Show logo
| `display_infos`         |  `bool`                                            | `False`                                                                                  | Display additonal information about ISO codes and the NLU namespace structure.
| `n_jobs`|          `Optional[int]` | `3`|   `False` | How many cores to use for paralellzing when using Sklearn Dimension Reduction algorithms.  

### Larger Example showcasing more dimension reduction techniques on a larger corpus :

<img  src="https://github.com/JohnSnowLabs/nlu/blob/3.1.0rc1/docs/assets/streamlit_docs_assets/gif/word_embed_dimension_reduction/big_example_word_embedding_dimension_reduction.gif?raw=true">


### [Supported Manifold Algorithms ](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold)
- [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)
- [ISOMAP](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html#sklearn.manifold.Isomap)
- [LLE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html#sklearn.manifold.LocallyLinearEmbedding)
- [Spectral Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html#sklearn.manifold.SpectralEmbedding)
- [MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html#sklearn.manifold.MDS)

### [Supported Matrix Decomposition Algorithms ](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition)
- [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)
- [Truncated SVD aka LSA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD)
- [DictionaryLearning](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.DictionaryLearning.html#sklearn.decomposition.DictionaryLearning)
- [FactorAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html#sklearn.decomposition.FactorAnalysis)
- [FastICA](https://scikit-learn.org/stable/modules/generated/fastica-function.html#sklearn.decomposition.fastica)
- [KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA)




## New Healthcare Pipelines Pipelines
Five new healthcare code mapping pipelines:
- `nlu.load(en.resolve.icd10cm.umls)`: This pretrained pipeline maps ICD10CM codes to UMLS codes without using any text data. You’ll just feed white space-delimited ICD10CM codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

`{'icd10cm': ['M89.50', 'R82.2', 'R09.01'],'umls': ['C4721411', 'C0159076', 'C0004044']}`

- `nlu.load(en.resolve.mesh.umls)`: This pretrained pipeline maps MeSH codes to UMLS codes without using any text data. You’ll just feed white space-delimited MeSH codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

`{'mesh': ['C028491', 'D019326', 'C579867'],'umls': ['C0970275', 'C0886627', 'C3696376']}`

- `nlu.load(en.resolve.rxnorm.umls)`: This pretrained pipeline maps RxNorm codes to UMLS codes without using any text data. You’ll just feed white space-delimited RxNorm codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.

`{'rxnorm': ['1161611', '315677', '343663'],'umls': ['C3215948', 'C0984912', 'C1146501']}`

- `nlu.load(en.resolve.rxnorm.mesh)`: This pretrained pipeline maps RxNorm codes to MeSH codes without using any text data. You’ll just feed white space-delimited RxNorm codes and it will return the corresponding MeSH codes as a list. If there is no mapping, the original code is returned with no mapping.

`{'rxnorm': ['1191', '6809', '47613'],'mesh': ['D001241', 'D008687', 'D019355']}`

- `nlu.load(en.resolve.snomed.umls)`: This pretrained pipeline maps SNOMED codes to UMLS codes without using any text data. You’ll just feed white space-delimited SNOMED codes and it will return the corresponding UMLS codes as a list. If there is no mapping, the original code is returned with no mapping.
  `{'snomed': ['733187009', '449433008', '51264003'],'umls': ['C4546029', 'C3164619', 'C0271267']}`

## New Healthcare Pipelines


|NLU Reference| Spark NLP Reference  | 
|---------------|---------------------|
|[en.resolve.icd10cm.umls]((https://nlp.johnsnowlabs.com/2021/05/04/icd10cm_umls_mapping_en.html)) | [icd10cm_umls_mapping](https://nlp.johnsnowlabs.com/2021/05/04/icd10cm_umls_mapping_en.html)  |
|[en.resolve.mesh.umls   ]((https://nlp.johnsnowlabs.com/2021/05/04/mesh_umls_mapping_en.html)) | [mesh_umls_mapping](https://nlp.johnsnowlabs.com/2021/05/04/mesh_umls_mapping_en.html)  |
|[en.resolve.rxnorm.umls ]((https://nlp.johnsnowlabs.com/2021/05/04/rxnorm_umls_mapping_en.html)) | [rxnorm_umls_mapping](https://nlp.johnsnowlabs.com/2021/05/04/rxnorm_umls_mapping_en.html)  |
|[en.resolve.rxnorm.mesh ]((https://nlp.johnsnowlabs.com/2021/05/04/rxnorm_mesh_mapping_en.html)) | [rxnorm_mesh_mapping](https://nlp.johnsnowlabs.com/2021/05/04/rxnorm_mesh_mapping_en.html)  |
|[en.resolve.snomed.umls ]((https://nlp.johnsnowlabs.com/2021/05/04/snomed_umls_mapping_en.html)) | [snomed_umls_mapping](https://nlp.johnsnowlabs.com/2021/05/04/snomed_umls_mapping_en.html)  |
|[en.explain_doc.carp    ]((https://nlp.johnsnowlabs.com/2021/04/01/explain_clinical_doc_carp_en.html)) | [explain_clinical_doc_carp](https://nlp.johnsnowlabs.com/2021/04/01/explain_clinical_doc_carp_en.html)  |
|[en.explain_doc.era     ]((https://nlp.johnsnowlabs.com/2021/04/01/explain_clinical_doc_era_en.html)) | [explain_clinical_doc_era](https://nlp.johnsnowlabs.com/2021/04/01/explain_clinical_doc_era_en.html)  |


## New Open Source Models and Pipelines


| nlu.load() Refrence                                          | Spark NLP Refrence                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [en.embed.distilbert](https://nlp.johnsnowlabs.com//2021/05/20/distilbert_base_cased_en.html) | [distilbert_base_cased](https://nlp.johnsnowlabs.com//2021/05/20/distilbert_base_cased_en.html) |
| [en.embed.distilbert.base](https://nlp.johnsnowlabs.com//2021/05/20/distilbert_base_cased_en.html) | [distilbert_base_cased](https://nlp.johnsnowlabs.com//2021/05/20/distilbert_base_cased_en.html) |
| [en.embed.distilbert.base.uncased](https://nlp.johnsnowlabs.com//2021/05/20/distilbert_base_uncased_en.html) | [distilbert_base_uncased](https://nlp.johnsnowlabs.com//2021/05/20/distilbert_base_uncased_en.html) |
| [en.embed.distilroberta](https://nlp.johnsnowlabs.com//2021/05/20/distilroberta_base_en.html) | [distilroberta_base](https://nlp.johnsnowlabs.com//2021/05/20/distilroberta_base_en.html) |
| [en.embed.roberta](https://nlp.johnsnowlabs.com//2021/05/20/roberta_base_en.html) | [roberta_base](https://nlp.johnsnowlabs.com//2021/05/20/roberta_base_en.html) |
| [en.embed.roberta.base](https://nlp.johnsnowlabs.com//2021/05/20/roberta_base_en.html) | [roberta_base](https://nlp.johnsnowlabs.com//2021/05/20/roberta_base_en.html) |
| [en.embed.roberta.large](https://nlp.johnsnowlabs.com//2021/05/20/roberta_large_en.html) | [roberta_large](https://nlp.johnsnowlabs.com//2021/05/20/roberta_large_en.html) |
| [xx.marian](https://nlp.johnsnowlabs.com//2020/12/28/opus_mt_en_fr_xx.html) | [opus_mt_en_fr](https://nlp.johnsnowlabs.com//2020/12/28/opus_mt_en_fr_xx.html) |
| [xx.embed.distilbert.](https://nlp.johnsnowlabs.com//2021/05/20/distilbert_base_multilingual_cased_xx.html) | [distilbert_base_multilingual_cased](https://nlp.johnsnowlabs.com//2021/05/20/distilbert_base_multilingual_cased_xx.html) |
| [xx.embed.xlm](https://nlp.johnsnowlabs.com//2021/05/25/xlm_roberta_base_xx.html) | [xlm_roberta_base](https://nlp.johnsnowlabs.com//2021/05/25/xlm_roberta_base_xx.html) |
| [xx.embed.xlm.base](https://nlp.johnsnowlabs.com//2021/05/25/xlm_roberta_base_xx.html) | [xlm_roberta_base](https://nlp.johnsnowlabs.com//2021/05/25/xlm_roberta_base_xx.html) |
| [xx.embed.xlm.twitter](https://nlp.johnsnowlabs.com//2021/05/25/twitter_xlm_roberta_base_xx.html) | [twitter_xlm_roberta_base](https://nlp.johnsnowlabs.com//2021/05/25/twitter_xlm_roberta_base_xx.html) |
| [zh.embed.bert](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_chinese_zh.html) | [bert_base_chinese](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_chinese_zh.html) |
| [zh.embed.bert.wwm](https://nlp.johnsnowlabs.com//2021/05/20/chinese_bert_wwm_zh.html) | [chinese_bert_wwm](https://nlp.johnsnowlabs.com//2021/05/20/chinese_bert_wwm_zh.html) |
| [de.embed.bert](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_german_cased_de.html) | [bert_base_german_cased](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_german_cased_de.html) |
| [de.embed.bert.uncased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_german_uncased_de.html) |  [bert_base_german_uncased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_german_uncased_de.html) |
| [nl.embed.bert](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_dutch_cased_nl.html) | [bert_base_dutch_cased](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_dutch_cased_nl.html) |
| [it.embed.bert](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_italian_cased_it.html) | [bert_base_italian_cased](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_italian_cased_it.html) |
| [tr.embed.bert](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_turkish_cased_tr.html) | [bert_base_turkish_cased](https://nlp.johnsnowlabs.com//2021/05/20/bert_base_turkish_cased_tr.html) |
| [tr.embed.bert.uncased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_turkish_uncased_tr.html) |   [bert_base_turkish_uncased](https://nlp.johnsnowlabs.com/2021/05/20/bert_base_turkish_uncased_tr.html) |
| [xx.fr.marian.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_fr_xx.html) | [opus_mt_bcl_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_fr_xx.html) |
| [xx.tr.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_tr_xx.html) | [opus_mt_ar_tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_tr_xx.html) |
| [xx.sv.marian.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_sv_xx.html) | [opus_mt_af_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_sv_xx.html) |
| [xx.de.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_de_xx.html) | [opus_mt_ar_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_de_xx.html) |
| [xx.fr.marian.translate_to.bi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bi_fr_xx.html) | [opus_mt_bi_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bi_fr_xx.html) |
| [xx.es.marian.translate_to.bi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bi_es_xx.html) | [opus_mt_bi_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bi_es_xx.html) |
| [xx.fi.marian.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_fi_xx.html) | [opus_mt_af_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_fi_xx.html) |
| [xx.fi.marian.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_crs_fi_xx.html) | [opus_mt_crs_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_crs_fi_xx.html) |
| [xx.fi.marian.translate_to.bem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bem_fi_xx.html) | [opus_mt_bem_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bem_fi_xx.html) |
| [xx.sv.marian.translate_to.bem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bem_sv_xx.html) | [opus_mt_bem_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bem_sv_xx.html) |
| [xx.it.marian.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_it_xx.html) | [opus_mt_ca_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_it_xx.html) |
| [xx.fr.marian.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_fr_xx.html) | [opus_mt_ca_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_fr_xx.html) |
| [xx.es.marian.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_es_xx.html) | [opus_mt_bcl_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_es_xx.html) |
| [xx.uk.marian.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_uk_xx.html) | [opus_mt_ca_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_uk_xx.html) |
| [xx.fr.marian.translate_to.bem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bem_fr_xx.html) | [opus_mt_bem_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bem_fr_xx.html) |
| [xx.de.marian.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_de_xx.html) | [opus_mt_af_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_de_xx.html) |
| [xx.nl.marian.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_nl_xx.html) | [opus_mt_af_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_nl_xx.html) |
| [xx.fr.marian.translate_to.ase](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ase_fr_xx.html) | [opus_mt_ase_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ase_fr_xx.html) |
| [xx.es.marian.translate_to.az](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_az_es_xx.html) | [opus_mt_az_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_az_es_xx.html) |
| [xx.es.marian.translate_to.chk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_chk_es_xx.html) | [opus_mt_chk_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_chk_es_xx.html) |
| [xx.sv.marian.translate_to.ceb](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ceb_sv_xx.html) | [opus_mt_ceb_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ceb_sv_xx.html) |
| [xx.es.marian.translate_to.ceb](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ceb_es_xx.html) | [opus_mt_ceb_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ceb_es_xx.html) |
| [xx.es.marian.translate_to.aed](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_aed_es_xx.html) | [opus_mt_aed_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_aed_es_xx.html) |
| [xx.pl.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_pl_xx.html) | [opus_mt_ar_pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_pl_xx.html) |
| [xx.es.marian.translate_to.bem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bem_es_xx.html) | [opus_mt_bem_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bem_es_xx.html) |
| [xx.eo.marian.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_eo_xx.html) | [opus_mt_af_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_eo_xx.html) |
| [xx.fr.marian.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_fr_xx.html) | [opus_mt_cs_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_fr_xx.html) |
| [xx.fi.marian.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_fi_xx.html) | [opus_mt_bcl_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_fi_xx.html) |
| [xx.es.marian.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_crs_es_xx.html) | [opus_mt_crs_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_crs_es_xx.html) |
| [xx.sv.marian.translate_to.bi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bi_sv_xx.html) | [opus_mt_bi_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bi_sv_xx.html) |
| [xx.de.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_de_xx.html) | [opus_mt_bg_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_de_xx.html) |
| [xx.ru.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_ru_xx.html) | [opus_mt_ar_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_ru_xx.html) |
| [xx.es.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_es_xx.html) | [opus_mt_bg_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_es_xx.html) |
| [xx.uk.marian.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_uk_xx.html) | [opus_mt_cs_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_uk_xx.html) |
| [xx.sv.marian.translate_to.bzs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bzs_sv_xx.html) | [opus_mt_bzs_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bzs_sv_xx.html) |
| [xx.es.marian.translate_to.be](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_be_es_xx.html) | [opus_mt_be_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_be_es_xx.html) |
| [xx.es.marian.translate_to.bzs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bzs_es_xx.html) | [opus_mt_bzs_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bzs_es_xx.html) |
| [xx.fr.marian.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_fr_xx.html) | [opus_mt_af_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_fr_xx.html) |
| [xx.pt.marian.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_pt_xx.html) | [opus_mt_ca_pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_pt_xx.html) |
| [xx.fr.marian.translate_to.chk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_chk_fr_xx.html) | [opus_mt_chk_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_chk_fr_xx.html) |
| [xx.de.marian.translate_to.ase](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ase_de_xx.html) | [opus_mt_ase_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ase_de_xx.html) |
| [xx.it.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_it_xx.html) | [opus_mt_ar_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_it_xx.html) |
| [xx.fi.marian.translate_to.ceb](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ceb_fi_xx.html) | [opus_mt_ceb_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ceb_fi_xx.html) |
| [xx.cpp.marian.translate_to.cpp](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cpp_cpp_xx.html) | [opus_mt_cpp_cpp](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cpp_cpp_xx.html) |
| [xx.fr.marian.translate_to.ber](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ber_fr_xx.html) | [opus_mt_ber_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ber_fr_xx.html) |
| [xx.ru.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_ru_xx.html) | [opus_mt_bg_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_ru_xx.html) |
| [xx.es.marian.translate_to.ase](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ase_es_xx.html) | [opus_mt_ase_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ase_es_xx.html) |
| [xx.es.marian.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_es_xx.html) | [opus_mt_af_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_es_xx.html) |
| [xx.it.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_it_xx.html) | [opus_mt_bg_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_it_xx.html) |
| [xx.sv.marian.translate_to.am](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_am_sv_xx.html) | [opus_mt_am_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_am_sv_xx.html) |
| [xx.eo.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_eo_xx.html) | [opus_mt_ar_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_eo_xx.html) |
| [xx.fr.marian.translate_to.ceb](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ceb_fr_xx.html) | [opus_mt_ceb_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ceb_fr_xx.html) |
| [xx.es.marian.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_es_xx.html) | [opus_mt_ca_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_es_xx.html) |
| [xx.fi.marian.translate_to.bzs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bzs_fi_xx.html) | [opus_mt_bzs_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bzs_fi_xx.html) |
| [xx.de.marian.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_crs_de_xx.html) | [opus_mt_crs_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_crs_de_xx.html) |
| [xx.fi.marian.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_fi_xx.html) | [opus_mt_cs_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_fi_xx.html) |
| [xx.afa.marian.translate_to.afa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_afa_afa_xx.html) | [opus_mt_afa_afa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_afa_afa_xx.html) |
| [xx.sv.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_sv_xx.html) | [opus_mt_bg_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_sv_xx.html) |
| [xx.tr.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_tr_xx.html) | [opus_mt_bg_tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_tr_xx.html) |
| [xx.fr.marian.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_crs_fr_xx.html) | [opus_mt_crs_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_crs_fr_xx.html) |
| [xx.sv.marian.translate_to.ase](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ase_sv_xx.html) | [opus_mt_ase_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ase_sv_xx.html) |
| [xx.de.marian.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_de_xx.html) | [opus_mt_cs_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_de_xx.html) |
| [xx.eo.marian.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_eo_xx.html) | [opus_mt_cs_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_cs_eo_xx.html) |
| [xx.sv.marian.translate_to.chk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_chk_sv_xx.html) | [opus_mt_chk_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_chk_sv_xx.html) |
| [xx.sv.marian.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_sv_xx.html) | [opus_mt_bcl_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_sv_xx.html) |
| [xx.fr.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_fr_xx.html) | [opus_mt_ar_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_fr_xx.html) |
| [xx.ru.marian.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_ru_xx.html) | [opus_mt_af_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_af_ru_xx.html) |
| [xx.he.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_he_xx.html) | [opus_mt_ar_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_he_xx.html) |
| [xx.fi.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_fi_xx.html) | [opus_mt_bg_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_fi_xx.html) |
| [xx.es.marian.translate_to.ber](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ber_es_xx.html) | [opus_mt_ber_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ber_es_xx.html) |
| [xx.es.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_es_xx.html) | [opus_mt_ar_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_es_xx.html) |
| [xx.uk.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_uk_xx.html) | [opus_mt_bg_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_uk_xx.html) |
| [xx.fr.marian.translate_to.bzs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bzs_fr_xx.html) | [opus_mt_bzs_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bzs_fr_xx.html) |
| [xx.el.marian.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_el_xx.html) | [opus_mt_ar_el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ar_el_xx.html) |
| [xx.nl.marian.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_nl_xx.html) | [opus_mt_ca_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ca_nl_xx.html) |
| [xx.de.marian.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_de_xx.html) | [opus_mt_bcl_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bcl_de_xx.html) |
| [xx.eo.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_eo_xx.html) | [opus_mt_bg_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_bg_eo_xx.html) |
| [xx.de.marian.translate_to.efi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_efi_de_xx.html) | [opus_mt_efi_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_efi_de_xx.html) |
| [xx.bzs.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_bzs_xx.html) | [opus_mt_de_bzs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_bzs_xx.html) |
| [xx.fj.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_fj_xx.html) | [opus_mt_de_fj](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_fj_xx.html) |
| [xx.fi.marian.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_fi_xx.html) | [opus_mt_da_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_fi_xx.html) |
| [xx.no.marian.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_no_xx.html) | [opus_mt_da_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_no_xx.html) |
| [xx.cs.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_cs_xx.html) | [opus_mt_de_cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_cs_xx.html) |
| [xx.efi.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_efi_xx.html) | [opus_mt_de_efi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_efi_xx.html) |
| [xx.gil.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_gil_xx.html) | [opus_mt_de_gil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_gil_xx.html) |
| [xx.bcl.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_bcl_xx.html) | [opus_mt_de_bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_bcl_xx.html) |
| [xx.pag.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pag_xx.html) | [opus_mt_de_pag](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pag_xx.html) |
| [xx.kg.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_kg_xx.html) | [opus_mt_de_kg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_kg_xx.html) |
| [xx.fi.marian.translate_to.efi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_efi_fi_xx.html) | [opus_mt_efi_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_efi_fi_xx.html) |
| [xx.is.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_is_xx.html) | [opus_mt_de_is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_is_xx.html) |
| [xx.fr.marian.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_fr_xx.html) | [opus_mt_da_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_fr_xx.html) |
| [xx.pl.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pl_xx.html) | [opus_mt_de_pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pl_xx.html) |
| [xx.ln.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ln_xx.html) | [opus_mt_de_ln](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ln_xx.html) |
| [xx.pap.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pap_xx.html) | [opus_mt_de_pap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pap_xx.html) |
| [xx.vi.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_vi_xx.html) | [opus_mt_de_vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_vi_xx.html) |
| [xx.no.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_no_xx.html) | [opus_mt_de_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_no_xx.html) |
| [xx.eo.marian.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_eo_xx.html) | [opus_mt_el_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_eo_xx.html) |
| [xx.af.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_af_xx.html) | [opus_mt_de_af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_af_xx.html) |
| [xx.es.marian.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_es_xx.html) | [opus_mt_ee_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_es_xx.html) |
| [xx.eo.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_eo_xx.html) | [opus_mt_de_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_eo_xx.html) |
| [xx.bi.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_bi_xx.html) | [opus_mt_de_bi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_bi_xx.html) |
| [xx.mt.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_mt_xx.html) | [opus_mt_de_mt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_mt_xx.html) |
| [xx.lt.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_lt_xx.html) | [opus_mt_de_lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_lt_xx.html) |
| [xx.bg.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_bg_xx.html) | [opus_mt_de_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_bg_xx.html) |
| [xx.hil.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_hil_xx.html) | [opus_mt_de_hil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_hil_xx.html) |
| [xx.eu.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_eu_xx.html) | [opus_mt_de_eu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_eu_xx.html) |
| [xx.da.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_da_xx.html) | [opus_mt_de_da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_da_xx.html) |
| [xx.ms.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ms_xx.html) | [opus_mt_de_ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ms_xx.html) |
| [xx.he.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_he_xx.html) | [opus_mt_de_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_he_xx.html) |
| [xx.et.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_et_xx.html) | [opus_mt_de_et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_et_xx.html) |
| [xx.es.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_es_xx.html) | [opus_mt_de_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_es_xx.html) |
| [xx.fr.marian.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_fr_xx.html) | [opus_mt_el_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_fr_xx.html) |
| [xx.fr.marian.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_fr_xx.html) | [opus_mt_ee_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_fr_xx.html) |
| [xx.el.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_el_xx.html) | [opus_mt_de_el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_el_xx.html) |
| [xx.sv.marian.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_sv_xx.html) | [opus_mt_el_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_sv_xx.html) |
| [xx.es.marian.translate_to.csn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_csn_es_xx.html) | [opus_mt_csn_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_csn_es_xx.html) |
| [xx.tl.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_tl_xx.html) | [opus_mt_de_tl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_tl_xx.html) |
| [xx.pon.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pon_xx.html) | [opus_mt_de_pon](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pon_xx.html) |
| [xx.fr.marian.translate_to.efi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_efi_fr_xx.html) | [opus_mt_efi_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_efi_fr_xx.html) |
| [xx.uk.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_uk_xx.html) | [opus_mt_de_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_uk_xx.html) |
| [xx.ar.marian.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_ar_xx.html) | [opus_mt_el_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_ar_xx.html) |
| [xx.fi.marian.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_fi_xx.html) | [opus_mt_el_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_el_fi_xx.html) |
| [xx.ig.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ig_xx.html) | [opus_mt_de_ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ig_xx.html) |
| [xx.guw.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_guw_xx.html) | [opus_mt_de_guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_guw_xx.html) |
| [xx.iso.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_iso_xx.html) | [opus_mt_de_iso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_iso_xx.html) |
| [xx.sv.marian.translate_to.efi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_efi_sv_xx.html) | [opus_mt_efi_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_efi_sv_xx.html) |
| [xx.ha.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ha_xx.html) | [opus_mt_de_ha](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ha_xx.html) |
| [xx.fr.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_fr_xx.html) | [opus_mt_de_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_fr_xx.html) |
| [xx.gaa.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_gaa_xx.html) | [opus_mt_de_gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_gaa_xx.html) |
| [xx.nso.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_nso_xx.html) | [opus_mt_de_nso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_nso_xx.html) |
| [xx.ht.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ht_xx.html) | [opus_mt_de_ht](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ht_xx.html) |
| [xx.nl.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_nl_xx.html) | [opus_mt_de_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_nl_xx.html) |
| [xx.sv.marian.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_sv_xx.html) | [opus_mt_ee_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_sv_xx.html) |
| [xx.fi.marian.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_fi_xx.html) | [opus_mt_ee_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_fi_xx.html) |
| [xx.de.marian.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_de_xx.html) | [opus_mt_ee_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ee_de_xx.html) |
| [xx.eo.marian.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_eo_xx.html) | [opus_mt_da_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_eo_xx.html) |
| [xx.es.marian.translate_to.csg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_csg_es_xx.html) | [opus_mt_csg_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_csg_es_xx.html) |
| [xx.de.marian.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_de_xx.html) | [opus_mt_da_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_de_xx.html) |
| [xx.ar.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ar_xx.html) | [opus_mt_de_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ar_xx.html) |
| [xx.hu.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_hu_xx.html) | [opus_mt_de_hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_hu_xx.html) |
| [xx.ca.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ca_xx.html) | [opus_mt_de_ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ca_xx.html) |
| [xx.pis.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pis_xx.html) | [opus_mt_de_pis](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_pis_xx.html) |
| [xx.ho.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ho_xx.html) | [opus_mt_de_ho](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ho_xx.html) |
| [xx.de.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_de_xx.html) | [opus_mt_de_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_de_xx.html) |
| [xx.lua.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_lua_xx.html) | [opus_mt_de_lua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_lua_xx.html) |
| [xx.loz.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_loz_xx.html) | [opus_mt_de_loz](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_loz_xx.html) |
| [xx.crs.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_crs_xx.html) | [opus_mt_de_crs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_crs_xx.html) |
| [xx.es.marian.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_es_xx.html) | [opus_mt_da_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_da_es_xx.html) |
| [xx.ee.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ee_xx.html) | [opus_mt_de_ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ee_xx.html) |
| [xx.it.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_it_xx.html) | [opus_mt_de_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_it_xx.html) |
| [xx.ilo.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ilo_xx.html) | [opus_mt_de_ilo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ilo_xx.html) |
| [xx.ny.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ny_xx.html) | [opus_mt_de_ny](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ny_xx.html) |
| [xx.fi.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_fi_xx.html) | [opus_mt_de_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_fi_xx.html) |
| [xx.ase.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ase_xx.html) | [opus_mt_de_ase](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_ase_xx.html) |
| [xx.hr.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_hr_xx.html) | [opus_mt_de_hr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_de_hr_xx.html) |
| [xx.sl.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sl_xx.html) | [opus_mt_fi_sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sl_xx.html) |
| [xx.sk.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sk_xx.html) | [opus_mt_fi_sk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sk_xx.html) |
| [xx.ru.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ru_xx.html) | [opus_mt_es_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ru_xx.html) |
| [xx.sn.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sn_xx.html) | [opus_mt_fi_sn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sn_xx.html) |
| [xx.pl.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_pl_xx.html) | [opus_mt_eo_pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_pl_xx.html) |
| [xx.cs.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_cs_xx.html) | [opus_mt_es_cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_cs_xx.html) |
| [xx.wls.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_wls_xx.html) | [opus_mt_fi_wls](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_wls_xx.html) |
| [xx.gaa.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_gaa_xx.html) | [opus_mt_fi_gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_gaa_xx.html) |
| [xx.is.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_is_xx.html) | [opus_mt_fi_is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_is_xx.html) |
| [xx.ha.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ha_xx.html) | [opus_mt_es_ha](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ha_xx.html) |
| [xx.nl.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_nl_xx.html) | [opus_mt_es_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_nl_xx.html) |
| [xx.ha.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ha_xx.html) | [opus_mt_fi_ha](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ha_xx.html) |
| [xx.fj.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_fj_xx.html) | [opus_mt_fi_fj](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_fj_xx.html) |
| [xx.ber.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ber_xx.html) | [opus_mt_es_ber](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ber_xx.html) |
| [xx.ho.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ho_xx.html) | [opus_mt_fi_ho](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ho_xx.html) |
| [xx.ny.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ny_xx.html) | [opus_mt_fi_ny](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ny_xx.html) |
| [xx.sl.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_sl_xx.html) | [opus_mt_es_sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_sl_xx.html) |
| [xx.ts.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ts_xx.html) | [opus_mt_fi_ts](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ts_xx.html) |
| [xx.el.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_el_xx.html) | [opus_mt_eo_el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_el_xx.html) |
| [xx.war.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_war_xx.html) | [opus_mt_fi_war](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_war_xx.html) |
| [xx.cs.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_cs_xx.html) | [opus_mt_fi_cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_cs_xx.html) |
| [xx.loz.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_loz_xx.html) | [opus_mt_es_loz](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_loz_xx.html) |
| [xx.mk.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mk_xx.html) | [opus_mt_fi_mk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mk_xx.html) |
| [xx.bg.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_bg_xx.html) | [opus_mt_es_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_bg_xx.html) |
| [xx.srn.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_srn_xx.html) | [opus_mt_fi_srn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_srn_xx.html) |
| [xx.is.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_is_xx.html) | [opus_mt_es_is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_is_xx.html) |
| [xx.hu.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_hu_xx.html) | [opus_mt_eo_hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_hu_xx.html) |
| [xx.tw.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tw_xx.html) | [opus_mt_fi_tw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tw_xx.html) |
| [xx.mt.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mt_xx.html) | [opus_mt_fi_mt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mt_xx.html) |
| [xx.fr.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_fr_xx.html) | [opus_mt_es_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_fr_xx.html) |
| [xx.yo.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_yo_xx.html) | [opus_mt_es_yo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_yo_xx.html) |
| [xx.xh.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_xh_xx.html) | [opus_mt_fi_xh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_xh_xx.html) |
| [xx.lv.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lv_xx.html) | [opus_mt_fi_lv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lv_xx.html) |
| [xx.de.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_de_xx.html) | [opus_mt_fi_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_de_xx.html) |
| [xx.ve.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ve_xx.html) | [opus_mt_es_ve](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ve_xx.html) |
| [xx.es.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_es_xx.html) | [opus_mt_fi_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_es_xx.html) |
| [xx.eo.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_eo_xx.html) | [opus_mt_es_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_eo_xx.html) |
| [xx.cs.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_cs_xx.html) | [opus_mt_eo_cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_cs_xx.html) |
| [xx.mt.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_mt_xx.html) | [opus_mt_es_mt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_mt_xx.html) |
| [xx.el.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_el_xx.html) | [opus_mt_es_el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_el_xx.html) |
| [xx.ee.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ee_xx.html) | [opus_mt_es_ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ee_xx.html) |
| [xx.de.marian.translate_to.eu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eu_de_xx.html) | [opus_mt_eu_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eu_de_xx.html) |
| [xx.et.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_et_xx.html) | [opus_mt_es_et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_et_xx.html) |
| [xx.fi.marian.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_fi_xx.html) | [opus_mt_et_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_fi_xx.html) |
| [xx.wls.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_wls_xx.html) | [opus_mt_es_wls](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_wls_xx.html) |
| [xx.mg.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mg_xx.html) | [opus_mt_fi_mg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mg_xx.html) |
| [xx.eu.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_eu_xx.html) | [opus_mt_es_eu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_eu_xx.html) |
| [xx.lua.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_lua_xx.html) | [opus_mt_es_lua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_lua_xx.html) |
| [xx.pon.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pon_xx.html) | [opus_mt_es_pon](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pon_xx.html) |
| [xx.mfe.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mfe_xx.html) | [opus_mt_fi_mfe](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mfe_xx.html) |
| [xx.he.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_he_xx.html) | [opus_mt_eo_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_he_xx.html) |
| [xx.id.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_id_xx.html) | [opus_mt_es_id](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_id_xx.html) |
| [xx.xh.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_xh_xx.html) | [opus_mt_es_xh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_xh_xx.html) |
| [xx.ar.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ar_xx.html) | [opus_mt_es_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ar_xx.html) |
| [xx.crs.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_crs_xx.html) | [opus_mt_es_crs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_crs_xx.html) |
| [xx.es.marian.translate_to.eu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eu_es_xx.html) | [opus_mt_eu_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eu_es_xx.html) |
| [xx.tpi.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tpi_xx.html) | [opus_mt_fi_tpi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tpi_xx.html) |
| [xx.pis.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_pis_xx.html) | [opus_mt_fi_pis](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_pis_xx.html) |
| [xx.vi.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_vi_xx.html) | [opus_mt_es_vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_vi_xx.html) |
| [xx.es.marian.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_es_xx.html) | [opus_mt_et_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_es_xx.html) |
| [xx.rw.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_rw_xx.html) | [opus_mt_fi_rw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_rw_xx.html) |
| [xx.gl.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_gl_xx.html) | [opus_mt_es_gl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_gl_xx.html) |
| [xx.pt.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_pt_xx.html) | [opus_mt_eo_pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_pt_xx.html) |
| [xx.he.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_he_xx.html) | [opus_mt_fi_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_he_xx.html) |
| [xx.af.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_af_xx.html) | [opus_mt_fi_af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_af_xx.html) |
| [xx.ru.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ru_xx.html) | [opus_mt_fi_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ru_xx.html) |
| [xx.ve.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ve_xx.html) | [opus_mt_fi_ve](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ve_xx.html) |
| [xx.ca.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ca_xx.html) | [opus_mt_es_ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ca_xx.html) |
| [xx.tr.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tr_xx.html) | [opus_mt_fi_tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tr_xx.html) |
| [xx.ht.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ht_xx.html) | [opus_mt_fi_ht](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ht_xx.html) |
| [xx.nl.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_nl_xx.html) | [opus_mt_fi_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_nl_xx.html) |
| [xx.iso.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_iso_xx.html) | [opus_mt_fi_iso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_iso_xx.html) |
| [xx.fi.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_fi_xx.html) | [opus_mt_es_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_fi_xx.html) |
| [xx.da.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_da_xx.html) | [opus_mt_eo_da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_da_xx.html) |
| [xx.ln.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ln_xx.html) | [opus_mt_es_ln](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ln_xx.html) |
| [xx.csn.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_csn_xx.html) | [opus_mt_es_csn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_csn_xx.html) |
| [xx.pon.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_pon_xx.html) | [opus_mt_fi_pon](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_pon_xx.html) |
| [xx.af.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_af_xx.html) | [opus_mt_eo_af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_af_xx.html) |
| [xx.bzs.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_bzs_xx.html) | [opus_mt_fi_bzs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_bzs_xx.html) |
| [xx.no.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_no_xx.html) | [opus_mt_es_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_no_xx.html) |
| [xx.es.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_es_xx.html) | [opus_mt_es_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_es_xx.html) |
| [xx.lua.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lua_xx.html) | [opus_mt_fi_lua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lua_xx.html) |
| [xx.yua.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_yua_xx.html) | [opus_mt_es_yua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_yua_xx.html) |
| [xx.ru.marian.translate_to.eu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eu_ru_xx.html) | [opus_mt_eu_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eu_ru_xx.html) |
| [xx.tpi.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tpi_xx.html) | [opus_mt_es_tpi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tpi_xx.html) |
| [xx.lue.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lue_xx.html) | [opus_mt_fi_lue](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lue_xx.html) |
| [xx.sv.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_sv_xx.html) | [opus_mt_eo_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_sv_xx.html) |
| [xx.niu.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_niu_xx.html) | [opus_mt_es_niu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_niu_xx.html) |
| [xx.tiv.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tiv_xx.html) | [opus_mt_fi_tiv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tiv_xx.html) |
| [xx.pag.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pag_xx.html) | [opus_mt_es_pag](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pag_xx.html) |
| [xx.run.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_run_xx.html) | [opus_mt_fi_run](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_run_xx.html) |
| [xx.ty.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ty_xx.html) | [opus_mt_es_ty](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ty_xx.html) |
| [xx.gil.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_gil_xx.html) | [opus_mt_es_gil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_gil_xx.html) |
| [xx.ln.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ln_xx.html) | [opus_mt_fi_ln](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ln_xx.html) |
| [xx.ty.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ty_xx.html) | [opus_mt_fi_ty](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ty_xx.html) |
| [xx.prl.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_prl_xx.html) | [opus_mt_es_prl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_prl_xx.html) |
| [xx.kg.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_kg_xx.html) | [opus_mt_es_kg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_kg_xx.html) |
| [xx.rw.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_rw_xx.html) | [opus_mt_es_rw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_rw_xx.html) |
| [xx.kqn.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_kqn_xx.html) | [opus_mt_fi_kqn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_kqn_xx.html) |
| [xx.sq.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sq_xx.html) | [opus_mt_fi_sq](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sq_xx.html) |
| [xx.sw.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sw_xx.html) | [opus_mt_fi_sw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sw_xx.html) |
| [xx.csg.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_csg_xx.html) | [opus_mt_es_csg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_csg_xx.html) |
| [xx.ro.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ro_xx.html) | [opus_mt_es_ro](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ro_xx.html) |
| [xx.ee.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ee_xx.html) | [opus_mt_fi_ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ee_xx.html) |
| [xx.ilo.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ilo_xx.html) | [opus_mt_fi_ilo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ilo_xx.html) |
| [xx.eo.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_eo_xx.html) | [opus_mt_fi_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_eo_xx.html) |
| [xx.iso.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_iso_xx.html) | [opus_mt_es_iso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_iso_xx.html) |
| [xx.bem.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_bem_xx.html) | [opus_mt_fi_bem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_bem_xx.html) |
| [xx.tn.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tn_xx.html) | [opus_mt_fi_tn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tn_xx.html) |
| [xx.da.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_da_xx.html) | [opus_mt_es_da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_da_xx.html) |
| [xx.es.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_es_xx.html) | [opus_mt_eo_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_es_xx.html) |
| [xx.ru.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_ru_xx.html) | [opus_mt_eo_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_ru_xx.html) |
| [xx.rn.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_rn_xx.html) | [opus_mt_es_rn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_rn_xx.html) |
| [xx.lt.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_lt_xx.html) | [opus_mt_es_lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_lt_xx.html) |
| [xx.guw.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_guw_xx.html) | [opus_mt_es_guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_guw_xx.html) |
| [xx.tvl.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tvl_xx.html) | [opus_mt_es_tvl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tvl_xx.html) |
| [xx.fr.marian.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_fr_xx.html) | [opus_mt_et_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_fr_xx.html) |
| [xx.ht.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ht_xx.html) | [opus_mt_es_ht](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ht_xx.html) |
| [xx.mos.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mos_xx.html) | [opus_mt_fi_mos](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mos_xx.html) |
| [xx.ase.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ase_xx.html) | [opus_mt_es_ase](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ase_xx.html) |
| [xx.crs.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_crs_xx.html) | [opus_mt_fi_crs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_crs_xx.html) |
| [xx.bcl.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_bcl_xx.html) | [opus_mt_fi_bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_bcl_xx.html) |
| [xx.tvl.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tvl_xx.html) | [opus_mt_fi_tvl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tvl_xx.html) |
| [xx.lus.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lus_xx.html) | [opus_mt_fi_lus](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lus_xx.html) |
| [xx.he.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_he_xx.html) | [opus_mt_es_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_he_xx.html) |
| [xx.pis.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pis_xx.html) | [opus_mt_es_pis](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pis_xx.html) |
| [xx.it.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_it_xx.html) | [opus_mt_es_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_it_xx.html) |
| [xx.fi.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_fi_xx.html) | [opus_mt_eo_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_fi_xx.html) |
| [xx.tw.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tw_xx.html) | [opus_mt_es_tw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tw_xx.html) |
| [xx.aed.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_aed_xx.html) | [opus_mt_es_aed](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_aed_xx.html) |
| [xx.bzs.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_bzs_xx.html) | [opus_mt_es_bzs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_bzs_xx.html) |
| [xx.nso.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_nso_xx.html) | [opus_mt_fi_nso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_nso_xx.html) |
| [xx.gaa.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_gaa_xx.html) | [opus_mt_es_gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_gaa_xx.html) |
| [xx.zai.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_zai_xx.html) | [opus_mt_es_zai](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_zai_xx.html) |
| [xx.no.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_no_xx.html) | [opus_mt_fi_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_no_xx.html) |
| [xx.uk.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_uk_xx.html) | [opus_mt_fi_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_uk_xx.html) |
| [xx.sg.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_sg_xx.html) | [opus_mt_es_sg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_sg_xx.html) |
| [xx.ilo.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ilo_xx.html) | [opus_mt_es_ilo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ilo_xx.html) |
| [xx.bg.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_bg_xx.html) | [opus_mt_eo_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_bg_xx.html) |
| [xx.pap.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_pap_xx.html) | [opus_mt_fi_pap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_pap_xx.html) |
| [xx.ho.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ho_xx.html) | [opus_mt_es_ho](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ho_xx.html) |
| [xx.toi.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_toi_xx.html) | [opus_mt_fi_toi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_toi_xx.html) |
| [xx.st.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_st_xx.html) | [opus_mt_es_st](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_st_xx.html) |
| [xx.to.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_to_xx.html) | [opus_mt_fi_to](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_to_xx.html) |
| [xx.kg.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_kg_xx.html) | [opus_mt_fi_kg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_kg_xx.html) |
| [xx.sv.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sv_xx.html) | [opus_mt_fi_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sv_xx.html) |
| [xx.tll.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tll_xx.html) | [opus_mt_fi_tll](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_tll_xx.html) |
| [xx.ceb.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ceb_xx.html) | [opus_mt_es_ceb](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ceb_xx.html) |
| [xx.ig.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ig_xx.html) | [opus_mt_es_ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ig_xx.html) |
| [xx.sv.marian.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_sv_xx.html) | [opus_mt_et_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_sv_xx.html) |
| [xx.af.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_af_xx.html) | [opus_mt_es_af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_af_xx.html) |
| [xx.pl.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pl_xx.html) | [opus_mt_es_pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pl_xx.html) |
| [xx.ro.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_ro_xx.html) | [opus_mt_eo_ro](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_ro_xx.html) |
| [xx.tn.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tn_xx.html) | [opus_mt_es_tn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tn_xx.html) |
| [xx.sm.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sm_xx.html) | [opus_mt_fi_sm](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sm_xx.html) |
| [xx.mk.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_mk_xx.html) | [opus_mt_es_mk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_mk_xx.html) |
| [xx.id.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_id_xx.html) | [opus_mt_fi_id](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_id_xx.html) |
| [xx.hr.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_hr_xx.html) | [opus_mt_fi_hr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_hr_xx.html) |
| [xx.sg.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sg_xx.html) | [opus_mt_fi_sg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_sg_xx.html) |
| [xx.hil.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_hil_xx.html) | [opus_mt_fi_hil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_hil_xx.html) |
| [xx.nl.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_nl_xx.html) | [opus_mt_eo_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_nl_xx.html) |
| [xx.pap.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pap_xx.html) | [opus_mt_es_pap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_pap_xx.html) |
| [xx.fr.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_fr_xx.html) | [opus_mt_fi_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_fr_xx.html) |
| [xx.bi.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_bi_xx.html) | [opus_mt_es_bi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_bi_xx.html) |
| [xx.fi.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_fi_xx.html) | [opus_mt_fi_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_fi_xx.html) |
| [xx.nso.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_nso_xx.html) | [opus_mt_es_nso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_nso_xx.html) |
| [xx.et.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_et_xx.html) | [opus_mt_fi_et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_et_xx.html) |
| [xx.uk.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_uk_xx.html) | [opus_mt_es_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_uk_xx.html) |
| [xx.sh.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_sh_xx.html) | [opus_mt_eo_sh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_sh_xx.html) |
| [xx.lu.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lu_xx.html) | [opus_mt_fi_lu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lu_xx.html) |
| [xx.gil.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_gil_xx.html) | [opus_mt_fi_gil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_gil_xx.html) |
| [xx.ro.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ro_xx.html) | [opus_mt_fi_ro](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ro_xx.html) |
| [xx.it.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_it_xx.html) | [opus_mt_eo_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_it_xx.html) |
| [xx.hu.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_hu_xx.html) | [opus_mt_fi_hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_hu_xx.html) |
| [xx.bcl.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_bcl_xx.html) | [opus_mt_es_bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_bcl_xx.html) |
| [xx.fse.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_fse_xx.html) | [opus_mt_fi_fse](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_fse_xx.html) |
| [xx.hil.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_hil_xx.html) | [opus_mt_es_hil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_hil_xx.html) |
| [xx.ig.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ig_xx.html) | [opus_mt_fi_ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_ig_xx.html) |
| [xx.tl.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tl_xx.html) | [opus_mt_es_tl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_tl_xx.html) |
| [xx.pag.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_pag_xx.html) | [opus_mt_fi_pag](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_pag_xx.html) |
| [xx.guw.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_guw_xx.html) | [opus_mt_fi_guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_guw_xx.html) |
| [xx.swc.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_swc_xx.html) | [opus_mt_es_swc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_swc_xx.html) |
| [xx.swc.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_swc_xx.html) | [opus_mt_fi_swc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_swc_xx.html) |
| [xx.lg.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lg_xx.html) | [opus_mt_fi_lg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_lg_xx.html) |
| [xx.srn.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_srn_xx.html) | [opus_mt_es_srn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_srn_xx.html) |
| [xx.hr.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_hr_xx.html) | [opus_mt_es_hr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_hr_xx.html) |
| [xx.sm.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_sm_xx.html) | [opus_mt_es_sm](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_sm_xx.html) |
| [xx.de.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_de_xx.html) | [opus_mt_es_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_de_xx.html) |
| [xx.st.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_st_xx.html) | [opus_mt_fi_st](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_st_xx.html) |
| [xx.fr.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_fr_xx.html) | [opus_mt_eo_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_fr_xx.html) |
| [xx.de.marian.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_de_xx.html) | [opus_mt_et_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_de_xx.html) |
| [xx.niu.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_niu_xx.html) | [opus_mt_fi_niu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_niu_xx.html) |
| [xx.el.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_el_xx.html) | [opus_mt_fi_el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_el_xx.html) |
| [xx.efi.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_efi_xx.html) | [opus_mt_fi_efi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_efi_xx.html) |
| [xx.war.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_war_xx.html) | [opus_mt_es_war](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_war_xx.html) |
| [xx.mfs.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_mfs_xx.html) | [opus_mt_es_mfs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_mfs_xx.html) |
| [xx.bg.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_bg_xx.html) | [opus_mt_fi_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_bg_xx.html) |
| [xx.lus.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_lus_xx.html) | [opus_mt_es_lus](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_lus_xx.html) |
| [xx.de.marian.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_de_xx.html) | [opus_mt_eo_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_eo_de_xx.html) |
| [xx.it.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_it_xx.html) | [opus_mt_fi_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_it_xx.html) |
| [xx.efi.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_efi_xx.html) | [opus_mt_es_efi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_efi_xx.html) |
| [xx.ny.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ny_xx.html) | [opus_mt_es_ny](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_ny_xx.html) |
| [xx.fj.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_fj_xx.html) | [opus_mt_es_fj](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_es_fj_xx.html) |
| [xx.ru.marian.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_ru_xx.html) | [opus_mt_et_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_et_ru_xx.html) |
| [xx.mh.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mh_xx.html) | [opus_mt_fi_mh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_mh_xx.html) |
| [xx.es.marian.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_es_xx.html) | [opus_mt_ig_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_es_xx.html) |
| [xx.sv.marian.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_sv_xx.html) | [opus_mt_hu_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_sv_xx.html) |
| [xx.lue.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lue_xx.html) | [opus_mt_fr_lue](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lue_xx.html) |
| [xx.fi.marian.translate_to.ha](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ha_fi_xx.html) | [opus_mt_ha_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ha_fi_xx.html) |
| [xx.ca.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_ca_xx.html) | [opus_mt_it_ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_ca_xx.html) |
| [xx.de.marian.translate_to.ilo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ilo_de_xx.html) | [opus_mt_ilo_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ilo_de_xx.html) |
| [xx.it.marian.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_it_he_xx.html) | [opus_tatoeba_it_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_it_he_xx.html) |
| [xx.loz.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_loz_xx.html) | [opus_mt_fr_loz](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_loz_xx.html) |
| [xx.ms.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ms_xx.html) | [opus_mt_fr_ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ms_xx.html) |
| [xx.uk.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_uk_xx.html) | [opus_mt_it_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_uk_xx.html) |
| [xx.gaa.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_gaa_xx.html) | [opus_mt_fr_gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_gaa_xx.html) |
| [xx.pap.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pap_xx.html) | [opus_mt_fr_pap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pap_xx.html) |
| [xx.fi.marian.translate_to.ilo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ilo_fi_xx.html) | [opus_mt_ilo_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ilo_fi_xx.html) |
| [xx.lg.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lg_xx.html) | [opus_mt_fr_lg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lg_xx.html) |
| [xx.it.marian.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_it_xx.html) | [opus_mt_is_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_it_xx.html) |
| [xx.ms.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_ms_xx.html) | [opus_mt_it_ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_ms_xx.html) |
| [xx.es.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_es_xx.html) | [opus_mt_fr_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_es_xx.html) |
| [xx.ar.marian.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_ar_xx.html) | [opus_mt_he_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_ar_xx.html) |
| [xx.ro.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ro_xx.html) | [opus_mt_fr_ro](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ro_xx.html) |
| [xx.ru.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ru_xx.html) | [opus_mt_fr_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ru_xx.html) |
| [xx.fi.marian.translate_to.ht](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ht_fi_xx.html) | [opus_mt_ht_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ht_fi_xx.html) |
| [xx.bg.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_bg_xx.html) | [opus_mt_it_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_bg_xx.html) |
| [xx.mh.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_mh_xx.html) | [opus_mt_fr_mh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_mh_xx.html) |
| [xx.to.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_to_xx.html) | [opus_mt_fr_to](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_to_xx.html) |
| [xx.sl.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sl_xx.html) | [opus_mt_fr_sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sl_xx.html) |
| [xx.fr.marian.translate_to.gil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gil_fr_xx.html) | [opus_mt_gil_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gil_fr_xx.html) |
| [xx.es.marian.translate_to.hr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hr_es_xx.html) | [opus_mt_hr_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hr_es_xx.html) |
| [xx.ilo.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ilo_xx.html) | [opus_mt_fr_ilo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ilo_xx.html) |
| [xx.ee.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ee_xx.html) | [opus_mt_fr_ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ee_xx.html) |
| [xx.sv.marian.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_sv_xx.html) | [opus_mt_he_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_sv_xx.html) |
| [xx.fr.marian.translate_to.ha](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ha_fr_xx.html) | [opus_mt_ha_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ha_fr_xx.html) |
| [xx.gil.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_gil_xx.html) | [opus_mt_fr_gil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_gil_xx.html) |
| [xx.fi.marian.translate_to.id](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_id_fi_xx.html) | [opus_mt_id_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_id_fi_xx.html) |
| [xx.iir.marian.translate_to.iir](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_iir_iir_xx.html) | [opus_mt_iir_iir](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_iir_iir_xx.html) |
| [xx.pl.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pl_xx.html) | [opus_mt_fr_pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pl_xx.html) |
| [xx.tw.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tw_xx.html) | [opus_mt_fr_tw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tw_xx.html) |
| [xx.sv.marian.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_sv_xx.html) | [opus_mt_gaa_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_sv_xx.html) |
| [xx.ar.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_ar_xx.html) | [opus_mt_it_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_ar_xx.html) |
| [xx.es.marian.translate_to.gil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gil_es_xx.html) | [opus_mt_gil_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gil_es_xx.html) |
| [xx.ase.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ase_xx.html) | [opus_mt_fr_ase](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ase_xx.html) |
| [xx.fr.marian.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_fr_xx.html) | [opus_mt_gaa_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_fr_xx.html) |
| [xx.lus.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lus_xx.html) | [opus_mt_fr_lus](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lus_xx.html) |
| [xx.fr.marian.translate_to.iso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_iso_fr_xx.html) | [opus_mt_iso_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_iso_fr_xx.html) |
| [xx.sm.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sm_xx.html) | [opus_mt_fr_sm](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sm_xx.html) |
| [xx.mfe.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_mfe_xx.html) | [opus_mt_fr_mfe](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_mfe_xx.html) |
| [xx.af.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_af_xx.html) | [opus_mt_fr_af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_af_xx.html) |
| [xx.de.marian.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_de_xx.html) | [opus_mt_ig_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_de_xx.html) |
| [xx.es.marian.translate_to.id](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_id_es_xx.html) | [opus_mt_id_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_id_es_xx.html) |
| [xx.kqn.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_kqn_xx.html) | [opus_mt_fr_kqn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_kqn_xx.html) |
| [xx.zne.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_zne_xx.html) | [opus_mt_fi_zne](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_zne_xx.html) |
| [xx.rw.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_rw_xx.html) | [opus_mt_fr_rw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_rw_xx.html) |
| [xx.ny.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ny_xx.html) | [opus_mt_fr_ny](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ny_xx.html) |
| [xx.ig.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ig_xx.html) | [opus_mt_fr_ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ig_xx.html) |
| [xx.ur.marian.translate_to.hi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hi_ur_xx.html) | [opus_mt_hi_ur](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hi_ur_xx.html) |
| [xx.lt.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_lt_xx.html) | [opus_mt_it_lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_lt_xx.html) |
| [xx.srn.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_srn_xx.html) | [opus_mt_fr_srn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_srn_xx.html) |
| [xx.tiv.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tiv_xx.html) | [opus_mt_fr_tiv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tiv_xx.html) |
| [xx.war.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_war_xx.html) | [opus_mt_fr_war](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_war_xx.html) |
| [xx.fr.marian.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_fr_xx.html) | [opus_mt_is_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_fr_xx.html) |
| [xx.de.marian.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_de_xx.html) | [opus_mt_gaa_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_de_xx.html) |
| [xx.kwy.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_kwy_xx.html) | [opus_mt_fr_kwy](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_kwy_xx.html) |
| [xx.sv.marian.translate_to.gil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gil_sv_xx.html) | [opus_mt_gil_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gil_sv_xx.html) |
| [xx.hr.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_hr_xx.html) | [opus_mt_fr_hr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_hr_xx.html) |
| [xx.fr.marian.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_fr_xx.html) | [opus_mt_ig_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_fr_xx.html) |
| [xx.sv.marian.translate_to.ht](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ht_sv_xx.html) | [opus_mt_ht_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ht_sv_xx.html) |
| [xx.de.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_de_xx.html) | [opus_mt_fr_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_de_xx.html) |
| [xx.fiu.marian.translate_to.fiu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fiu_fiu_xx.html) | [opus_mt_fiu_fiu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fiu_fiu_xx.html) |
| [xx.wls.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_wls_xx.html) | [opus_mt_fr_wls](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_wls_xx.html) |
| [xx.eo.marian.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_eo_xx.html) | [opus_mt_hu_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_eo_xx.html) |
| [xx.guw.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_guw_xx.html) | [opus_mt_fr_guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_guw_xx.html) |
| [xx.de.marian.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_de_xx.html) | [opus_mt_is_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_de_xx.html) |
| [xx.tvl.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tvl_xx.html) | [opus_mt_fr_tvl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tvl_xx.html) |
| [xx.zne.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_zne_xx.html) | [opus_mt_fr_zne](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_zne_xx.html) |
| [xx.ha.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ha_xx.html) | [opus_mt_fr_ha](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ha_xx.html) |
| [xx.fi.marian.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_fi_xx.html) | [opus_mt_guw_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_fi_xx.html) |
| [xx.es.marian.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_es_xx.html) | [opus_mt_is_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_es_xx.html) |
| [xx.sv.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_sv_xx.html) | [opus_mt_it_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_sv_xx.html) |
| [xx.uk.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_uk_xx.html) | [opus_mt_fr_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_uk_xx.html) |
| [xx.uk.marian.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_uk_xx.html) | [opus_mt_hu_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_uk_xx.html) |
| [xx.mt.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_mt_xx.html) | [opus_mt_fr_mt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_mt_xx.html) |
| [xx.gem.marian.translate_to.gem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gem_gem_xx.html) | [opus_mt_gem_gem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gem_gem_xx.html) |
| [xx.fr.marian.translate_to.fj](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fj_fr_xx.html) | [opus_mt_fj_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fj_fr_xx.html) |
| [xx.fi.marian.translate_to.gil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gil_fi_xx.html) | [opus_mt_gil_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gil_fi_xx.html) |
| [xx.fr.marian.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_fr_xx.html) | [opus_mt_hu_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_fr_xx.html) |
| [xx.bcl.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bcl_xx.html) | [opus_mt_fr_bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bcl_xx.html) |
| [xx.gmq.marian.translate_to.gmq](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gmq_gmq_xx.html) | [opus_mt_gmq_gmq](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gmq_gmq_xx.html) |
| [xx.kg.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_kg_xx.html) | [opus_mt_fr_kg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_kg_xx.html) |
| [xx.sn.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sn_xx.html) | [opus_mt_fr_sn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sn_xx.html) |
| [xx.bg.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bg_xx.html) | [opus_mt_fr_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bg_xx.html) |
| [xx.fr.marian.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_fr_xx.html) | [opus_mt_guw_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_fr_xx.html) |
| [xx.ts.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ts_xx.html) | [opus_mt_fr_ts](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ts_xx.html) |
| [xx.pis.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pis_xx.html) | [opus_mt_fr_pis](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pis_xx.html) |
| [xx.bi.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bi_xx.html) | [opus_mt_fr_bi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bi_xx.html) |
| [xx.ln.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ln_xx.html) | [opus_mt_fr_ln](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ln_xx.html) |
| [xx.de.marian.translate_to.hil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hil_de_xx.html) | [opus_mt_hil_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hil_de_xx.html) |
| [xx.nso.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_nso_xx.html) | [opus_mt_fr_nso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_nso_xx.html) |
| [xx.es.marian.translate_to.iso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_iso_es_xx.html) | [opus_mt_iso_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_iso_es_xx.html) |
| [xx.crs.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_crs_xx.html) | [opus_mt_fr_crs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_crs_xx.html) |
| [xx.niu.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_niu_xx.html) | [opus_mt_fr_niu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_niu_xx.html) |
| [xx.fr.marian.translate_to.ht](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ht_fr_xx.html) | [opus_mt_ht_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ht_fr_xx.html) |
| [xx.fi.marian.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_fi_xx.html) | [opus_mt_he_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_fi_xx.html) |
| [xx.gmw.marian.translate_to.gmw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gmw_gmw_xx.html) | [opus_mt_gmw_gmw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gmw_gmw_xx.html) |
| [xx.fr.marian.translate_to.hr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hr_fr_xx.html) | [opus_mt_hr_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hr_fr_xx.html) |
| [xx.sg.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sg_xx.html) | [opus_mt_fr_sg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sg_xx.html) |
| [xx.pon.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pon_xx.html) | [opus_mt_fr_pon](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pon_xx.html) |
| [xx.fi.marian.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_fi_xx.html) | [opus_mt_gaa_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_fi_xx.html) |
| [xx.pag.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pag_xx.html) | [opus_mt_fr_pag](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_pag_xx.html) |
| [xx.fi.marian.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_fi_xx.html) | [opus_mt_is_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_fi_xx.html) |
| [xx.sk.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sk_xx.html) | [opus_mt_fr_sk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sk_xx.html) |
| [xx.yap.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_yap_xx.html) | [opus_mt_fr_yap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_yap_xx.html) |
| [xx.es.marian.translate_to.ha](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ha_es_xx.html) | [opus_mt_ha_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ha_es_xx.html) |
| [xx.no.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_no_xx.html) | [opus_mt_fr_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_no_xx.html) |
| [xx.ine.marian.translate_to.ine](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ine_ine_xx.html) | [opus_mt_ine_ine](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ine_ine_xx.html) |
| [xx.fr.marian.translate_to.id](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_id_fr_xx.html) | [opus_mt_id_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_id_fr_xx.html) |
| [xx.bzs.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bzs_xx.html) | [opus_mt_fr_bzs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bzs_xx.html) |
| [xx.he.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_he_fr_xx.html) | [opus_tatoeba_he_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_he_fr_xx.html) |
| [xx.sv.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sv_xx.html) | [opus_mt_fr_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_sv_xx.html) |
| [xx.uk.marian.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_uk_xx.html) | [opus_mt_he_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_uk_xx.html) |
| [xx.fr.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_fr_xx.html) | [opus_mt_it_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_fr_xx.html) |
| [xx.fi.marian.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_fi_xx.html) | [opus_mt_ig_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_fi_xx.html) |
| [xx.vi.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_vi_xx.html) | [opus_mt_fr_vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_vi_xx.html) |
| [xx.fi.marian.translate_to.fse](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fse_fi_xx.html) | [opus_mt_fse_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fse_fi_xx.html) |
| [xx.es.marian.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_es_xx.html) | [opus_mt_guw_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_es_xx.html) |
| [xx.tll.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tll_xx.html) | [opus_mt_fr_tll](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tll_xx.html) |
| [xx.lua.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lua_xx.html) | [opus_mt_fr_lua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lua_xx.html) |
| [xx.yap.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_yap_xx.html) | [opus_mt_fi_yap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_yap_xx.html) |
| [xx.es.marian.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_es_xx.html) | [opus_mt_gaa_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gaa_es_xx.html) |
| [xx.sv.marian.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_sv_xx.html) | [opus_mt_ig_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ig_sv_xx.html) |
| [xx.ht.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ht_xx.html) | [opus_mt_fr_ht](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ht_xx.html) |
| [xx.el.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_el_xx.html) | [opus_mt_fr_el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_el_xx.html) |
| [xx.inc.marian.translate_to.inc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_inc_inc_xx.html) | [opus_mt_inc_inc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_inc_inc_xx.html) |
| [xx.swc.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_swc_xx.html) | [opus_mt_fr_swc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_swc_xx.html) |
| [xx.ar.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ar_xx.html) | [opus_mt_fr_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ar_xx.html) |
| [xx.es.marian.translate_to.ilo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ilo_es_xx.html) | [opus_mt_ilo_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ilo_es_xx.html) |
| [xx.fi.marian.translate_to.hr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hr_fi_xx.html) | [opus_mt_hr_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hr_fi_xx.html) |
| [xx.tpi.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tpi_xx.html) | [opus_mt_fr_tpi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tpi_xx.html) |
| [xx.ve.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ve_xx.html) | [opus_mt_fr_ve](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ve_xx.html) |
| [xx.sv.marian.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_sv_xx.html) | [opus_mt_guw_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_sv_xx.html) |
| [xx.sv.marian.translate_to.iso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_iso_sv_xx.html) | [opus_mt_iso_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_iso_sv_xx.html) |
| [xx.sv.marian.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_sv_xx.html) | [opus_mt_is_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_sv_xx.html) |
| [xx.tum.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tum_xx.html) | [opus_mt_fr_tum](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tum_xx.html) |
| [xx.es.marian.translate_to.ht](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ht_es_xx.html) | [opus_mt_ht_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ht_es_xx.html) |
| [xx.ho.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ho_xx.html) | [opus_mt_fr_ho](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ho_xx.html) |
| [xx.efi.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_efi_xx.html) | [opus_mt_fr_efi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_efi_xx.html) |
| [xx.es.marian.translate_to.gl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gl_es_xx.html) | [opus_mt_gl_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gl_es_xx.html) |
| [xx.ru.marian.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_ru_xx.html) | [opus_mt_he_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_ru_xx.html) |
| [xx.fi.marian.translate_to.hil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hil_fi_xx.html) | [opus_mt_hil_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hil_fi_xx.html) |
| [xx.eo.marian.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_eo_xx.html) | [opus_mt_he_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_eo_xx.html) |
| [xx.lu.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lu_xx.html) | [opus_mt_fr_lu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_lu_xx.html) |
| [xx.sv.marian.translate_to.ha](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ha_sv_xx.html) | [opus_mt_ha_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ha_sv_xx.html) |
| [xx.rnd.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_rnd_xx.html) | [opus_mt_fr_rnd](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_rnd_xx.html) |
| [xx.st.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_st_xx.html) | [opus_mt_fr_st](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_st_xx.html) |
| [xx.tl.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tl_xx.html) | [opus_mt_fr_tl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_tl_xx.html) |
| [xx.bem.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bem_xx.html) | [opus_mt_fr_bem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_bem_xx.html) |
| [xx.eo.marian.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_eo_xx.html) | [opus_mt_is_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_is_eo_xx.html) |
| [xx.is.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_is_xx.html) | [opus_mt_it_is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_is_xx.html) |
| [xx.hu.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_hu_xx.html) | [opus_mt_fr_hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_hu_xx.html) |
| [xx.yo.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_yo_xx.html) | [opus_mt_fi_yo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fi_yo_xx.html) |
| [xx.iso.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_iso_xx.html) | [opus_mt_fr_iso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_iso_xx.html) |
| [xx.de.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_de_xx.html) | [opus_mt_it_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_de_xx.html) |
| [xx.ty.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ty_xx.html) | [opus_mt_fr_ty](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ty_xx.html) |
| [xx.hil.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_hil_xx.html) | [opus_mt_fr_hil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_hil_xx.html) |
| [xx.eo.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_eo_xx.html) | [opus_mt_it_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_eo_xx.html) |
| [xx.sv.marian.translate_to.hr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hr_sv_xx.html) | [opus_mt_hr_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hr_sv_xx.html) |
| [xx.ber.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ber_xx.html) | [opus_mt_fr_ber](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ber_xx.html) |
| [xx.de.marian.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_de_xx.html) | [opus_mt_guw_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_guw_de_xx.html) |
| [xx.fi.marian.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_fi_xx.html) | [opus_mt_hu_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_fi_xx.html) |
| [xx.es.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_es_xx.html) | [opus_mt_it_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_es_xx.html) |
| [xx.de.marian.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_de_xx.html) | [opus_mt_hu_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hu_de_xx.html) |
| [xx.fj.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_fj_xx.html) | [opus_mt_fr_fj](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_fj_xx.html) |
| [xx.sv.marian.translate_to.id](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_id_sv_xx.html) | [opus_mt_id_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_id_sv_xx.html) |
| [xx.xh.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_xh_xx.html) | [opus_mt_fr_xh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_xh_xx.html) |
| [xx.yo.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_yo_xx.html) | [opus_mt_fr_yo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_yo_xx.html) |
| [xx.ca.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ca_xx.html) | [opus_mt_fr_ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ca_xx.html) |
| [xx.es.marian.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_es_xx.html) | [opus_mt_he_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_es_xx.html) |
| [xx.de.marian.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_de_xx.html) | [opus_mt_he_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_he_de_xx.html) |
| [xx.pt.marian.translate_to.gl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gl_pt_xx.html) | [opus_mt_gl_pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_gl_pt_xx.html) |
| [xx.ru.marian.translate_to.hy](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hy_ru_xx.html) | [opus_mt_hy_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_hy_ru_xx.html) |
| [xx.mos.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_mos_xx.html) | [opus_mt_fr_mos](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_mos_xx.html) |
| [xx.ceb.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ceb_xx.html) | [opus_mt_fr_ceb](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_fr_ceb_xx.html) |
| [xx.sh.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_sh_xx.html) | [opus_mt_ja_sh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_sh_xx.html) |
| [xx.bg.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_bg_xx.html) | [opus_mt_ja_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_bg_xx.html) |
| [xx.sv.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_sv_xx.html) | [opus_mt_ja_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_sv_xx.html) |
| [xx.ru.marian.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_ru_xx.html) | [opus_mt_lv_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_ru_xx.html) |
| [xx.fr.marian.translate_to.ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ms_fr_xx.html) | [opus_mt_ms_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ms_fr_xx.html) |
| [xx.sv.marian.translate_to.mt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mt_sv_xx.html) | [opus_mt_mt_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mt_sv_xx.html) |
| [xx.da.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_da_xx.html) | [opus_mt_ja_da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_da_xx.html) |
| [xx.de.marian.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_de_xx.html) | [opus_mt_niu_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_de_xx.html) |
| [xx.es.marian.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_es_xx.html) | [opus_mt_niu_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_es_xx.html) |
| [xx.sv.marian.translate_to.lus](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lus_sv_xx.html) | [opus_mt_lus_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lus_sv_xx.html) |
| [xx.sv.marian.translate_to.lg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lg_sv_xx.html) | [opus_mt_lg_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lg_sv_xx.html) |
| [xx.sv.marian.translate_to.pon](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pon_sv_xx.html) | [opus_mt_pon_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pon_sv_xx.html) |
| [xx.ru.marian.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_ru_xx.html) | [opus_mt_lt_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_ru_xx.html) |
| [xx.fi.marian.translate_to.lg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lg_fi_xx.html) | [opus_mt_lg_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lg_fi_xx.html) |
| [xx.sv.marian.translate_to.kg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kg_sv_xx.html) | [opus_mt_kg_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kg_sv_xx.html) |
| [xx.fr.marian.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_fr_xx.html) | [opus_mt_nl_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_fr_xx.html) |
| [xx.ms.marian.translate_to.ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ms_ms_xx.html) | [opus_mt_ms_ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ms_ms_xx.html) |
| [xx.es.marian.translate_to.lg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lg_es_xx.html) | [opus_mt_lg_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lg_es_xx.html) |
| [xx.fr.marian.translate_to.lu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lu_fr_xx.html) | [opus_mt_lu_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lu_fr_xx.html) |
| [xx.fr.marian.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_fr_xx.html) | [opus_mt_loz_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_fr_xx.html) |
| [xx.ca.marian.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_ca_xx.html) | [opus_mt_nl_ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_ca_xx.html) |
| [xx.sv.marian.translate_to.lue](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lue_sv_xx.html) | [opus_mt_lue_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lue_sv_xx.html) |
| [xx.vi.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_vi_xx.html) | [opus_mt_ja_vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_vi_xx.html) |
| [xx.fr.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_fr_xx.html) | [opus_mt_ja_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_fr_xx.html) |
| [xx.fi.marian.translate_to.pap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pap_fi_xx.html) | [opus_mt_pap_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pap_fi_xx.html) |
| [xx.pl.marian.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_pl_xx.html) | [opus_mt_lt_pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_pl_xx.html) |
| [xx.de.marian.translate_to.ny](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ny_de_xx.html) | [opus_mt_ny_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ny_de_xx.html) |
| [xx.fr.marian.translate_to.lue](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lue_fr_xx.html) | [opus_mt_lue_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lue_fr_xx.html) |
| [xx.gl.marian.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_gl_xx.html) | [opus_mt_pt_gl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_gl_xx.html) |
| [xx.fr.marian.translate_to.pap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pap_fr_xx.html) | [opus_mt_pap_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pap_fr_xx.html) |
| [xx.uk.marian.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_uk_xx.html) | [opus_mt_pl_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_uk_xx.html) |
| [xx.fi.marian.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_fi_xx.html) | [opus_mt_niu_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_fi_xx.html) |
| [xx.ar.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_ar_xx.html) | [opus_mt_ja_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_ar_xx.html) |
| [xx.es.marian.translate_to.mh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mh_es_xx.html) | [opus_mt_mh_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mh_es_xx.html) |
| [xx.ar.marian.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_ar_xx.html) | [opus_mt_pl_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_ar_xx.html) |
| [xx.de.marian.translate_to.pag](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pag_de_xx.html) | [opus_mt_pag_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pag_de_xx.html) |
| [xx.es.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_es_xx.html) | [opus_mt_no_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_es_xx.html) |
| [xx.es.marian.translate_to.mfs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mfs_es_xx.html) | [opus_mt_mfs_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mfs_es_xx.html) |
| [xx.fr.marian.translate_to.pis](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pis_fr_xx.html) | [opus_mt_pis_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pis_fr_xx.html) |
| [xx.eo.marian.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_eo_xx.html) | [opus_mt_pt_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_eo_xx.html) |
| [xx.de.marian.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_de_xx.html) | [opus_mt_lt_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_de_xx.html) |
| [xx.fr.marian.translate_to.ln](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ln_fr_xx.html) | [opus_mt_ln_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ln_fr_xx.html) |
| [xx.es.marian.translate_to.pag](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pag_es_xx.html) | [opus_mt_pag_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pag_es_xx.html) |
| [xx.fi.marian.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_fi_xx.html) | [opus_mt_nl_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_fi_xx.html) |
| [xx.vi.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_vi_xx.html) | [opus_mt_it_vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_it_vi_xx.html) |
| [xx.fi.marian.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_fi_xx.html) | [opus_mt_ko_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_fi_xx.html) |
| [xx.de.marian.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nso_de_xx.html) | [opus_mt_nso_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nso_de_xx.html) |
| [xx.fr.marian.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_fr_xx.html) | [opus_mt_niu_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_fr_xx.html) |
| [xx.ca.marian.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_ca_xx.html) | [opus_mt_pt_ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_ca_xx.html) |
| [xx.fr.marian.translate_to.kwy](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kwy_fr_xx.html) | [opus_mt_kwy_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kwy_fr_xx.html) |
| [xx.ru.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_ru_xx.html) | [opus_mt_no_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_ru_xx.html) |
| [xx.fi.marian.translate_to.pon](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pon_fi_xx.html) | [opus_mt_pon_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pon_fi_xx.html) |
| [xx.fi.marian.translate_to.lu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lu_fi_xx.html) | [opus_mt_lu_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lu_fi_xx.html) |
| [xx.es.marian.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_es_xx.html) | [opus_mt_ko_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_es_xx.html) |
| [xx.es.marian.translate_to.ny](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ny_es_xx.html) | [opus_mt_ny_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ny_es_xx.html) |
| [xx.itc.marian.translate_to.itc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_itc_itc_xx.html) | [opus_mt_itc_itc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_itc_itc_xx.html) |
| [xx.es.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_es_xx.html) | [opus_mt_ja_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_es_xx.html) |
| [xx.fr.marian.translate_to.mk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mk_fr_xx.html) | [opus_mt_mk_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mk_fr_xx.html) |
| [xx.it.marian.translate_to.ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ms_it_xx.html) | [opus_mt_ms_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ms_it_xx.html) |
| [xx.sv.marian.translate_to.lu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lu_sv_xx.html) | [opus_mt_lu_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lu_sv_xx.html) |
| [xx.fr.marian.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nso_fr_xx.html) | [opus_mt_nso_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nso_fr_xx.html) |
| [xx.uk.marian.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_uk_xx.html) | [opus_mt_pt_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_uk_xx.html) |
| [xx.no.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_no_xx.html) | [opus_mt_no_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_no_xx.html) |
| [xx.sv.marian.translate_to.lua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lua_sv_xx.html) | [opus_mt_lua_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lua_sv_xx.html) |
| [xx.es.marian.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_es_xx.html) | [opus_mt_pl_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_es_xx.html) |
| [xx.es.marian.translate_to.lu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lu_es_xx.html) | [opus_mt_lu_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lu_es_xx.html) |
| [xx.fr.marian.translate_to.lus](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lus_fr_xx.html) | [opus_mt_lus_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lus_fr_xx.html) |
| [xx.tr.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_tr_xx.html) | [opus_mt_ja_tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_tr_xx.html) |
| [xx.fi.marian.translate_to.pag](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pag_fi_xx.html) | [opus_mt_pag_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pag_fi_xx.html) |
| [xx.fr.marian.translate_to.kqn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kqn_fr_xx.html) | [opus_mt_kqn_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kqn_fr_xx.html) |
| [xx.fi.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_fi_xx.html) | [opus_mt_ja_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_fi_xx.html) |
| [xx.af.marian.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_af_xx.html) | [opus_mt_nl_af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_af_xx.html) |
| [xx.sv.marian.translate_to.pag](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pag_sv_xx.html) | [opus_mt_pag_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pag_sv_xx.html) |
| [xx.sv.marian.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_sv_xx.html) | [opus_mt_nl_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_sv_xx.html) |
| [xx.uk.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_uk_xx.html) | [opus_mt_no_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_uk_xx.html) |
| [xx.es.marian.translate_to.lua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lua_es_xx.html) | [opus_mt_lua_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lua_es_xx.html) |
| [xx.fi.marian.translate_to.mt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mt_fi_xx.html) | [opus_mt_mt_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mt_fi_xx.html) |
| [xx.eo.marian.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_eo_xx.html) | [opus_mt_lt_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_eo_xx.html) |
| [xx.de.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_de_xx.html) | [opus_mt_no_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_de_xx.html) |
| [xx.eo.marian.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_eo_xx.html) | [opus_mt_pl_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_eo_xx.html) |
| [xx.es.marian.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_es_xx.html) | [opus_mt_loz_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_es_xx.html) |
| [xx.ru.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_ru_xx.html) | [opus_mt_ja_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_ru_xx.html) |
| [xx.sv.marian.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_sv_xx.html) | [opus_mt_pl_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_sv_xx.html) |
| [xx.fi.marian.translate_to.mh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mh_fi_xx.html) | [opus_mt_mh_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mh_fi_xx.html) |
| [xx.hu.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_hu_xx.html) | [opus_mt_ja_hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_hu_xx.html) |
| [xx.fi.marian.translate_to.mk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mk_fi_xx.html) | [opus_mt_mk_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mk_fi_xx.html) |
| [xx.es.marian.translate_to.lue](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lue_es_xx.html) | [opus_mt_lue_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lue_es_xx.html) |
| [xx.sv.marian.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_sv_xx.html) | [opus_mt_lt_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_sv_xx.html) |
| [xx.fr.marian.translate_to.pon](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pon_fr_xx.html) | [opus_mt_pon_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pon_fr_xx.html) |
| [xx.es.marian.translate_to.pap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pap_es_xx.html) | [opus_mt_pap_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pap_es_xx.html) |
| [xx.es.marian.translate_to.ln](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ln_es_xx.html) | [opus_mt_ln_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ln_es_xx.html) |
| [xx.de.marian.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_de_xx.html) | [opus_mt_loz_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_de_xx.html) |
| [xx.ru.marian.translate_to.ka](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ka_ru_xx.html) | [opus_mt_ka_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ka_ru_xx.html) |
| [xx.sv.marian.translate_to.kwy](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kwy_sv_xx.html) | [opus_mt_kwy_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kwy_sv_xx.html) |
| [xx.fi.marian.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_fi_xx.html) | [opus_mt_lv_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_fi_xx.html) |
| [xx.pl.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_pl_xx.html) | [opus_mt_ja_pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_pl_xx.html) |
| [xx.hu.marian.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_hu_xx.html) | [opus_mt_ko_hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_hu_xx.html) |
| [xx.de.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_de_xx.html) | [opus_mt_ja_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_de_xx.html) |
| [xx.de.marian.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_de_xx.html) | [opus_mt_ko_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_de_xx.html) |
| [xx.es.marian.translate_to.kg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kg_es_xx.html) | [opus_mt_kg_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kg_es_xx.html) |
| [xx.de.marian.translate_to.pap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pap_de_xx.html) | [opus_mt_pap_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pap_de_xx.html) |
| [xx.fi.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_fi_xx.html) | [opus_mt_no_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_fi_xx.html) |
| [xx.fi.marian.translate_to.lue](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lue_fi_xx.html) | [opus_mt_lue_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lue_fi_xx.html) |
| [xx.no.marian.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_no_xx.html) | [opus_mt_pl_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_no_xx.html) |
| [xx.fr.marian.translate_to.mt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mt_fr_xx.html) | [opus_mt_mt_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mt_fr_xx.html) |
| [xx.es.marian.translate_to.mg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mg_es_xx.html) | [opus_mt_mg_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mg_es_xx.html) |
| [xx.es.marian.translate_to.pis](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pis_es_xx.html) | [opus_mt_pis_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pis_es_xx.html) |
| [xx.fr.marian.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_fr_xx.html) | [opus_mt_pl_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_fr_xx.html) |
| [xx.sv.marian.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_sv_xx.html) | [opus_mt_ko_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_sv_xx.html) |
| [xx.sv.marian.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_sv_xx.html) | [opus_mt_loz_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_sv_xx.html) |
| [xx.fi.marian.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_fi_xx.html) | [opus_mt_loz_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_loz_fi_xx.html) |
| [xx.pl.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_pl_xx.html) | [opus_mt_no_pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_pl_xx.html) |
| [xx.nl.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_nl_xx.html) | [opus_mt_ja_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_nl_xx.html) |
| [xx.de.marian.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_de_xx.html) | [opus_mt_pl_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_de_xx.html) |
| [xx.lt.marian.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_lt_xx.html) | [opus_mt_pl_lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pl_lt_xx.html) |
| [xx.ru.marian.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_ru_xx.html) | [opus_mt_ko_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_ru_xx.html) |
| [xx.fr.marian.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_fr_xx.html) | [opus_mt_lv_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_fr_xx.html) |
| [xx.he.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_he_xx.html) | [opus_mt_ja_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_he_xx.html) |
| [xx.sv.marian.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_sv_xx.html) | [opus_mt_niu_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_niu_sv_xx.html) |
| [xx.de.marian.translate_to.ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ms_de_xx.html) | [opus_mt_ms_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ms_de_xx.html) |
| [xx.es.marian.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_es_xx.html) | [opus_mt_lt_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_es_xx.html) |
| [xx.sv.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_sv_xx.html) | [opus_mt_no_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_sv_xx.html) |
| [xx.nl.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_nl_xx.html) | [opus_mt_no_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_nl_xx.html) |
| [xx.fi.marian.translate_to.lua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lua_fi_xx.html) | [opus_mt_lua_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lua_fi_xx.html) |
| [xx.fr.marian.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_fr_xx.html) | [opus_mt_lt_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_fr_xx.html) |
| [xx.ms.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_ms_xx.html) | [opus_mt_ja_ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_ms_xx.html) |
| [xx.es.marian.translate_to.kqn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kqn_es_xx.html) | [opus_mt_kqn_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kqn_es_xx.html) |
| [xx.fr.marian.translate_to.lg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lg_fr_xx.html) | [opus_mt_lg_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lg_fr_xx.html) |
| [xx.es.marian.translate_to.mk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mk_es_xx.html) | [opus_mt_mk_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mk_es_xx.html) |
| [xx.da.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_da_xx.html) | [opus_mt_no_da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_no_da_xx.html) |
| [xx.it.marian.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_it_xx.html) | [opus_mt_lt_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_it_xx.html) |
| [xx.es.marian.translate_to.prl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_prl_es_xx.html) | [opus_mt_prl_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_prl_es_xx.html) |
| [xx.fr.marian.translate_to.lua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lua_fr_xx.html) | [opus_mt_lua_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lua_fr_xx.html) |
| [xx.es.marian.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nso_es_xx.html) | [opus_mt_nso_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nso_es_xx.html) |
| [xx.sv.marian.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_sv_xx.html) | [opus_mt_lv_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_sv_xx.html) |
| [xx.fi.marian.translate_to.pis](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pis_fi_xx.html) | [opus_mt_pis_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pis_fi_xx.html) |
| [xx.es.marian.translate_to.pon](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pon_es_xx.html) | [opus_mt_pon_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pon_es_xx.html) |
| [xx.fr.marian.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_fr_xx.html) | [opus_mt_ko_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ko_fr_xx.html) |
| [xx.de.marian.translate_to.ln](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ln_de_xx.html) | [opus_mt_ln_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ln_de_xx.html) |
| [xx.uk.marian.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_uk_xx.html) | [opus_mt_nl_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_uk_xx.html) |
| [xx.eo.marian.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_eo_xx.html) | [opus_mt_nl_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_eo_xx.html) |
| [xx.es.marian.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_es_xx.html) | [opus_mt_lv_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lv_es_xx.html) |
| [xx.tr.marian.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_tr_xx.html) | [opus_mt_lt_tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lt_tr_xx.html) |
| [xx.es.marian.translate_to.mt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mt_es_xx.html) | [opus_mt_mt_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_mt_es_xx.html) |
| [xx.fi.marian.translate_to.lus](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lus_fi_xx.html) | [opus_mt_lus_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_lus_fi_xx.html) |
| [xx.tl.marian.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_tl_xx.html) | [opus_mt_pt_tl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pt_tl_xx.html) |
| [xx.no.marian.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_no_xx.html) | [opus_mt_nl_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nl_no_xx.html) |
| [xx.sv.marian.translate_to.kqn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kqn_sv_xx.html) | [opus_mt_kqn_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kqn_sv_xx.html) |
| [xx.pt.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_pt_xx.html) | [opus_mt_ja_pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ja_pt_xx.html) |
| [xx.fi.marian.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nso_fi_xx.html) | [opus_mt_nso_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_nso_fi_xx.html) |
| [xx.fr.marian.translate_to.kg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kg_fr_xx.html) | [opus_mt_kg_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_kg_fr_xx.html) |
| [xx.sv.marian.translate_to.pis](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pis_sv_xx.html) | [opus_mt_pis_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_pis_sv_xx.html) |
| [xx.is.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_is_xx.html) | [opus_mt_sv_is](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_is_xx.html) |
| [xx.sla.marian.translate_to.sla](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sla_sla_xx.html) | [opus_mt_sla_sla](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sla_sla_xx.html) |
| [xx.sv.marian.translate_to.srn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_srn_sv_xx.html) | [opus_mt_srn_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_srn_sv_xx.html) |
| [xx.niu.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_niu_xx.html) | [opus_mt_sv_niu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_niu_xx.html) |
| [xx.to.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_to_xx.html) | [opus_mt_sv_to](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_to_xx.html) |
| [xx.guw.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_guw_xx.html) | [opus_mt_sv_guw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_guw_xx.html) |
| [xx.sn.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sn_xx.html) | [opus_mt_sv_sn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sn_xx.html) |
| [xx.sv.marian.translate_to.rnd](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rnd_sv_xx.html) | [opus_mt_rnd_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rnd_sv_xx.html) |
| [xx.tum.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tum_xx.html) | [opus_mt_sv_tum](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tum_xx.html) |
| [xx.mos.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_mos_xx.html) | [opus_mt_sv_mos](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_mos_xx.html) |
| [xx.srn.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_srn_xx.html) | [opus_mt_sv_srn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_srn_xx.html) |
| [xx.ht.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ht_xx.html) | [opus_mt_sv_ht](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ht_xx.html) |
| [xx.no.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_no_xx.html) | [opus_mt_ru_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_no_xx.html) |
| [xx.sl.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sl_xx.html) | [opus_mt_sv_sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sl_xx.html) |
| [xx.fr.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_fr_xx.html) | [opus_mt_sv_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_fr_xx.html) |
| [xx.uk.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_uk_xx.html) | [opus_mt_ru_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_uk_xx.html) |
| [xx.tiv.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tiv_xx.html) | [opus_mt_sv_tiv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tiv_xx.html) |
| [xx.es.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_es_xx.html) | [opus_mt_ru_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_es_xx.html) |
| [xx.pag.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_pag_xx.html) | [opus_mt_sv_pag](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_pag_xx.html) |
| [xx.gaa.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_gaa_xx.html) | [opus_mt_sv_gaa](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_gaa_xx.html) |
| [xx.kqn.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_kqn_xx.html) | [opus_mt_sv_kqn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_kqn_xx.html) |
| [xx.fr.marian.translate_to.sg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sg_fr_xx.html) | [opus_mt_sg_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sg_fr_xx.html) |
| [xx.st.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_st_xx.html) | [opus_mt_sv_st](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_st_xx.html) |
| [xx.ase.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ase_xx.html) | [opus_mt_sv_ase](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ase_xx.html) |
| [xx.es.marian.translate_to.rn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rn_es_xx.html) | [opus_mt_rn_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rn_es_xx.html) |
| [xx.ru.marian.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_ru_xx.html) | [opus_mt_sl_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_ru_xx.html) |
| [xx.lu.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lu_xx.html) | [opus_mt_sv_lu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lu_xx.html) |
| [xx.eu.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_eu_xx.html) | [opus_mt_ru_eu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_eu_xx.html) |
| [xx.no.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_no_xx.html) | [opus_mt_sv_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_no_xx.html) |
| [xx.sq.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sq_xx.html) | [opus_mt_sv_sq](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sq_xx.html) |
| [xx.da.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_da_xx.html) | [opus_mt_ru_da](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_da_xx.html) |
| [xx.ny.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ny_xx.html) | [opus_mt_sv_ny](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ny_xx.html) |
| [xx.kg.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_kg_xx.html) | [opus_mt_sv_kg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_kg_xx.html) |
| [xx.pis.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_pis_xx.html) | [opus_mt_sv_pis](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_pis_xx.html) |
| [xx.sv.marian.translate_to.sk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sk_sv_xx.html) | [opus_mt_sk_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sk_sv_xx.html) |
| [xx.lus.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lus_xx.html) | [opus_mt_sv_lus](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lus_xx.html) |
| [xx.fi.marian.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_fi_xx.html) | [opus_mt_sl_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_fi_xx.html) |
| [xx.tn.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tn_xx.html) | [opus_mt_sv_tn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tn_xx.html) |
| [xx.fr.marian.translate_to.srn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_srn_fr_xx.html) | [opus_mt_srn_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_srn_fr_xx.html) |
| [xx.lv.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lv_xx.html) | [opus_mt_sv_lv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lv_xx.html) |
| [xx.uk.marian.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_uk_xx.html) | [opus_mt_sl_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_uk_xx.html) |
| [xx.sg.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sg_xx.html) | [opus_mt_sv_sg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sg_xx.html) |
| [xx.he.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_he_xx.html) | [opus_mt_sv_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_he_xx.html) |
| [xx.eo.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_eo_xx.html) | [opus_mt_ru_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_eo_xx.html) |
| [xx.fr.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_fr_xx.html) | [opus_mt_ru_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_fr_xx.html) |
| [xx.lv.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_lv_xx.html) | [opus_mt_ru_lv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_lv_xx.html) |
| [xx.lua.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lua_xx.html) | [opus_mt_sv_lua](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lua_xx.html) |
| [xx.ar.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_ar_xx.html) | [opus_mt_ru_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_ar_xx.html) |
| [xx.tll.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tll_xx.html) | [opus_mt_sv_tll](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tll_xx.html) |
| [xx.lue.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lue_xx.html) | [opus_mt_sv_lue](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lue_xx.html) |
| [xx.bi.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bi_xx.html) | [opus_mt_sv_bi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bi_xx.html) |
| [xx.hu.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_hu_xx.html) | [opus_mt_sv_hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_hu_xx.html) |
| [xx.bzs.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bzs_xx.html) | [opus_mt_sv_bzs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bzs_xx.html) |
| [xx.ru.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ru_xx.html) | [opus_mt_sv_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ru_xx.html) |
| [xx.eo.marian.translate_to.ro](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ro_eo_xx.html) | [opus_mt_ro_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ro_eo_xx.html) |
| [xx.es.marian.translate_to.st](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_st_es_xx.html) | [opus_mt_st_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_st_es_xx.html) |
| [xx.mt.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_mt_xx.html) | [opus_mt_sv_mt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_mt_xx.html) |
| [xx.af.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_af_xx.html) | [opus_mt_sv_af](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_af_xx.html) |
| [xx.ts.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ts_xx.html) | [opus_mt_sv_ts](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ts_xx.html) |
| [xx.af.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_af_ru_xx.html) | [opus_tatoeba_af_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_af_ru_xx.html) |
| [xx.efi.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_efi_xx.html) | [opus_mt_sv_efi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_efi_xx.html) |
| [xx.es.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_es_xx.html) | [opus_mt_sv_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_es_xx.html) |
| [xx.fi.marian.translate_to.sk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sk_fi_xx.html) | [opus_mt_sk_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sk_fi_xx.html) |
| [xx.fr.marian.translate_to.rw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rw_fr_xx.html) | [opus_mt_rw_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rw_fr_xx.html) |
| [xx.sv.marian.translate_to.run](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_run_sv_xx.html) | [opus_mt_run_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_run_sv_xx.html) |
| [xx.th.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_th_xx.html) | [opus_mt_sv_th](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_th_xx.html) |
| [xx.ln.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ln_xx.html) | [opus_mt_sv_ln](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ln_xx.html) |
| [xx.es.marian.translate_to.sk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sk_es_xx.html) | [opus_mt_sk_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sk_es_xx.html) |
| [xx.lt.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_lt_xx.html) | [opus_mt_ru_lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_lt_xx.html) |
| [xx.mfe.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_mfe_xx.html) | [opus_mt_sv_mfe](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_mfe_xx.html) |
| [xx.cs.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_cs_xx.html) | [opus_mt_sv_cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_cs_xx.html) |
| [xx.vi.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_vi_xx.html) | [opus_mt_ru_vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_vi_xx.html) |
| [xx.ee.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ee_xx.html) | [opus_mt_sv_ee](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ee_xx.html) |
| [xx.bg.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_bg_xx.html) | [opus_mt_ru_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_bg_xx.html) |
| [xx.nso.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_nso_xx.html) | [opus_mt_sv_nso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_nso_xx.html) |
| [xx.mh.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_mh_xx.html) | [opus_mt_sv_mh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_mh_xx.html) |
| [xx.iso.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_iso_xx.html) | [opus_mt_sv_iso](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_iso_xx.html) |
| [xx.fi.marian.translate_to.st](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_st_fi_xx.html) | [opus_mt_st_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_st_fi_xx.html) |
| [xx.bg.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bg_xx.html) | [opus_mt_sv_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bg_xx.html) |
| [xx.sv.marian.translate_to.sq](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sq_sv_xx.html) | [opus_mt_sq_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sq_sv_xx.html) |
| [xx.sv.marian.translate_to.sn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sn_sv_xx.html) | [opus_mt_sn_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sn_sv_xx.html) |
| [xx.de.marian.translate_to.rn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rn_de_xx.html) | [opus_mt_rn_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rn_de_xx.html) |
| [xx.pon.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_pon_xx.html) | [opus_mt_sv_pon](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_pon_xx.html) |
| [xx.ha.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ha_xx.html) | [opus_mt_sv_ha](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ha_xx.html) |
| [xx.fi.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_fi_xx.html) | [opus_mt_ru_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_fi_xx.html) |
| [xx.sk.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sk_xx.html) | [opus_mt_sv_sk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sk_xx.html) |
| [xx.es.marian.translate_to.run](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_run_es_xx.html) | [opus_mt_run_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_run_es_xx.html) |
| [xx.et.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_et_xx.html) | [opus_mt_ru_et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_et_xx.html) |
| [xx.swc.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_swc_xx.html) | [opus_mt_sv_swc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_swc_xx.html) |
| [xx.hil.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_hil_xx.html) | [opus_mt_sv_hil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_hil_xx.html) |
| [xx.ro.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ro_xx.html) | [opus_mt_sv_ro](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ro_xx.html) |
| [xx.fr.marian.translate_to.rnd](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rnd_fr_xx.html) | [opus_mt_rnd_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rnd_fr_xx.html) |
| [xx.kwy.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_kwy_xx.html) | [opus_mt_sv_kwy](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_kwy_xx.html) |
| [xx.uk.marian.translate_to.sh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sh_uk_xx.html) | [opus_mt_sh_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sh_uk_xx.html) |
| [xx.sm.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sm_xx.html) | [opus_mt_sv_sm](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sm_xx.html) |
| [xx.sv.marian.translate_to.rw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rw_sv_xx.html) | [opus_mt_rw_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rw_sv_xx.html) |
| [xx.et.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_et_xx.html) | [opus_mt_sv_et](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_et_xx.html) |
| [xx.eo.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_eo_xx.html) | [opus_mt_sv_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_eo_xx.html) |
| [xx.rnd.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_rnd_xx.html) | [opus_mt_sv_rnd](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_rnd_xx.html) |
| [xx.eo.marian.translate_to.sh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sh_eo_xx.html) | [opus_mt_sh_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sh_eo_xx.html) |
| [xx.ru.marian.translate_to.rn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rn_ru_xx.html) | [opus_mt_rn_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rn_ru_xx.html) |
| [xx.rw.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_rw_xx.html) | [opus_mt_sv_rw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_rw_xx.html) |
| [xx.fr.marian.translate_to.sn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sn_fr_xx.html) | [opus_mt_sn_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sn_fr_xx.html) |
| [xx.ig.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ig_xx.html) | [opus_mt_sv_ig](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ig_xx.html) |
| [xx.fj.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_fj_xx.html) | [opus_mt_sv_fj](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_fj_xx.html) |
| [xx.sl.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_sl_xx.html) | [opus_mt_ru_sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_sl_xx.html) |
| [xx.ho.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ho_xx.html) | [opus_mt_sv_ho](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ho_xx.html) |
| [xx.sv.marian.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_sv_xx.html) | [opus_mt_sl_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_sv_xx.html) |
| [xx.pap.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_pap_xx.html) | [opus_mt_sv_pap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_pap_xx.html) |
| [xx.fr.marian.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_fr_xx.html) | [opus_mt_sl_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_fr_xx.html) |
| [xx.es.marian.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_es_xx.html) | [opus_mt_sl_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sl_es_xx.html) |
| [xx.run.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_run_xx.html) | [opus_mt_sv_run](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_run_xx.html) |
| [xx.el.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_el_xx.html) | [opus_mt_sv_el](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_el_xx.html) |
| [xx.gil.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_gil_xx.html) | [opus_mt_sv_gil](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_gil_xx.html) |
| [xx.crs.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_crs_xx.html) | [opus_mt_sv_crs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_crs_xx.html) |
| [xx.fr.marian.translate_to.sk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sk_fr_xx.html) | [opus_mt_sk_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sk_fr_xx.html) |
| [xx.es.marian.translate_to.sq](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sq_es_xx.html) | [opus_mt_sq_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sq_es_xx.html) |
| [xx.sv.marian.translate_to.sg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sg_sv_xx.html) | [opus_mt_sg_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sg_sv_xx.html) |
| [xx.es.marian.translate_to.srn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_srn_es_xx.html) | [opus_mt_srn_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_srn_es_xx.html) |
| [xx.fr.marian.translate_to.ro](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ro_fr_xx.html) | [opus_mt_ro_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ro_fr_xx.html) |
| [xx.fr.marian.translate_to.rn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rn_fr_xx.html) | [opus_mt_rn_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rn_fr_xx.html) |
| [xx.fr.marian.translate_to.st](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_st_fr_xx.html) | [opus_mt_st_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_st_fr_xx.html) |
| [xx.es.marian.translate_to.rw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rw_es_xx.html) | [opus_mt_rw_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_rw_es_xx.html) |
| [xx.hr.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_hr_xx.html) | [opus_mt_sv_hr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_hr_xx.html) |
| [xx.es.marian.translate_to.sm](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sm_es_xx.html) | [opus_mt_sm_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sm_es_xx.html) |
| [xx.es.marian.translate_to.ssp](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ssp_es_xx.html) | [opus_mt_ssp_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ssp_es_xx.html) |
| [xx.nl.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_nl_xx.html) | [opus_mt_sv_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_nl_xx.html) |
| [xx.bem.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bem_xx.html) | [opus_mt_sv_bem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bem_xx.html) |
| [xx.sem.marian.translate_to.sem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sem_sem_xx.html) | [opus_mt_sem_sem](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sem_sem_xx.html) |
| [xx.sv.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sv_xx.html) | [opus_mt_sv_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_sv_xx.html) |
| [xx.sv.marian.translate_to.st](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_st_sv_xx.html) | [opus_mt_st_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_st_sv_xx.html) |
| [xx.lg.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lg_xx.html) | [opus_mt_sv_lg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_lg_xx.html) |
| [xx.bcl.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bcl_xx.html) | [opus_mt_sv_bcl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_bcl_xx.html) |
| [xx.toi.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_toi_xx.html) | [opus_mt_sv_toi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_toi_xx.html) |
| [xx.id.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_id_xx.html) | [opus_mt_sv_id](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_id_xx.html) |
| [xx.he.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_he_xx.html) | [opus_mt_ru_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_he_xx.html) |
| [xx.ceb.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ceb_xx.html) | [opus_mt_sv_ceb](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ceb_xx.html) |
| [xx.tw.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tw_xx.html) | [opus_mt_sv_tw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tw_xx.html) |
| [xx.chk.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_chk_xx.html) | [opus_mt_sv_chk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_chk_xx.html) |
| [xx.fr.marian.translate_to.sm](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sm_fr_xx.html) | [opus_mt_sm_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sm_fr_xx.html) |
| [xx.tvl.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tvl_xx.html) | [opus_mt_sv_tvl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tvl_xx.html) |
| [xx.es.marian.translate_to.sg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sg_es_xx.html) | [opus_mt_sg_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sg_es_xx.html) |
| [xx.ilo.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ilo_xx.html) | [opus_mt_sv_ilo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ilo_xx.html) |
| [xx.sv.marian.translate_to.ro](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ro_sv_xx.html) | [opus_mt_ro_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ro_sv_xx.html) |
| [xx.fi.marian.translate_to.sg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sg_fi_xx.html) | [opus_mt_sg_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sg_fi_xx.html) |
| [xx.hy.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_hy_xx.html) | [opus_mt_ru_hy](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_hy_xx.html) |
| [xx.fi.marian.translate_to.ro](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ro_fi_xx.html) | [opus_mt_ro_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ro_fi_xx.html) |
| [xx.tpi.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tpi_xx.html) | [opus_mt_sv_tpi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_tpi_xx.html) |
| [xx.fi.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_fi_xx.html) | [opus_mt_sv_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_fi_xx.html) |
| [xx.sv.marian.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_sv_xx.html) | [opus_mt_ru_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ru_sv_xx.html) |
| [xx.es.marian.translate_to.toi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_toi_es_xx.html) | [opus_mt_toi_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_toi_es_xx.html) |
| [xx.no.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_no_xx.html) | [opus_mt_uk_no](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_no_xx.html) |
| [xx.ar.marian.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_ar_xx.html) | [opus_mt_tr_ar](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_ar_xx.html) |
| [xx.he.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_he_xx.html) | [opus_mt_uk_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_he_xx.html) |
| [xx.sv.marian.translate_to.tvl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tvl_sv_xx.html) | [opus_mt_tvl_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tvl_sv_xx.html) |
| [xx.uk.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_uk_xx.html) | [opus_mt_sv_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_uk_xx.html) |
| [xx.fr.marian.translate_to.tvl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tvl_fr_xx.html) | [opus_mt_tvl_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tvl_fr_xx.html) |
| [xx.bg.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_bg_xx.html) | [opus_mt_uk_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_bg_xx.html) |
| [xx.fi.marian.translate_to.toi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_toi_fi_xx.html) | [opus_mt_toi_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_toi_fi_xx.html) |
| [xx.ca.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_ca_xx.html) | [opus_mt_uk_ca](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_ca_xx.html) |
| [xx.fr.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_fr_xx.html) | [opus_mt_uk_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_fr_xx.html) |
| [xx.eo.marian.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_eo_xx.html) | [opus_mt_tr_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_eo_xx.html) |
| [xx.uk.marian.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_uk_xx.html) | [opus_mt_tr_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_uk_xx.html) |
| [xx.es.marian.translate_to.tl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tl_es_xx.html) | [opus_mt_tl_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tl_es_xx.html) |
| [xx.es.marian.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_es_xx.html) | [opus_mt_tr_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_es_xx.html) |
| [xx.it.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_it_xx.html) | [opus_mt_uk_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_it_xx.html) |
| [xx.fi.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_fi_xx.html) | [opus_mt_uk_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_fi_xx.html) |
| [xx.lt.marian.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_lt_xx.html) | [opus_mt_tr_lt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_lt_xx.html) |
| [xx.es.marian.translate_to.swc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_swc_es_xx.html) | [opus_mt_swc_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_swc_es_xx.html) |
| [xx.umb.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_umb_xx.html) | [opus_mt_sv_umb](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_umb_xx.html) |
| [xx.sv.marian.translate_to.tw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tw_sv_xx.html) | [opus_mt_tw_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tw_sv_xx.html) |
| [xx.urj.marian.translate_to.urj](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_urj_urj_xx.html) | [opus_mt_urj_urj](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_urj_urj_xx.html) |
| [xx.yap.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_yap_xx.html) | [opus_mt_sv_yap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_yap_xx.html) |
| [xx.fr.marian.translate_to.ty](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ty_fr_xx.html) | [opus_mt_ty_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ty_fr_xx.html) |
| [xx.fr.marian.translate_to.swc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_swc_fr_xx.html) | [opus_mt_swc_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_swc_fr_xx.html) |
| [xx.pt.marian.translate_to.tl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tl_pt_xx.html) | [opus_mt_tl_pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tl_pt_xx.html) |
| [xx.tr.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_tr_xx.html) | [opus_mt_uk_tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_tr_xx.html) |
| [xx.sv.marian.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_sv_xx.html) | [opus_mt_tr_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_sv_xx.html) |
| [xx.fi.marian.translate_to.tvl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tvl_fi_xx.html) | [opus_mt_tvl_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tvl_fi_xx.html) |
| [xx.es.marian.translate_to.tn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tn_es_xx.html) | [opus_mt_tn_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tn_es_xx.html) |
| [xx.fi.marian.translate_to.swc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_swc_fi_xx.html) | [opus_mt_swc_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_swc_fi_xx.html) |
| [xx.fr.marian.translate_to.toi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_toi_fr_xx.html) | [opus_mt_toi_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_toi_fr_xx.html) |
| [xx.fi.marian.translate_to.ts](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ts_fi_xx.html) | [opus_mt_ts_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ts_fi_xx.html) |
| [xx.de.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_de_xx.html) | [opus_mt_uk_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_de_xx.html) |
| [xx.sv.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_sv_xx.html) | [opus_mt_uk_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_sv_xx.html) |
| [xx.fi.marian.translate_to.tw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tw_fi_xx.html) | [opus_mt_tw_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tw_fi_xx.html) |
| [xx.sv.marian.translate_to.to](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_to_sv_xx.html) | [opus_mt_to_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_to_sv_xx.html) |
| [xx.sv.marian.translate_to.tll](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tll_sv_xx.html) | [opus_mt_tll_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tll_sv_xx.html) |
| [xx.fr.marian.translate_to.th](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_th_fr_xx.html) | [opus_mt_th_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_th_fr_xx.html) |
| [xx.es.marian.translate_to.ty](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ty_es_xx.html) | [opus_mt_ty_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ty_es_xx.html) |
| [xx.fr.marian.translate_to.tw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tw_fr_xx.html) | [opus_mt_tw_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tw_fr_xx.html) |
| [xx.fr.marian.translate_to.to](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_to_fr_xx.html) | [opus_mt_to_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_to_fr_xx.html) |
| [xx.sl.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_sl_xx.html) | [opus_mt_uk_sl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_sl_xx.html) |
| [xx.xh.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_xh_xx.html) | [opus_mt_sv_xh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_xh_xx.html) |
| [xx.war.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_war_xx.html) | [opus_mt_sv_war](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_war_xx.html) |
| [xx.hu.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_hu_xx.html) | [opus_mt_uk_hu](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_hu_xx.html) |
| [xx.ru.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_ru_xx.html) | [opus_mt_uk_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_ru_xx.html) |
| [xx.sv.marian.translate_to.tn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tn_sv_xx.html) | [opus_mt_tn_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tn_sv_xx.html) |
| [xx.fr.marian.translate_to.tum](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tum_fr_xx.html) | [opus_mt_tum_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tum_fr_xx.html) |
| [xx.sv.marian.translate_to.toi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_toi_sv_xx.html) | [opus_mt_toi_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_toi_sv_xx.html) |
| [xx.sv.marian.translate_to.ty](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ty_sv_xx.html) | [opus_mt_ty_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ty_sv_xx.html) |
| [xx.fr.marian.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_fr_xx.html) | [opus_mt_tr_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_fr_xx.html) |
| [xx.fr.marian.translate_to.tn](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tn_fr_xx.html) | [opus_mt_tn_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tn_fr_xx.html) |
| [xx.cs.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_cs_xx.html) | [opus_mt_uk_cs](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_cs_xx.html) |
| [xx.fr.marian.translate_to.ts](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ts_fr_xx.html) | [opus_mt_ts_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ts_fr_xx.html) |
| [xx.sv.marian.translate_to.swc](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_swc_sv_xx.html) | [opus_mt_swc_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_swc_sv_xx.html) |
| [xx.es.marian.translate_to.to](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_to_es_xx.html) | [opus_mt_to_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_to_es_xx.html) |
| [xx.es.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_es_xx.html) | [opus_mt_uk_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_es_xx.html) |
| [xx.nl.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_nl_xx.html) | [opus_mt_uk_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_nl_xx.html) |
| [xx.zne.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_zne_xx.html) | [opus_mt_sv_zne](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_zne_xx.html) |
| [xx.es.marian.translate_to.tvl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tvl_es_xx.html) | [opus_mt_tvl_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tvl_es_xx.html) |
| [xx.pt.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_pt_xx.html) | [opus_mt_uk_pt](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_pt_xx.html) |
| [xx.fr.marian.translate_to.tiv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tiv_fr_xx.html) | [opus_mt_tiv_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tiv_fr_xx.html) |
| [xx.fr.marian.translate_to.tll](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tll_fr_xx.html) | [opus_mt_tll_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tll_fr_xx.html) |
| [xx.sh.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_sh_xx.html) | [opus_mt_uk_sh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_sh_xx.html) |
| [xx.wls.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_wls_xx.html) | [opus_mt_sv_wls](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_wls_xx.html) |
| [xx.ve.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ve_xx.html) | [opus_mt_sv_ve](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ve_xx.html) |
| [xx.es.marian.translate_to.tum](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tum_es_xx.html) | [opus_mt_tum_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tum_es_xx.html) |
| [xx.fi.marian.translate_to.tll](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tll_fi_xx.html) | [opus_mt_tll_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tll_fi_xx.html) |
| [xx.es.marian.translate_to.tw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tw_es_xx.html) | [opus_mt_tw_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tw_es_xx.html) |
| [xx.sv.marian.translate_to.tiv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tiv_sv_xx.html) | [opus_mt_tiv_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tiv_sv_xx.html) |
| [xx.fi.marian.translate_to.ty](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ty_fi_xx.html) | [opus_mt_ty_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ty_fi_xx.html) |
| [xx.pl.marian.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_pl_xx.html) | [opus_mt_uk_pl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_uk_pl_xx.html) |
| [xx.sv.marian.translate_to.tpi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tpi_sv_xx.html) | [opus_mt_tpi_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tpi_sv_xx.html) |
| [xx.az.marian.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_az_xx.html) | [opus_mt_tr_az](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tr_az_xx.html) |
| [xx.es.marian.translate_to.tll](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tll_es_xx.html) | [opus_mt_tll_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_tll_es_xx.html) |
| [xx.ty.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ty_xx.html) | [opus_mt_sv_ty](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_sv_ty_xx.html) |
| [xx.tzo.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_es_tzo_xx.html) | [opus_mt_es_tzo](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_es_tzo_xx.html) |
| [xx.sv.marian.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_crs_sv_xx.html) | [opus_mt_crs_sv](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_crs_sv_xx.html) |
| [xx.es.marian.translate_to.zai](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_zai_es_xx.html) | [opus_mt_zai_es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_zai_es_xx.html) |
| [xx.niu.marian.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_de_niu_xx.html) | [opus_mt_de_niu](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_de_niu_xx.html) |
| [xx.sv.marian.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_nso_sv_xx.html) | [opus_mt_nso_sv](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_nso_sv_xx.html) |
| [xx.fr.marian.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_bg_fr_xx.html) | [opus_mt_bg_fr](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_bg_fr_xx.html) |
| [xx.es.marian.translate_to.lus](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_lus_es_xx.html) | [opus_mt_lus_es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_lus_es_xx.html) |
| [xx.es.marian.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_nl_es_xx.html) | [opus_mt_nl_es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_nl_es_xx.html) |
| [xx.fr.marian.translate_to.yo](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_yo_fr_xx.html) | [opus_mt_yo_fr](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_yo_fr_xx.html) |
| [xx.sv.marian.translate_to.ilo](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ilo_sv_xx.html) | [opus_mt_ilo_sv](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ilo_sv_xx.html) |
| [xx.es.marian.translate_to.ts](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ts_es_xx.html) | [opus_mt_ts_es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ts_es_xx.html) |
| [xx.run.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fr_run_xx.html) | [opus_mt_fr_run](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fr_run_xx.html) |
| [xx.to.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_es_to_xx.html) | [opus_mt_es_to](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_es_to_xx.html) |
| [xx.ceb.marian.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fi_ceb_xx.html) | [opus_mt_fi_ceb](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fi_ceb_xx.html) |
| [xx.it.marian.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ja_it_xx.html) | [opus_mt_ja_it](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ja_it_xx.html) |
| [xx.es.marian.translate_to.sn](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_sn_es_xx.html) | [opus_mt_sn_es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_sn_es_xx.html) |
| [xx.yo.marian.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_sv_yo_xx.html) | [opus_mt_sv_yo](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_sv_yo_xx.html) |
| [xx.tr.marian.translate_to.az](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_az_tr_xx.html) | [opus_mt_az_tr](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_az_tr_xx.html) |
| [xx.fr.marian.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_no_fr_xx.html) | [opus_mt_no_fr](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_no_fr_xx.html) |
| [xx.tn.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fr_tn_xx.html) | [opus_mt_fr_tn](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fr_tn_xx.html) |
| [xx.id.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fr_id_xx.html) | [opus_mt_fr_id](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fr_id_xx.html) |
| [xx.de.marian.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ca_de_xx.html) | [opus_mt_ca_de](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ca_de_xx.html) |
| [xx.sv.marian.translate_to.tum](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_tum_sv_xx.html) | [opus_mt_tum_sv](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_tum_sv_xx.html) |
| [xx.ru.marian.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_da_ru_xx.html) | [opus_mt_da_ru](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_da_ru_xx.html) |
| [xx.de.marian.translate_to.tl](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_tl_de_xx.html) | [opus_mt_tl_de](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_tl_de_xx.html) |
| [xx.eo.marian.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fr_eo_xx.html) | [opus_mt_fr_eo](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_fr_eo_xx.html) |
| [xx.vi.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_vi_xx.html) | [opus_mt_zh_vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_vi_xx.html) |
| [xx.es.marian.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_es_xx.html) | [opus_mt_vi_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_es_xx.html) |
| [xx.es.marian.translate_to.mfe](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_mfe_es_xx.html) | [opus_mt_mfe_es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_mfe_es_xx.html) |
| [xx.fi.marian.translate_to.iso](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_iso_fi_xx.html) | [opus_mt_iso_fi](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_iso_fi_xx.html) |
| [xx.es.marian.translate_to.tzo](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_tzo_es_xx.html) | [opus_mt_tzo_es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_tzo_es_xx.html) |
| [xx.sn.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_es_sn_xx.html) | [opus_mt_es_sn](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_es_sn_xx.html) |
| [xx.es.marian.translate_to.xh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_xh_es_xx.html) | [opus_mt_xh_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_xh_es_xx.html) |
| [xx.sv.marian.translate_to.zne](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zne_sv_xx.html) | [opus_mt_zne_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zne_sv_xx.html) |
| [xx.sv.marian.translate_to.ts](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ts_sv_xx.html) | [opus_mt_ts_sv](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_ts_sv_xx.html) |
| [xx.it.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_it_xx.html) | [opus_mt_zh_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_it_xx.html) |
| [xx.uk.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_uk_xx.html) | [opus_mt_zh_uk](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_uk_xx.html) |
| [xx.fi.marian.translate_to.yo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yo_fi_xx.html) | [opus_mt_yo_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yo_fi_xx.html) |
| [xx.sv.marian.translate_to.war](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_war_sv_xx.html) | [opus_mt_war_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_war_sv_xx.html) |
| [xx.sv.marian.translate_to.yo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yo_sv_xx.html) | [opus_mt_yo_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yo_sv_xx.html) |
| [xx.tll.marian.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_es_tll_xx.html) | [opus_mt_es_tll](https://nlp.johnsnowlabs.com//2021/06/02/opus_mt_es_tll_xx.html) |
| [xx.nl.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_nl_xx.html) | [opus_mt_zh_nl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_nl_xx.html) |
| [xx.fr.marian.translate_to.wls](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_wls_fr_xx.html) | [opus_mt_wls_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_wls_fr_xx.html) |
| [xx.it.marian.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_it_xx.html) | [opus_mt_vi_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_it_xx.html) |
| [xx.bg.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_bg_xx.html) | [opus_mt_zh_bg](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_bg_xx.html) |
| [xx.sv.marian.translate_to.xh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_xh_sv_xx.html) | [opus_mt_xh_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_xh_sv_xx.html) |
| [xx.es.marian.translate_to.zne](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zne_es_xx.html) | [opus_mt_zne_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zne_es_xx.html) |
| [xx.zlw.marian.translate_to.zlw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zlw_zlw_xx.html) | [opus_mt_zlw_zlw](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zlw_zlw_xx.html) |
| [xx.sv.marian.translate_to.yap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yap_sv_xx.html) | [opus_mt_yap_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yap_sv_xx.html) |
| [xx.he.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_he_xx.html) | [opus_mt_zh_he](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_he_xx.html) |
| [xx.fr.marian.translate_to.xh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_xh_fr_xx.html) | [opus_mt_xh_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_xh_fr_xx.html) |
| [xx.fi.marian.translate_to.war](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_war_fi_xx.html) | [opus_mt_war_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_war_fi_xx.html) |
| [xx.sv.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_sv_xx.html) | [opus_mt_zh_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_sv_xx.html) |
| [xx.zls.marian.translate_to.zls](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zls_zls_xx.html) | [opus_mt_zls_zls](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zls_zls_xx.html) |
| [xx.fi.marian.translate_to.zne](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zne_fi_xx.html) | [opus_mt_zne_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zne_fi_xx.html) |
| [xx.es.marian.translate_to.ve](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ve_es_xx.html) | [opus_mt_ve_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_ve_es_xx.html) |
| [xx.de.marian.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_de_xx.html) | [opus_mt_vi_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_de_xx.html) |
| [xx.eo.marian.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_eo_xx.html) | [opus_mt_vi_eo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_eo_xx.html) |
| [xx.sv.marian.translate_to.wls](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_wls_sv_xx.html) | [opus_mt_wls_sv](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_wls_sv_xx.html) |
| [xx.es.marian.translate_to.war](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_war_es_xx.html) | [opus_mt_war_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_war_es_xx.html) |
| [xx.ru.marian.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_ru_xx.html) | [opus_mt_vi_ru](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_ru_xx.html) |
| [xx.ms.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_ms_xx.html) | [opus_mt_zh_ms](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_ms_xx.html) |
| [xx.fr.marian.translate_to.zne](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zne_fr_xx.html) | [opus_mt_zne_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zne_fr_xx.html) |
| [xx.fr.marian.translate_to.yap](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yap_fr_xx.html) | [opus_mt_yap_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yap_fr_xx.html) |
| [xx.de.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_de_xx.html) | [opus_mt_zh_de](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_de_xx.html) |
| [xx.es.marian.translate_to.yo](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yo_es_xx.html) | [opus_mt_yo_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_yo_es_xx.html) |
| [xx.es.marian.translate_to.vsl](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vsl_es_xx.html) | [opus_mt_vsl_es](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vsl_es_xx.html) |
| [xx.zle.marian.translate_to.zle](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zle_zle_xx.html) | [opus_mt_zle_zle](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zle_zle_xx.html) |
| [xx.fr.marian.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_fr_xx.html) | [opus_mt_vi_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_vi_fr_xx.html) |
| [xx.fr.marian.translate_to.war](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_war_fr_xx.html) | [opus_mt_war_fr](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_war_fr_xx.html) |
| [xx.fi.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_fi_xx.html) | [opus_mt_zh_fi](https://nlp.johnsnowlabs.com//2021/06/01/opus_mt_zh_fi_xx.html) |
| [xx.he.marian.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_he_it_xx.html) | [opus_tatoeba_he_it](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_he_it_xx.html) |
| [xx.es.marian.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_es_zh_xx.html) | [opus_tatoeba_es_zh](https://nlp.johnsnowlabs.com//2021/06/01/opus_tatoeba_es_zh_xx.html) |
| [xx.es.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_es_xx.html) | [translate_af_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_es_xx.html) |
| [xx.nl.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_nl_xx.html) | [translate_af_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_nl_xx.html) |
| [xx.eo.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_eo_xx.html) | [translate_af_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_eo_xx.html) |
| [xx.afa.translate_to.afa](https://nlp.johnsnowlabs.com//2021/06/04/translate_afa_afa_xx.html) | [translate_afa_afa](https://nlp.johnsnowlabs.com//2021/06/04/translate_afa_afa_xx.html) |
| [xx.sv.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_sv_xx.html) | [translate_af_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_sv_xx.html) |
| [xx.es.translate_to.aed](https://nlp.johnsnowlabs.com//2021/06/04/translate_aed_es_xx.html) | [translate_aed_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_aed_es_xx.html) |
| [xx.fr.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_fr_xx.html) | [translate_af_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_fr_xx.html) |
| [xx.fi.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_fi_xx.html) | [translate_af_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_fi_xx.html) |
| [xx.de.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_de_xx.html) | [translate_af_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_de_xx.html) |
| [xx.ru.translate_to.af](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_ru_xx.html) | [translate_af_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_af_ru_xx.html) |
| [xx.es.translate_to.az](https://nlp.johnsnowlabs.com//2021/06/04/translate_az_es_xx.html) | [translate_az_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_az_es_xx.html) |
| [xx.de.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_de_xx.html) | [translate_bcl_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_de_xx.html) |
| [xx.sv.translate_to.bem](https://nlp.johnsnowlabs.com//2021/06/04/translate_bem_sv_xx.html) | [translate_bem_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_bem_sv_xx.html) |
| [xx.tr.translate_to.az](https://nlp.johnsnowlabs.com//2021/06/04/translate_az_tr_xx.html) | [translate_az_tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_az_tr_xx.html) |
| [xx.sv.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_sv_xx.html) | [translate_bcl_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_sv_xx.html) |
| [xx.es.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_es_xx.html) | [translate_ar_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_es_xx.html) |
| [xx.es.translate_to.bem](https://nlp.johnsnowlabs.com//2021/06/04/translate_bem_es_xx.html) | [translate_bem_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_bem_es_xx.html) |
| [xx.ru.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_ru_xx.html) | [translate_ar_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_ru_xx.html) |
| [xx.es.translate_to.be](https://nlp.johnsnowlabs.com//2021/06/04/translate_be_es_xx.html) | [translate_be_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_be_es_xx.html) |
| [xx.fr.translate_to.bem](https://nlp.johnsnowlabs.com//2021/06/04/translate_bem_fr_xx.html) | [translate_bem_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_bem_fr_xx.html) |
| [xx.he.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_he_xx.html) | [translate_ar_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_he_xx.html) |
| [xx.es.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_es_xx.html) | [translate_bcl_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_es_xx.html) |
| [xx.es.translate_to.ase](https://nlp.johnsnowlabs.com//2021/06/04/translate_ase_es_xx.html) | [translate_ase_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ase_es_xx.html) |
| [xx.de.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_de_xx.html) | [translate_ar_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_de_xx.html) |
| [xx.pl.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_pl_xx.html) | [translate_ar_pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_pl_xx.html) |
| [xx.tr.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_tr_xx.html) | [translate_ar_tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_tr_xx.html) |
| [xx.sv.translate_to.ase](https://nlp.johnsnowlabs.com//2021/06/04/translate_ase_sv_xx.html) | [translate_ase_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ase_sv_xx.html) |
| [xx.fi.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_fi_xx.html) | [translate_bcl_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_fi_xx.html) |
| [xx.el.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_el_xx.html) | [translate_ar_el](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_el_xx.html) |
| [xx.fr.translate_to.bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_fr_xx.html) | [translate_bcl_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_bcl_fr_xx.html) |
| [xx.fi.translate_to.bem](https://nlp.johnsnowlabs.com//2021/06/04/translate_bem_fi_xx.html) | [translate_bem_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_bem_fi_xx.html) |
| [xx.fr.translate_to.ase](https://nlp.johnsnowlabs.com//2021/06/04/translate_ase_fr_xx.html) | [translate_ase_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ase_fr_xx.html) |
| [xx.fr.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_fr_xx.html) | [translate_ar_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_fr_xx.html) |
| [xx.eo.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_eo_xx.html) | [translate_ar_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_eo_xx.html) |
| [xx.it.translate_to.ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_it_xx.html) | [translate_ar_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_ar_it_xx.html) |
| [xx.sv.translate_to.am](https://nlp.johnsnowlabs.com//2021/06/04/translate_am_sv_xx.html) | [translate_am_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_am_sv_xx.html) |
| [xx.de.translate_to.ase](https://nlp.johnsnowlabs.com//2021/06/04/translate_ase_de_xx.html) | [translate_ase_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ase_de_xx.html) |
| [xx.uk.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_uk_xx.html) | [translate_bg_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_uk_xx.html) |
| [xx.it.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_it_xx.html) | [translate_bg_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_it_xx.html) |
| [xx.sv.translate_to.bzs](https://nlp.johnsnowlabs.com//2021/06/04/translate_bzs_sv_xx.html) | [translate_bzs_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_bzs_sv_xx.html) |
| [xx.pt.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_pt_xx.html) | [translate_ca_pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_pt_xx.html) |
| [xx.es.translate_to.ber](https://nlp.johnsnowlabs.com//2021/06/04/translate_ber_es_xx.html) | [translate_ber_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ber_es_xx.html) |
| [xx.it.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_it_xx.html) | [translate_ca_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_it_xx.html) |
| [xx.eo.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_eo_xx.html) | [translate_bg_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_eo_xx.html) |
| [xx.sv.translate_to.ceb](https://nlp.johnsnowlabs.com//2021/06/04/translate_ceb_sv_xx.html) | [translate_ceb_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ceb_sv_xx.html) |
| [xx.fr.translate_to.bi](https://nlp.johnsnowlabs.com//2021/06/04/translate_bi_fr_xx.html) | [translate_bi_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_bi_fr_xx.html) |
| [xx.sv.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_sv_xx.html) | [translate_bg_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_sv_xx.html) |
| [xx.fr.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_fr_xx.html) | [translate_ca_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_fr_xx.html) |
| [xx.tr.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_tr_xx.html) | [translate_bg_tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_tr_xx.html) |
| [xx.es.translate_to.ceb](https://nlp.johnsnowlabs.com//2021/06/04/translate_ceb_es_xx.html) | [translate_ceb_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ceb_es_xx.html) |
| [xx.de.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_de_xx.html) | [translate_ca_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_de_xx.html) |
| [xx.fi.translate_to.ceb](https://nlp.johnsnowlabs.com//2021/06/04/translate_ceb_fi_xx.html) | [translate_ceb_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ceb_fi_xx.html) |
| [xx.es.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_es_xx.html) | [translate_ca_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_es_xx.html) |
| [xx.es.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_es_xx.html) | [translate_bg_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_es_xx.html) |
| [xx.uk.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_uk_xx.html) | [translate_ca_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_uk_xx.html) |
| [xx.sv.translate_to.bi](https://nlp.johnsnowlabs.com//2021/06/04/translate_bi_sv_xx.html) | [translate_bi_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_bi_sv_xx.html) |
| [xx.sv.translate_to.chk](https://nlp.johnsnowlabs.com//2021/06/04/translate_chk_sv_xx.html) | [translate_chk_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_chk_sv_xx.html) |
| [xx.fr.translate_to.ceb](https://nlp.johnsnowlabs.com//2021/06/04/translate_ceb_fr_xx.html) | [translate_ceb_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ceb_fr_xx.html) |
| [xx.es.translate_to.bzs](https://nlp.johnsnowlabs.com//2021/06/04/translate_bzs_es_xx.html) | [translate_bzs_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_bzs_es_xx.html) |
| [xx.de.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_de_xx.html) | [translate_crs_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_de_xx.html) |
| [xx.nl.translate_to.ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_nl_xx.html) | [translate_ca_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_ca_nl_xx.html) |
| [xx.es.translate_to.chk](https://nlp.johnsnowlabs.com//2021/06/04/translate_chk_es_xx.html) | [translate_chk_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_chk_es_xx.html) |
| [xx.fr.translate_to.ber](https://nlp.johnsnowlabs.com//2021/06/04/translate_ber_fr_xx.html) | [translate_ber_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ber_fr_xx.html) |
| [xx.fi.translate_to.bzs](https://nlp.johnsnowlabs.com//2021/06/04/translate_bzs_fi_xx.html) | [translate_bzs_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_bzs_fi_xx.html) |
| [xx.es.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_es_xx.html) | [translate_crs_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_es_xx.html) |
| [xx.fi.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_fi_xx.html) | [translate_bg_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_fi_xx.html) |
| [xx.cpp.translate_to.cpp](https://nlp.johnsnowlabs.com//2021/06/04/translate_cpp_cpp_xx.html) | [translate_cpp_cpp](https://nlp.johnsnowlabs.com//2021/06/04/translate_cpp_cpp_xx.html) |
| [xx.de.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_de_xx.html) | [translate_bg_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_de_xx.html) |
| [xx.es.translate_to.bi](https://nlp.johnsnowlabs.com//2021/06/04/translate_bi_es_xx.html) | [translate_bi_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_bi_es_xx.html) |
| [xx.fr.translate_to.bzs](https://nlp.johnsnowlabs.com//2021/06/04/translate_bzs_fr_xx.html) | [translate_bzs_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_bzs_fr_xx.html) |
| [xx.fr.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_fr_xx.html) | [translate_bg_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_fr_xx.html) |
| [xx.fr.translate_to.chk](https://nlp.johnsnowlabs.com//2021/06/04/translate_chk_fr_xx.html) | [translate_chk_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_chk_fr_xx.html) |
| [xx.ru.translate_to.bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_ru_xx.html) | [translate_bg_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_bg_ru_xx.html) |
| [xx.fi.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_fi_xx.html) | [translate_cs_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_fi_xx.html) |
| [xx.ha.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ha_xx.html) | [translate_de_ha](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ha_xx.html) |
| [xx.ee.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ee_xx.html) | [translate_de_ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ee_xx.html) |
| [xx.eo.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_eo_xx.html) | [translate_de_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_eo_xx.html) |
| [xx.gil.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_gil_xx.html) | [translate_de_gil](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_gil_xx.html) |
| [xx.fj.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_fj_xx.html) | [translate_de_fj](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_fj_xx.html) |
| [xx.fr.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_fr_xx.html) | [translate_de_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_fr_xx.html) |
| [xx.sv.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_sv_xx.html) | [translate_cs_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_sv_xx.html) |
| [xx.es.translate_to.csn](https://nlp.johnsnowlabs.com//2021/06/04/translate_csn_es_xx.html) | [translate_csn_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_csn_es_xx.html) |
| [xx.ru.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_ru_xx.html) | [translate_da_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_ru_xx.html) |
| [xx.no.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_no_xx.html) | [translate_da_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_no_xx.html) |
| [xx.iso.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_iso_xx.html) | [translate_de_iso](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_iso_xx.html) |
| [xx.eu.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_eu_xx.html) | [translate_de_eu](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_eu_xx.html) |
| [xx.nl.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_nl_xx.html) | [translate_de_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_nl_xx.html) |
| [xx.ilo.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ilo_xx.html) | [translate_de_ilo](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ilo_xx.html) |
| [xx.hr.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_hr_xx.html) | [translate_de_hr](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_hr_xx.html) |
| [xx.mt.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_mt_xx.html) | [translate_de_mt](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_mt_xx.html) |
| [xx.es.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_es_xx.html) | [translate_da_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_es_xx.html) |
| [xx.ar.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ar_xx.html) | [translate_de_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ar_xx.html) |
| [xx.is.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_is_xx.html) | [translate_de_is](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_is_xx.html) |
| [xx.sv.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_sv_xx.html) | [translate_crs_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_sv_xx.html) |
| [xx.fr.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_fr_xx.html) | [translate_da_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_fr_xx.html) |
| [xx.gaa.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_gaa_xx.html) | [translate_de_gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_gaa_xx.html) |
| [xx.niu.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_niu_xx.html) | [translate_de_niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_niu_xx.html) |
| [xx.da.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_da_xx.html) | [translate_de_da](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_da_xx.html) |
| [xx.de.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_de_xx.html) | [translate_da_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_de_xx.html) |
| [xx.ase.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ase_xx.html) | [translate_de_ase](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ase_xx.html) |
| [xx.ig.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ig_xx.html) | [translate_de_ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ig_xx.html) |
| [xx.lua.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_lua_xx.html) | [translate_de_lua](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_lua_xx.html) |
| [xx.de.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_de_xx.html) | [translate_de_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_de_xx.html) |
| [xx.bi.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_bi_xx.html) | [translate_de_bi](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_bi_xx.html) |
| [xx.fr.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_fr_xx.html) | [translate_cs_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_fr_xx.html) |
| [xx.ms.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ms_xx.html) | [translate_de_ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ms_xx.html) |
| [xx.fi.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_fi_xx.html) | [translate_crs_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_fi_xx.html) |
| [xx.eo.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_eo_xx.html) | [translate_da_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_eo_xx.html) |
| [xx.af.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_af_xx.html) | [translate_de_af](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_af_xx.html) |
| [xx.uk.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_uk_xx.html) | [translate_cs_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_uk_xx.html) |
| [xx.bg.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_bg_xx.html) | [translate_de_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_bg_xx.html) |
| [xx.no.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_no_xx.html) | [translate_de_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_no_xx.html) |
| [xx.de.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_de_xx.html) | [translate_cs_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_de_xx.html) |
| [xx.it.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_it_xx.html) | [translate_de_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_it_xx.html) |
| [xx.ho.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ho_xx.html) | [translate_de_ho](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ho_xx.html) |
| [xx.ln.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ln_xx.html) | [translate_de_ln](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ln_xx.html) |
| [xx.guw.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_guw_xx.html) | [translate_de_guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_guw_xx.html) |
| [xx.efi.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_efi_xx.html) | [translate_de_efi](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_efi_xx.html) |
| [xx.hil.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_hil_xx.html) | [translate_de_hil](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_hil_xx.html) |
| [xx.cs.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_cs_xx.html) | [translate_de_cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_cs_xx.html) |
| [xx.es.translate_to.csg](https://nlp.johnsnowlabs.com//2021/06/04/translate_csg_es_xx.html) | [translate_csg_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_csg_es_xx.html) |
| [xx.es.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_es_xx.html) | [translate_de_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_es_xx.html) |
| [xx.bcl.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_bcl_xx.html) | [translate_de_bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_bcl_xx.html) |
| [xx.ht.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ht_xx.html) | [translate_de_ht](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ht_xx.html) |
| [xx.loz.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_loz_xx.html) | [translate_de_loz](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_loz_xx.html) |
| [xx.kg.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_kg_xx.html) | [translate_de_kg](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_kg_xx.html) |
| [xx.eo.translate_to.cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_eo_xx.html) | [translate_cs_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_cs_eo_xx.html) |
| [xx.el.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_el_xx.html) | [translate_de_el](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_el_xx.html) |
| [xx.fi.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_fi_xx.html) | [translate_de_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_fi_xx.html) |
| [xx.he.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_he_xx.html) | [translate_de_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_he_xx.html) |
| [xx.bzs.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_bzs_xx.html) | [translate_de_bzs](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_bzs_xx.html) |
| [xx.fr.translate_to.crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_fr_xx.html) | [translate_crs_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_crs_fr_xx.html) |
| [xx.crs.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_crs_xx.html) | [translate_de_crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_crs_xx.html) |
| [xx.fi.translate_to.da](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_fi_xx.html) | [translate_da_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_da_fi_xx.html) |
| [xx.hu.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_hu_xx.html) | [translate_de_hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_hu_xx.html) |
| [xx.et.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_et_xx.html) | [translate_de_et](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_et_xx.html) |
| [xx.lt.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_lt_xx.html) | [translate_de_lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_lt_xx.html) |
| [xx.ca.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ca_xx.html) | [translate_de_ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ca_xx.html) |
| [xx.pl.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pl_xx.html) | [translate_de_pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pl_xx.html) |
| [xx.sv.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_sv_xx.html) | [translate_el_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_sv_xx.html) |
| [xx.de.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_de_xx.html) | [translate_ee_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_de_xx.html) |
| [xx.pag.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pag_xx.html) | [translate_de_pag](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pag_xx.html) |
| [xx.ar.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_ar_xx.html) | [translate_el_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_ar_xx.html) |
| [xx.nso.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_nso_xx.html) | [translate_de_nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_nso_xx.html) |
| [xx.pon.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pon_xx.html) | [translate_de_pon](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pon_xx.html) |
| [xx.pap.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pap_xx.html) | [translate_de_pap](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pap_xx.html) |
| [xx.fr.translate_to.efi](https://nlp.johnsnowlabs.com//2021/06/04/translate_efi_fr_xx.html) | [translate_efi_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_efi_fr_xx.html) |
| [xx.pis.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pis_xx.html) | [translate_de_pis](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_pis_xx.html) |
| [xx.de.translate_to.efi](https://nlp.johnsnowlabs.com//2021/06/04/translate_efi_de_xx.html) | [translate_efi_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_efi_de_xx.html) |
| [xx.eo.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_eo_xx.html) | [translate_el_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_eo_xx.html) |
| [xx.fi.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_fi_xx.html) | [translate_ee_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_fi_xx.html) |
| [xx.es.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_es_xx.html) | [translate_ee_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_es_xx.html) |
| [xx.fr.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_fr_xx.html) | [translate_ee_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_fr_xx.html) |
| [xx.fi.translate_to.efi](https://nlp.johnsnowlabs.com//2021/06/04/translate_efi_fi_xx.html) | [translate_efi_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_efi_fi_xx.html) |
| [xx.fr.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_fr_xx.html) | [translate_el_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_fr_xx.html) |
| [xx.tl.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_tl_xx.html) | [translate_de_tl](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_tl_xx.html) |
| [xx.ny.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ny_xx.html) | [translate_de_ny](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_ny_xx.html) |
| [xx.uk.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_uk_xx.html) | [translate_de_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_uk_xx.html) |
| [xx.sv.translate_to.efi](https://nlp.johnsnowlabs.com//2021/06/04/translate_efi_sv_xx.html) | [translate_efi_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_efi_sv_xx.html) |
| [xx.sv.translate_to.ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_sv_xx.html) | [translate_ee_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ee_sv_xx.html) |
| [xx.vi.translate_to.de](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_vi_xx.html) | [translate_de_vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_de_vi_xx.html) |
| [xx.fi.translate_to.el](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_fi_xx.html) | [translate_el_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_el_fi_xx.html) |
| [xx.cs.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_cs_xx.html) | [translate_eo_cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_cs_xx.html) |
| [xx.bzs.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_bzs_xx.html) | [translate_es_bzs](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_bzs_xx.html) |
| [xx.he.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_he_xx.html) | [translate_eo_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_he_xx.html) |
| [xx.hu.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_hu_xx.html) | [translate_eo_hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_hu_xx.html) |
| [xx.ro.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_ro_xx.html) | [translate_eo_ro](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_ro_xx.html) |
| [xx.ber.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ber_xx.html) | [translate_es_ber](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ber_xx.html) |
| [xx.ca.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ca_xx.html) | [translate_es_ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ca_xx.html) |
| [xx.bcl.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_bcl_xx.html) | [translate_es_bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_bcl_xx.html) |
| [xx.ceb.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ceb_xx.html) | [translate_es_ceb](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ceb_xx.html) |
| [xx.da.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_da_xx.html) | [translate_eo_da](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_da_xx.html) |
| [xx.bi.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_bi_xx.html) | [translate_es_bi](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_bi_xx.html) |
| [xx.ee.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ee_xx.html) | [translate_es_ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ee_xx.html) |
| [xx.ru.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_ru_xx.html) | [translate_eo_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_ru_xx.html) |
| [xx.csg.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_csg_xx.html) | [translate_es_csg](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_csg_xx.html) |
| [xx.fi.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_fi_xx.html) | [translate_eo_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_fi_xx.html) |
| [xx.it.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_it_xx.html) | [translate_eo_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_it_xx.html) |
| [xx.nl.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_nl_xx.html) | [translate_eo_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_nl_xx.html) |
| [xx.et.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_et_xx.html) | [translate_es_et](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_et_xx.html) |
| [xx.bg.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_bg_xx.html) | [translate_es_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_bg_xx.html) |
| [xx.de.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_de_xx.html) | [translate_eo_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_de_xx.html) |
| [xx.ar.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ar_xx.html) | [translate_es_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ar_xx.html) |
| [xx.cs.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_cs_xx.html) | [translate_es_cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_cs_xx.html) |
| [xx.aed.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_aed_xx.html) | [translate_es_aed](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_aed_xx.html) |
| [xx.ase.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ase_xx.html) | [translate_es_ase](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ase_xx.html) |
| [xx.el.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_el_xx.html) | [translate_es_el](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_el_xx.html) |
| [xx.eo.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_eo_xx.html) | [translate_es_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_eo_xx.html) |
| [xx.af.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_af_xx.html) | [translate_eo_af](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_af_xx.html) |
| [xx.af.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_af_xx.html) | [translate_es_af](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_af_xx.html) |
| [xx.pl.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_pl_xx.html) | [translate_eo_pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_pl_xx.html) |
| [xx.de.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_de_xx.html) | [translate_es_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_de_xx.html) |
| [xx.es.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_es_xx.html) | [translate_eo_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_es_xx.html) |
| [xx.da.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_da_xx.html) | [translate_es_da](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_da_xx.html) |
| [xx.crs.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_crs_xx.html) | [translate_es_crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_crs_xx.html) |
| [xx.pt.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_pt_xx.html) | [translate_eo_pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_pt_xx.html) |
| [xx.eu.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_eu_xx.html) | [translate_es_eu](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_eu_xx.html) |
| [xx.es.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_es_xx.html) | [translate_es_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_es_xx.html) |
| [xx.csn.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_csn_xx.html) | [translate_es_csn](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_csn_xx.html) |
| [xx.sv.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_sv_xx.html) | [translate_eo_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_sv_xx.html) |
| [xx.efi.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_efi_xx.html) | [translate_es_efi](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_efi_xx.html) |
| [xx.sh.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_sh_xx.html) | [translate_eo_sh](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_sh_xx.html) |
| [xx.bg.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_bg_xx.html) | [translate_eo_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_bg_xx.html) |
| [xx.fr.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_fr_xx.html) | [translate_eo_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_fr_xx.html) |
| [xx.el.translate_to.eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_el_xx.html) | [translate_eo_el](https://nlp.johnsnowlabs.com//2021/06/04/translate_eo_el_xx.html) |
| [xx.pl.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pl_xx.html) | [translate_es_pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pl_xx.html) |
| [xx.ro.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ro_xx.html) | [translate_es_ro](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ro_xx.html) |
| [xx.is.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_is_xx.html) | [translate_es_is](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_is_xx.html) |
| [xx.ln.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ln_xx.html) | [translate_es_ln](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ln_xx.html) |
| [xx.to.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_to_xx.html) | [translate_es_to](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_to_xx.html) |
| [xx.no.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_no_xx.html) | [translate_es_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_no_xx.html) |
| [xx.nl.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_nl_xx.html) | [translate_es_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_nl_xx.html) |
| [xx.pag.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pag_xx.html) | [translate_es_pag](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pag_xx.html) |
| [xx.tvl.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tvl_xx.html) | [translate_es_tvl](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tvl_xx.html) |
| [xx.fr.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_fr_xx.html) | [translate_es_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_fr_xx.html) |
| [xx.he.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_he_xx.html) | [translate_es_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_he_xx.html) |
| [xx.lus.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_lus_xx.html) | [translate_es_lus](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_lus_xx.html) |
| [xx.hil.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_hil_xx.html) | [translate_es_hil](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_hil_xx.html) |
| [xx.ny.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ny_xx.html) | [translate_es_ny](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ny_xx.html) |
| [xx.pap.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pap_xx.html) | [translate_es_pap](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pap_xx.html) |
| [xx.id.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_id_xx.html) | [translate_es_id](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_id_xx.html) |
| [xx.wls.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_wls_xx.html) | [translate_es_wls](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_wls_xx.html) |
| [xx.gaa.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_gaa_xx.html) | [translate_es_gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_gaa_xx.html) |
| [xx.nso.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_nso_xx.html) | [translate_es_nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_nso_xx.html) |
| [xx.mk.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_mk_xx.html) | [translate_es_mk](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_mk_xx.html) |
| [xx.mt.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_mt_xx.html) | [translate_es_mt](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_mt_xx.html) |
| [xx.pis.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pis_xx.html) | [translate_es_pis](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pis_xx.html) |
| [xx.gl.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_gl_xx.html) | [translate_es_gl](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_gl_xx.html) |
| [xx.sn.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_sn_xx.html) | [translate_es_sn](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_sn_xx.html) |
| [xx.hr.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_hr_xx.html) | [translate_es_hr](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_hr_xx.html) |
| [xx.swc.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_swc_xx.html) | [translate_es_swc](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_swc_xx.html) |
| [xx.lua.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_lua_xx.html) | [translate_es_lua](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_lua_xx.html) |
| [xx.it.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_it_xx.html) | [translate_es_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_it_xx.html) |
| [xx.fj.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_fj_xx.html) | [translate_es_fj](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_fj_xx.html) |
| [xx.gil.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_gil_xx.html) | [translate_es_gil](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_gil_xx.html) |
| [xx.sm.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_sm_xx.html) | [translate_es_sm](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_sm_xx.html) |
| [xx.guw.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_guw_xx.html) | [translate_es_guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_guw_xx.html) |
| [xx.kg.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_kg_xx.html) | [translate_es_kg](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_kg_xx.html) |
| [xx.tl.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tl_xx.html) | [translate_es_tl](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tl_xx.html) |
| [xx.rn.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_rn_xx.html) | [translate_es_rn](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_rn_xx.html) |
| [xx.mfs.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_mfs_xx.html) | [translate_es_mfs](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_mfs_xx.html) |
| [xx.iso.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_iso_xx.html) | [translate_es_iso](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_iso_xx.html) |
| [xx.loz.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_loz_xx.html) | [translate_es_loz](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_loz_xx.html) |
| [xx.tpi.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tpi_xx.html) | [translate_es_tpi](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tpi_xx.html) |
| [xx.ha.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ha_xx.html) | [translate_es_ha](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ha_xx.html) |
| [xx.ht.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ht_xx.html) | [translate_es_ht](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ht_xx.html) |
| [xx.uk.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_uk_xx.html) | [translate_es_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_uk_xx.html) |
| [xx.tw.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tw_xx.html) | [translate_es_tw](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tw_xx.html) |
| [xx.st.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_st_xx.html) | [translate_es_st](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_st_xx.html) |
| [xx.sg.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_sg_xx.html) | [translate_es_sg](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_sg_xx.html) |
| [xx.ilo.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ilo_xx.html) | [translate_es_ilo](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ilo_xx.html) |
| [xx.ru.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ru_xx.html) | [translate_es_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ru_xx.html) |
| [xx.yo.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_yo_xx.html) | [translate_es_yo](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_yo_xx.html) |
| [xx.pon.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pon_xx.html) | [translate_es_pon](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_pon_xx.html) |
| [xx.niu.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_niu_xx.html) | [translate_es_niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_niu_xx.html) |
| [xx.lt.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_lt_xx.html) | [translate_es_lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_lt_xx.html) |
| [xx.ty.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ty_xx.html) | [translate_es_ty](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ty_xx.html) |
| [xx.ig.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ig_xx.html) | [translate_es_ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ig_xx.html) |
| [xx.tzo.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tzo_xx.html) | [translate_es_tzo](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tzo_xx.html) |
| [xx.rw.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_rw_xx.html) | [translate_es_rw](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_rw_xx.html) |
| [xx.war.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_war_xx.html) | [translate_es_war](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_war_xx.html) |
| [xx.tll.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tll_xx.html) | [translate_es_tll](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tll_xx.html) |
| [xx.prl.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_prl_xx.html) | [translate_es_prl](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_prl_xx.html) |
| [xx.xh.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_xh_xx.html) | [translate_es_xh](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_xh_xx.html) |
| [xx.yua.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_yua_xx.html) | [translate_es_yua](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_yua_xx.html) |
| [xx.ho.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ho_xx.html) | [translate_es_ho](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ho_xx.html) |
| [xx.ve.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ve_xx.html) | [translate_es_ve](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_ve_xx.html) |
| [xx.sl.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_sl_xx.html) | [translate_es_sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_sl_xx.html) |
| [xx.tn.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tn_xx.html) | [translate_es_tn](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_tn_xx.html) |
| [xx.vi.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_vi_xx.html) | [translate_es_vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_vi_xx.html) |
| [xx.srn.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_srn_xx.html) | [translate_es_srn](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_srn_xx.html) |
| [xx.fi.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_fi_xx.html) | [translate_es_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_fi_xx.html) |
| [xx.lua.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lua_xx.html) | [translate_fi_lua](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lua_xx.html) |
| [xx.ny.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ny_xx.html) | [translate_fi_ny](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ny_xx.html) |
| [xx.pon.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_pon_xx.html) | [translate_fi_pon](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_pon_xx.html) |
| [xx.crs.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_crs_xx.html) | [translate_fi_crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_crs_xx.html) |
| [xx.nso.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_nso_xx.html) | [translate_fi_nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_nso_xx.html) |
| [xx.iso.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_iso_xx.html) | [translate_fi_iso](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_iso_xx.html) |
| [xx.kqn.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_kqn_xx.html) | [translate_fi_kqn](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_kqn_xx.html) |
| [xx.gaa.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_gaa_xx.html) | [translate_fi_gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_gaa_xx.html) |
| [xx.ru.translate_to.eu](https://nlp.johnsnowlabs.com//2021/06/04/translate_eu_ru_xx.html) | [translate_eu_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_eu_ru_xx.html) |
| [xx.eo.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_eo_xx.html) | [translate_fi_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_eo_xx.html) |
| [xx.ig.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ig_xx.html) | [translate_fi_ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ig_xx.html) |
| [xx.bem.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_bem_xx.html) | [translate_fi_bem](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_bem_xx.html) |
| [xx.es.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_es_xx.html) | [translate_et_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_es_xx.html) |
| [xx.fj.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_fj_xx.html) | [translate_fi_fj](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_fj_xx.html) |
| [xx.et.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_et_xx.html) | [translate_fi_et](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_et_xx.html) |
| [xx.bcl.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_bcl_xx.html) | [translate_fi_bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_bcl_xx.html) |
| [xx.fi.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_fi_xx.html) | [translate_fi_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_fi_xx.html) |
| [xx.el.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_el_xx.html) | [translate_fi_el](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_el_xx.html) |
| [xx.efi.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_efi_xx.html) | [translate_fi_efi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_efi_xx.html) |
| [xx.ht.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ht_xx.html) | [translate_fi_ht](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ht_xx.html) |
| [xx.ceb.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ceb_xx.html) | [translate_fi_ceb](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ceb_xx.html) |
| [xx.lg.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lg_xx.html) | [translate_fi_lg](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lg_xx.html) |
| [xx.pap.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_pap_xx.html) | [translate_fi_pap](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_pap_xx.html) |
| [xx.kg.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_kg_xx.html) | [translate_fi_kg](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_kg_xx.html) |
| [xx.ee.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ee_xx.html) | [translate_fi_ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ee_xx.html) |
| [xx.lv.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lv_xx.html) | [translate_fi_lv](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lv_xx.html) |
| [xx.fr.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_fr_xx.html) | [translate_et_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_fr_xx.html) |
| [xx.de.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_de_xx.html) | [translate_et_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_de_xx.html) |
| [xx.bzs.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_bzs_xx.html) | [translate_fi_bzs](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_bzs_xx.html) |
| [xx.mos.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mos_xx.html) | [translate_fi_mos](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mos_xx.html) |
| [xx.zh.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_zh_xx.html) | [translate_es_zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_zh_xx.html) |
| [xx.id.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_id_xx.html) | [translate_fi_id](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_id_xx.html) |
| [xx.gil.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_gil_xx.html) | [translate_fi_gil](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_gil_xx.html) |
| [xx.pis.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_pis_xx.html) | [translate_fi_pis](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_pis_xx.html) |
| [xx.no.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_no_xx.html) | [translate_fi_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_no_xx.html) |
| [xx.it.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_it_xx.html) | [translate_fi_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_it_xx.html) |
| [xx.es.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_es_xx.html) | [translate_fi_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_es_xx.html) |
| [xx.ha.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ha_xx.html) | [translate_fi_ha](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ha_xx.html) |
| [xx.fr.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_fr_xx.html) | [translate_fi_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_fr_xx.html) |
| [xx.de.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_de_xx.html) | [translate_fi_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_de_xx.html) |
| [xx.bg.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_bg_xx.html) | [translate_fi_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_bg_xx.html) |
| [xx.zai.translate_to.es](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_zai_xx.html) | [translate_es_zai](https://nlp.johnsnowlabs.com//2021/06/04/translate_es_zai_xx.html) |
| [xx.hil.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_hil_xx.html) | [translate_fi_hil](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_hil_xx.html) |
| [xx.cs.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_cs_xx.html) | [translate_fi_cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_cs_xx.html) |
| [xx.es.translate_to.eu](https://nlp.johnsnowlabs.com//2021/06/04/translate_eu_es_xx.html) | [translate_eu_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_eu_es_xx.html) |
| [xx.ilo.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ilo_xx.html) | [translate_fi_ilo](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ilo_xx.html) |
| [xx.pag.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_pag_xx.html) | [translate_fi_pag](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_pag_xx.html) |
| [xx.ln.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ln_xx.html) | [translate_fi_ln](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ln_xx.html) |
| [xx.sv.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_sv_xx.html) | [translate_et_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_sv_xx.html) |
| [xx.niu.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_niu_xx.html) | [translate_fi_niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_niu_xx.html) |
| [xx.hr.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_hr_xx.html) | [translate_fi_hr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_hr_xx.html) |
| [xx.de.translate_to.eu](https://nlp.johnsnowlabs.com//2021/06/04/translate_eu_de_xx.html) | [translate_eu_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_eu_de_xx.html) |
| [xx.lus.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lus_xx.html) | [translate_fi_lus](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lus_xx.html) |
| [xx.ru.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_ru_xx.html) | [translate_et_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_ru_xx.html) |
| [xx.af.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_af_xx.html) | [translate_fi_af](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_af_xx.html) |
| [xx.mh.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mh_xx.html) | [translate_fi_mh](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mh_xx.html) |
| [xx.guw.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_guw_xx.html) | [translate_fi_guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_guw_xx.html) |
| [xx.mfe.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mfe_xx.html) | [translate_fi_mfe](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mfe_xx.html) |
| [xx.ho.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ho_xx.html) | [translate_fi_ho](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ho_xx.html) |
| [xx.fse.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_fse_xx.html) | [translate_fi_fse](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_fse_xx.html) |
| [xx.lu.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lu_xx.html) | [translate_fi_lu](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lu_xx.html) |
| [xx.hu.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_hu_xx.html) | [translate_fi_hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_hu_xx.html) |
| [xx.mk.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mk_xx.html) | [translate_fi_mk](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mk_xx.html) |
| [xx.nl.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_nl_xx.html) | [translate_fi_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_nl_xx.html) |
| [xx.mg.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mg_xx.html) | [translate_fi_mg](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mg_xx.html) |
| [xx.mt.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mt_xx.html) | [translate_fi_mt](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_mt_xx.html) |
| [xx.he.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_he_xx.html) | [translate_fi_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_he_xx.html) |
| [xx.fi.translate_to.et](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_fi_xx.html) | [translate_et_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_et_fi_xx.html) |
| [xx.is.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_is_xx.html) | [translate_fi_is](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_is_xx.html) |
| [xx.lue.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lue_xx.html) | [translate_fi_lue](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_lue_xx.html) |
| [xx.guw.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_guw_xx.html) | [translate_fr_guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_guw_xx.html) |
| [xx.ber.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ber_xx.html) | [translate_fr_ber](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ber_xx.html) |
| [xx.uk.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_uk_xx.html) | [translate_fi_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_uk_xx.html) |
| [xx.efi.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_efi_xx.html) | [translate_fr_efi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_efi_xx.html) |
| [xx.tr.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tr_xx.html) | [translate_fi_tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tr_xx.html) |
| [xx.tn.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tn_xx.html) | [translate_fi_tn](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tn_xx.html) |
| [xx.es.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_es_xx.html) | [translate_fr_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_es_xx.html) |
| [xx.srn.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_srn_xx.html) | [translate_fi_srn](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_srn_xx.html) |
| [xx.bcl.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bcl_xx.html) | [translate_fr_bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bcl_xx.html) |
| [xx.sl.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sl_xx.html) | [translate_fi_sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sl_xx.html) |
| [xx.ht.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ht_xx.html) | [translate_fr_ht](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ht_xx.html) |
| [xx.zne.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_zne_xx.html) | [translate_fi_zne](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_zne_xx.html) |
| [xx.de.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_de_xx.html) | [translate_fr_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_de_xx.html) |
| [xx.war.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_war_xx.html) | [translate_fi_war](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_war_xx.html) |
| [xx.tpi.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tpi_xx.html) | [translate_fi_tpi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tpi_xx.html) |
| [xx.ca.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ca_xx.html) | [translate_fr_ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ca_xx.html) |
| [xx.yap.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_yap_xx.html) | [translate_fi_yap](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_yap_xx.html) |
| [xx.sn.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sn_xx.html) | [translate_fi_sn](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sn_xx.html) |
| [xx.hr.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_hr_xx.html) | [translate_fr_hr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_hr_xx.html) |
| [xx.gil.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_gil_xx.html) | [translate_fr_gil](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_gil_xx.html) |
| [xx.id.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_id_xx.html) | [translate_fr_id](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_id_xx.html) |
| [xx.sv.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sv_xx.html) | [translate_fi_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sv_xx.html) |
| [xx.toi.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_toi_xx.html) | [translate_fi_toi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_toi_xx.html) |
| [xx.sk.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sk_xx.html) | [translate_fi_sk](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sk_xx.html) |
| [xx.he.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_he_xx.html) | [translate_fr_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_he_xx.html) |
| [xx.sq.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sq_xx.html) | [translate_fi_sq](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sq_xx.html) |
| [xx.ve.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ve_xx.html) | [translate_fi_ve](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ve_xx.html) |
| [xx.tw.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tw_xx.html) | [translate_fi_tw](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tw_xx.html) |
| [xx.tvl.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tvl_xx.html) | [translate_fi_tvl](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tvl_xx.html) |
| [xx.hil.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_hil_xx.html) | [translate_fr_hil](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_hil_xx.html) |
| [xx.sw.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sw_xx.html) | [translate_fi_sw](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sw_xx.html) |
| [xx.eo.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_eo_xx.html) | [translate_fr_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_eo_xx.html) |
| [xx.xh.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_xh_xx.html) | [translate_fi_xh](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_xh_xx.html) |
| [xx.bi.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bi_xx.html) | [translate_fr_bi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bi_xx.html) |
| [xx.ru.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ru_xx.html) | [translate_fi_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ru_xx.html) |
| [xx.ceb.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ceb_xx.html) | [translate_fr_ceb](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ceb_xx.html) |
| [xx.ig.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ig_xx.html) | [translate_fr_ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ig_xx.html) |
| [xx.el.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_el_xx.html) | [translate_fr_el](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_el_xx.html) |
| [xx.sm.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sm_xx.html) | [translate_fi_sm](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sm_xx.html) |
| [xx.to.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_to_xx.html) | [translate_fi_to](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_to_xx.html) |
| [xx.ase.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ase_xx.html) | [translate_fr_ase](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ase_xx.html) |
| [xx.yo.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_yo_xx.html) | [translate_fi_yo](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_yo_xx.html) |
| [xx.sg.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sg_xx.html) | [translate_fi_sg](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_sg_xx.html) |
| [xx.rw.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_rw_xx.html) | [translate_fi_rw](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_rw_xx.html) |
| [xx.ts.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ts_xx.html) | [translate_fi_ts](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ts_xx.html) |
| [xx.wls.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_wls_xx.html) | [translate_fi_wls](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_wls_xx.html) |
| [xx.ho.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ho_xx.html) | [translate_fr_ho](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ho_xx.html) |
| [xx.tll.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tll_xx.html) | [translate_fi_tll](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tll_xx.html) |
| [xx.st.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_st_xx.html) | [translate_fi_st](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_st_xx.html) |
| [xx.fiu.translate_to.fiu](https://nlp.johnsnowlabs.com//2021/06/04/translate_fiu_fiu_xx.html) | [translate_fiu_fiu](https://nlp.johnsnowlabs.com//2021/06/04/translate_fiu_fiu_xx.html) |
| [xx.ro.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ro_xx.html) | [translate_fi_ro](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ro_xx.html) |
| [xx.tiv.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tiv_xx.html) | [translate_fi_tiv](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_tiv_xx.html) |
| [xx.ha.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ha_xx.html) | [translate_fr_ha](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ha_xx.html) |
| [xx.ee.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ee_xx.html) | [translate_fr_ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ee_xx.html) |
| [xx.gaa.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_gaa_xx.html) | [translate_fr_gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_gaa_xx.html) |
| [xx.hu.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_hu_xx.html) | [translate_fr_hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_hu_xx.html) |
| [xx.ty.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ty_xx.html) | [translate_fi_ty](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_ty_xx.html) |
| [xx.fr.translate_to.fj](https://nlp.johnsnowlabs.com//2021/06/04/translate_fj_fr_xx.html) | [translate_fj_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fj_fr_xx.html) |
| [xx.run.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_run_xx.html) | [translate_fi_run](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_run_xx.html) |
| [xx.bem.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bem_xx.html) | [translate_fr_bem](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bem_xx.html) |
| [xx.bzs.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bzs_xx.html) | [translate_fr_bzs](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bzs_xx.html) |
| [xx.fj.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_fj_xx.html) | [translate_fr_fj](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_fj_xx.html) |
| [xx.ar.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ar_xx.html) | [translate_fr_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ar_xx.html) |
| [xx.swc.translate_to.fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_swc_xx.html) | [translate_fi_swc](https://nlp.johnsnowlabs.com//2021/06/04/translate_fi_swc_xx.html) |
| [xx.crs.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_crs_xx.html) | [translate_fr_crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_crs_xx.html) |
| [xx.bg.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bg_xx.html) | [translate_fr_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_bg_xx.html) |
| [xx.af.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_af_xx.html) | [translate_fr_af](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_af_xx.html) |
| [xx.loz.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_loz_xx.html) | [translate_fr_loz](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_loz_xx.html) |
| [xx.st.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_st_xx.html) | [translate_fr_st](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_st_xx.html) |
| [xx.tn.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tn_xx.html) | [translate_fr_tn](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tn_xx.html) |
| [xx.srn.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_srn_xx.html) | [translate_fr_srn](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_srn_xx.html) |
| [xx.to.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_to_xx.html) | [translate_fr_to](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_to_xx.html) |
| [xx.sk.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sk_xx.html) | [translate_fr_sk](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sk_xx.html) |
| [xx.tum.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tum_xx.html) | [translate_fr_tum](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tum_xx.html) |
| [xx.ts.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ts_xx.html) | [translate_fr_ts](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ts_xx.html) |
| [xx.iso.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_iso_xx.html) | [translate_fr_iso](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_iso_xx.html) |
| [xx.sv.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sv_xx.html) | [translate_fr_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sv_xx.html) |
| [xx.mt.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_mt_xx.html) | [translate_fr_mt](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_mt_xx.html) |
| [xx.pap.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pap_xx.html) | [translate_fr_pap](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pap_xx.html) |
| [xx.wls.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_wls_xx.html) | [translate_fr_wls](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_wls_xx.html) |
| [xx.lua.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lua_xx.html) | [translate_fr_lua](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lua_xx.html) |
| [xx.ro.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ro_xx.html) | [translate_fr_ro](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ro_xx.html) |
| [xx.tll.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tll_xx.html) | [translate_fr_tll](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tll_xx.html) |
| [xx.ilo.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ilo_xx.html) | [translate_fr_ilo](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ilo_xx.html) |
| [xx.ve.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ve_xx.html) | [translate_fr_ve](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ve_xx.html) |
| [xx.ny.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ny_xx.html) | [translate_fr_ny](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ny_xx.html) |
| [xx.tpi.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tpi_xx.html) | [translate_fr_tpi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tpi_xx.html) |
| [xx.uk.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_uk_xx.html) | [translate_fr_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_uk_xx.html) |
| [xx.ln.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ln_xx.html) | [translate_fr_ln](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ln_xx.html) |
| [xx.mfe.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_mfe_xx.html) | [translate_fr_mfe](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_mfe_xx.html) |
| [xx.lue.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lue_xx.html) | [translate_fr_lue](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lue_xx.html) |
| [xx.mos.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_mos_xx.html) | [translate_fr_mos](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_mos_xx.html) |
| [xx.pon.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pon_xx.html) | [translate_fr_pon](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pon_xx.html) |
| [xx.tvl.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tvl_xx.html) | [translate_fr_tvl](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tvl_xx.html) |
| [xx.run.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_run_xx.html) | [translate_fr_run](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_run_xx.html) |
| [xx.pag.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pag_xx.html) | [translate_fr_pag](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pag_xx.html) |
| [xx.sg.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sg_xx.html) | [translate_fr_sg](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sg_xx.html) |
| [xx.no.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_no_xx.html) | [translate_fr_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_no_xx.html) |
| [xx.ty.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ty_xx.html) | [translate_fr_ty](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ty_xx.html) |
| [xx.tl.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tl_xx.html) | [translate_fr_tl](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tl_xx.html) |
| [xx.sl.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sl_xx.html) | [translate_fr_sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sl_xx.html) |
| [xx.tiv.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tiv_xx.html) | [translate_fr_tiv](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tiv_xx.html) |
| [xx.rw.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_rw_xx.html) | [translate_fr_rw](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_rw_xx.html) |
| [xx.lus.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lus_xx.html) | [translate_fr_lus](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lus_xx.html) |
| [xx.swc.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_swc_xx.html) | [translate_fr_swc](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_swc_xx.html) |
| [xx.sm.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sm_xx.html) | [translate_fr_sm](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sm_xx.html) |
| [xx.pl.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pl_xx.html) | [translate_fr_pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pl_xx.html) |
| [xx.kg.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_kg_xx.html) | [translate_fr_kg](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_kg_xx.html) |
| [xx.niu.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_niu_xx.html) | [translate_fr_niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_niu_xx.html) |
| [xx.lg.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lg_xx.html) | [translate_fr_lg](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lg_xx.html) |
| [xx.ms.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ms_xx.html) | [translate_fr_ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ms_xx.html) |
| [xx.nso.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_nso_xx.html) | [translate_fr_nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_nso_xx.html) |
| [xx.war.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_war_xx.html) | [translate_fr_war](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_war_xx.html) |
| [xx.xh.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_xh_xx.html) | [translate_fr_xh](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_xh_xx.html) |
| [xx.pis.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pis_xx.html) | [translate_fr_pis](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_pis_xx.html) |
| [xx.tw.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tw_xx.html) | [translate_fr_tw](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_tw_xx.html) |
| [xx.kwy.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_kwy_xx.html) | [translate_fr_kwy](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_kwy_xx.html) |
| [xx.rnd.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_rnd_xx.html) | [translate_fr_rnd](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_rnd_xx.html) |
| [xx.vi.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_vi_xx.html) | [translate_fr_vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_vi_xx.html) |
| [xx.lu.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lu_xx.html) | [translate_fr_lu](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_lu_xx.html) |
| [xx.mh.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_mh_xx.html) | [translate_fr_mh](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_mh_xx.html) |
| [xx.ru.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ru_xx.html) | [translate_fr_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_ru_xx.html) |
| [xx.sn.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sn_xx.html) | [translate_fr_sn](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_sn_xx.html) |
| [xx.kqn.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_kqn_xx.html) | [translate_fr_kqn](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_kqn_xx.html) |
| [xx.ar.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_ar_xx.html) | [translate_he_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_ar_xx.html) |
| [xx.de.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_de_xx.html) | [translate_he_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_de_xx.html) |
| [xx.es.translate_to.gil](https://nlp.johnsnowlabs.com//2021/06/04/translate_gil_es_xx.html) | [translate_gil_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_gil_es_xx.html) |
| [xx.de.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_de_xx.html) | [translate_gaa_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_de_xx.html) |
| [xx.fr.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_fr_xx.html) | [translate_hu_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_fr_xx.html) |
| [xx.fr.translate_to.gil](https://nlp.johnsnowlabs.com//2021/06/04/translate_gil_fr_xx.html) | [translate_gil_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_gil_fr_xx.html) |
| [xx.de.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_de_xx.html) | [translate_guw_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_de_xx.html) |
| [xx.fr.translate_to.ht](https://nlp.johnsnowlabs.com//2021/06/04/translate_ht_fr_xx.html) | [translate_ht_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ht_fr_xx.html) |
| [xx.uk.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_uk_xx.html) | [translate_he_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_uk_xx.html) |
| [xx.fi.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_fi_xx.html) | [translate_hu_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_fi_xx.html) |
| [xx.uk.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_uk_xx.html) | [translate_hu_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_uk_xx.html) |
| [xx.zne.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_zne_xx.html) | [translate_fr_zne](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_zne_xx.html) |
| [xx.sv.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_sv_xx.html) | [translate_gaa_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_sv_xx.html) |
| [xx.es.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_es_xx.html) | [translate_guw_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_es_xx.html) |
| [xx.gmq.translate_to.gmq](https://nlp.johnsnowlabs.com//2021/06/04/translate_gmq_gmq_xx.html) | [translate_gmq_gmq](https://nlp.johnsnowlabs.com//2021/06/04/translate_gmq_gmq_xx.html) |
| [xx.fi.translate_to.hil](https://nlp.johnsnowlabs.com//2021/06/04/translate_hil_fi_xx.html) | [translate_hil_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_hil_fi_xx.html) |
| [xx.fi.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_fi_xx.html) | [translate_guw_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_fi_xx.html) |
| [xx.es.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_es_xx.html) | [translate_he_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_es_xx.html) |
| [xx.ur.translate_to.hi](https://nlp.johnsnowlabs.com//2021/06/04/translate_hi_ur_xx.html) | [translate_hi_ur](https://nlp.johnsnowlabs.com//2021/06/04/translate_hi_ur_xx.html) |
| [xx.de.translate_to.hil](https://nlp.johnsnowlabs.com//2021/06/04/translate_hil_de_xx.html) | [translate_hil_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_hil_de_xx.html) |
| [xx.gmw.translate_to.gmw](https://nlp.johnsnowlabs.com//2021/06/04/translate_gmw_gmw_xx.html) | [translate_gmw_gmw](https://nlp.johnsnowlabs.com//2021/06/04/translate_gmw_gmw_xx.html) |
| [xx.fi.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_fi_xx.html) | [translate_gaa_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_fi_xx.html) |
| [xx.fi.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_fi_xx.html) | [translate_he_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_fi_xx.html) |
| [xx.eo.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_eo_xx.html) | [translate_hu_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_eo_xx.html) |
| [xx.fi.translate_to.ht](https://nlp.johnsnowlabs.com//2021/06/04/translate_ht_fi_xx.html) | [translate_ht_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ht_fi_xx.html) |
| [xx.yo.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_yo_xx.html) | [translate_fr_yo](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_yo_xx.html) |
| [xx.sv.translate_to.hr](https://nlp.johnsnowlabs.com//2021/06/04/translate_hr_sv_xx.html) | [translate_hr_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_hr_sv_xx.html) |
| [xx.fr.translate_to.ha](https://nlp.johnsnowlabs.com//2021/06/04/translate_ha_fr_xx.html) | [translate_ha_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ha_fr_xx.html) |
| [xx.fi.translate_to.ha](https://nlp.johnsnowlabs.com//2021/06/04/translate_ha_fi_xx.html) | [translate_ha_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ha_fi_xx.html) |
| [xx.sv.translate_to.ha](https://nlp.johnsnowlabs.com//2021/06/04/translate_ha_sv_xx.html) | [translate_ha_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ha_sv_xx.html) |
| [xx.pt.translate_to.gl](https://nlp.johnsnowlabs.com//2021/06/04/translate_gl_pt_xx.html) | [translate_gl_pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_gl_pt_xx.html) |
| [xx.fr.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_fr_xx.html) | [translate_guw_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_fr_xx.html) |
| [xx.es.translate_to.ht](https://nlp.johnsnowlabs.com//2021/06/04/translate_ht_es_xx.html) | [translate_ht_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ht_es_xx.html) |
| [xx.de.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_de_xx.html) | [translate_hu_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_de_xx.html) |
| [xx.sv.translate_to.ht](https://nlp.johnsnowlabs.com//2021/06/04/translate_ht_sv_xx.html) | [translate_ht_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ht_sv_xx.html) |
| [xx.es.translate_to.hr](https://nlp.johnsnowlabs.com//2021/06/04/translate_hr_es_xx.html) | [translate_hr_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_hr_es_xx.html) |
| [xx.fr.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_fr_xx.html) | [translate_gaa_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_fr_xx.html) |
| [xx.ru.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_ru_xx.html) | [translate_he_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_ru_xx.html) |
| [xx.es.translate_to.gl](https://nlp.johnsnowlabs.com//2021/06/04/translate_gl_es_xx.html) | [translate_gl_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_gl_es_xx.html) |
| [xx.ru.translate_to.hy](https://nlp.johnsnowlabs.com//2021/06/04/translate_hy_ru_xx.html) | [translate_hy_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_hy_ru_xx.html) |
| [xx.fi.translate_to.gil](https://nlp.johnsnowlabs.com//2021/06/04/translate_gil_fi_xx.html) | [translate_gil_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_gil_fi_xx.html) |
| [xx.sv.translate_to.hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_sv_xx.html) | [translate_hu_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_hu_sv_xx.html) |
| [xx.sv.translate_to.gil](https://nlp.johnsnowlabs.com//2021/06/04/translate_gil_sv_xx.html) | [translate_gil_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_gil_sv_xx.html) |
| [xx.fi.translate_to.fse](https://nlp.johnsnowlabs.com//2021/06/04/translate_fse_fi_xx.html) | [translate_fse_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_fse_fi_xx.html) |
| [xx.gem.translate_to.gem](https://nlp.johnsnowlabs.com//2021/06/04/translate_gem_gem_xx.html) | [translate_gem_gem](https://nlp.johnsnowlabs.com//2021/06/04/translate_gem_gem_xx.html) |
| [xx.es.translate_to.ha](https://nlp.johnsnowlabs.com//2021/06/04/translate_ha_es_xx.html) | [translate_ha_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ha_es_xx.html) |
| [xx.it.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_it_xx.html) | [translate_he_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_it_xx.html) |
| [xx.sv.translate_to.guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_sv_xx.html) | [translate_guw_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_guw_sv_xx.html) |
| [xx.sv.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_sv_xx.html) | [translate_he_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_sv_xx.html) |
| [xx.yap.translate_to.fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_yap_xx.html) | [translate_fr_yap](https://nlp.johnsnowlabs.com//2021/06/04/translate_fr_yap_xx.html) |
| [xx.fr.translate_to.hr](https://nlp.johnsnowlabs.com//2021/06/04/translate_hr_fr_xx.html) | [translate_hr_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_hr_fr_xx.html) |
| [xx.eo.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_eo_xx.html) | [translate_he_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_eo_xx.html) |
| [xx.es.translate_to.gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_es_xx.html) | [translate_gaa_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_gaa_es_xx.html) |
| [xx.fi.translate_to.hr](https://nlp.johnsnowlabs.com//2021/06/04/translate_hr_fi_xx.html) | [translate_hr_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_hr_fi_xx.html) |
| [xx.fr.translate_to.he](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_fr_xx.html) | [translate_he_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_he_fr_xx.html) |
| [xx.fi.translate_to.ilo](https://nlp.johnsnowlabs.com//2021/06/04/translate_ilo_fi_xx.html) | [translate_ilo_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ilo_fi_xx.html) |
| [xx.sv.translate_to.iso](https://nlp.johnsnowlabs.com//2021/06/04/translate_iso_sv_xx.html) | [translate_iso_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_iso_sv_xx.html) |
| [xx.he.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_he_xx.html) | [translate_ja_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_he_xx.html) |
| [xx.fi.translate_to.id](https://nlp.johnsnowlabs.com//2021/06/04/translate_id_fi_xx.html) | [translate_id_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_id_fi_xx.html) |
| [xx.de.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_de_xx.html) | [translate_ja_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_de_xx.html) |
| [xx.he.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_he_xx.html) | [translate_it_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_he_xx.html) |
| [xx.it.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_it_xx.html) | [translate_ja_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_it_xx.html) |
| [xx.is.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_is_xx.html) | [translate_it_is](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_is_xx.html) |
| [xx.bg.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_bg_xx.html) | [translate_ja_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_bg_xx.html) |
| [xx.de.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_de_xx.html) | [translate_ig_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_de_xx.html) |
| [xx.bg.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_bg_xx.html) | [translate_it_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_bg_xx.html) |
| [xx.es.translate_to.id](https://nlp.johnsnowlabs.com//2021/06/04/translate_id_es_xx.html) | [translate_id_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_id_es_xx.html) |
| [xx.fr.translate_to.id](https://nlp.johnsnowlabs.com//2021/06/04/translate_id_fr_xx.html) | [translate_id_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_id_fr_xx.html) |
| [xx.es.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_es_xx.html) | [translate_ja_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_es_xx.html) |
| [xx.sv.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_sv_xx.html) | [translate_ja_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_sv_xx.html) |
| [xx.es.translate_to.iso](https://nlp.johnsnowlabs.com//2021/06/04/translate_iso_es_xx.html) | [translate_iso_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_iso_es_xx.html) |
| [xx.es.translate_to.ilo](https://nlp.johnsnowlabs.com//2021/06/04/translate_ilo_es_xx.html) | [translate_ilo_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ilo_es_xx.html) |
| [xx.it.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_it_xx.html) | [translate_is_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_it_xx.html) |
| [xx.sv.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_sv_xx.html) | [translate_it_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_sv_xx.html) |
| [xx.sv.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_sv_xx.html) | [translate_is_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_sv_xx.html) |
| [xx.ru.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_ru_xx.html) | [translate_ja_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_ru_xx.html) |
| [xx.es.translate_to.kg](https://nlp.johnsnowlabs.com//2021/06/04/translate_kg_es_xx.html) | [translate_kg_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_kg_es_xx.html) |
| [xx.fi.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_fi_xx.html) | [translate_ig_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_fi_xx.html) |
| [xx.fr.translate_to.iso](https://nlp.johnsnowlabs.com//2021/06/04/translate_iso_fr_xx.html) | [translate_iso_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_iso_fr_xx.html) |
| [xx.de.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_de_xx.html) | [translate_ko_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_de_xx.html) |
| [xx.sv.translate_to.ilo](https://nlp.johnsnowlabs.com//2021/06/04/translate_ilo_sv_xx.html) | [translate_ilo_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ilo_sv_xx.html) |
| [xx.es.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_es_xx.html) | [translate_is_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_es_xx.html) |
| [xx.da.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_da_xx.html) | [translate_ja_da](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_da_xx.html) |
| [xx.nl.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_nl_xx.html) | [translate_ja_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_nl_xx.html) |
| [xx.inc.translate_to.inc](https://nlp.johnsnowlabs.com//2021/06/04/translate_inc_inc_xx.html) | [translate_inc_inc](https://nlp.johnsnowlabs.com//2021/06/04/translate_inc_inc_xx.html) |
| [xx.de.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_de_xx.html) | [translate_is_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_de_xx.html) |
| [xx.fr.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_fr_xx.html) | [translate_is_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_fr_xx.html) |
| [xx.lt.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_lt_xx.html) | [translate_it_lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_lt_xx.html) |
| [xx.sv.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_sv_xx.html) | [translate_ig_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_sv_xx.html) |
| [xx.de.translate_to.ilo](https://nlp.johnsnowlabs.com//2021/06/04/translate_ilo_de_xx.html) | [translate_ilo_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ilo_de_xx.html) |
| [xx.ar.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_ar_xx.html) | [translate_it_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_ar_xx.html) |
| [xx.fr.translate_to.kg](https://nlp.johnsnowlabs.com//2021/06/04/translate_kg_fr_xx.html) | [translate_kg_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_kg_fr_xx.html) |
| [xx.vi.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_vi_xx.html) | [translate_ja_vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_vi_xx.html) |
| [xx.ru.translate_to.ka](https://nlp.johnsnowlabs.com//2021/06/04/translate_ka_ru_xx.html) | [translate_ka_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ka_ru_xx.html) |
| [xx.uk.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_uk_xx.html) | [translate_it_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_uk_xx.html) |
| [xx.vi.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_vi_xx.html) | [translate_it_vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_vi_xx.html) |
| [xx.ms.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_ms_xx.html) | [translate_it_ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_ms_xx.html) |
| [xx.ar.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_ar_xx.html) | [translate_ja_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_ar_xx.html) |
| [xx.eo.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_eo_xx.html) | [translate_is_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_eo_xx.html) |
| [xx.ca.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_ca_xx.html) | [translate_it_ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_ca_xx.html) |
| [xx.sh.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_sh_xx.html) | [translate_ja_sh](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_sh_xx.html) |
| [xx.fi.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_fi_xx.html) | [translate_ja_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_fi_xx.html) |
| [xx.iir.translate_to.iir](https://nlp.johnsnowlabs.com//2021/06/04/translate_iir_iir_xx.html) | [translate_iir_iir](https://nlp.johnsnowlabs.com//2021/06/04/translate_iir_iir_xx.html) |
| [xx.itc.translate_to.itc](https://nlp.johnsnowlabs.com//2021/06/04/translate_itc_itc_xx.html) | [translate_itc_itc](https://nlp.johnsnowlabs.com//2021/06/04/translate_itc_itc_xx.html) |
| [xx.ms.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_ms_xx.html) | [translate_ja_ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_ms_xx.html) |
| [xx.fr.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_fr_xx.html) | [translate_it_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_fr_xx.html) |
| [xx.fr.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_fr_xx.html) | [translate_ja_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_fr_xx.html) |
| [xx.pt.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_pt_xx.html) | [translate_ja_pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_pt_xx.html) |
| [xx.eo.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_eo_xx.html) | [translate_it_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_eo_xx.html) |
| [xx.fi.translate_to.iso](https://nlp.johnsnowlabs.com//2021/06/04/translate_iso_fi_xx.html) | [translate_iso_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_iso_fi_xx.html) |
| [xx.pl.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_pl_xx.html) | [translate_ja_pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_pl_xx.html) |
| [xx.tr.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_tr_xx.html) | [translate_ja_tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_tr_xx.html) |
| [xx.es.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_es_xx.html) | [translate_ig_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_es_xx.html) |
| [xx.fr.translate_to.ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_fr_xx.html) | [translate_ig_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ig_fr_xx.html) |
| [xx.sv.translate_to.id](https://nlp.johnsnowlabs.com//2021/06/04/translate_id_sv_xx.html) | [translate_id_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_id_sv_xx.html) |
| [xx.hu.translate_to.ja](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_hu_xx.html) | [translate_ja_hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_ja_hu_xx.html) |
| [xx.sv.translate_to.kg](https://nlp.johnsnowlabs.com//2021/06/04/translate_kg_sv_xx.html) | [translate_kg_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_kg_sv_xx.html) |
| [xx.es.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_es_xx.html) | [translate_it_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_es_xx.html) |
| [xx.ine.translate_to.ine](https://nlp.johnsnowlabs.com//2021/06/04/translate_ine_ine_xx.html) | [translate_ine_ine](https://nlp.johnsnowlabs.com//2021/06/04/translate_ine_ine_xx.html) |
| [xx.de.translate_to.it](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_de_xx.html) | [translate_it_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_it_de_xx.html) |
| [xx.fi.translate_to.is](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_fi_xx.html) | [translate_is_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_is_fi_xx.html) |
| [xx.es.translate_to.mk](https://nlp.johnsnowlabs.com//2021/06/04/translate_mk_es_xx.html) | [translate_mk_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_mk_es_xx.html) |
| [xx.es.translate_to.lue](https://nlp.johnsnowlabs.com//2021/06/04/translate_lue_es_xx.html) | [translate_lue_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_lue_es_xx.html) |
| [xx.es.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_es_xx.html) | [translate_lv_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_es_xx.html) |
| [xx.fi.translate_to.lue](https://nlp.johnsnowlabs.com//2021/06/04/translate_lue_fi_xx.html) | [translate_lue_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_lue_fi_xx.html) |
| [xx.es.translate_to.ln](https://nlp.johnsnowlabs.com//2021/06/04/translate_ln_es_xx.html) | [translate_ln_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ln_es_xx.html) |
| [xx.fr.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_fr_xx.html) | [translate_loz_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_fr_xx.html) |
| [xx.sv.translate_to.kwy](https://nlp.johnsnowlabs.com//2021/06/04/translate_kwy_sv_xx.html) | [translate_kwy_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_kwy_sv_xx.html) |
| [xx.es.translate_to.lus](https://nlp.johnsnowlabs.com//2021/06/04/translate_lus_es_xx.html) | [translate_lus_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_lus_es_xx.html) |
| [xx.fr.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_fr_xx.html) | [translate_lv_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_fr_xx.html) |
| [xx.fr.translate_to.lu](https://nlp.johnsnowlabs.com//2021/06/04/translate_lu_fr_xx.html) | [translate_lu_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_lu_fr_xx.html) |
| [xx.de.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_de_xx.html) | [translate_lt_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_de_xx.html) |
| [xx.tr.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_tr_xx.html) | [translate_lt_tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_tr_xx.html) |
| [xx.fr.translate_to.lus](https://nlp.johnsnowlabs.com//2021/06/04/translate_lus_fr_xx.html) | [translate_lus_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_lus_fr_xx.html) |
| [xx.es.translate_to.mg](https://nlp.johnsnowlabs.com//2021/06/04/translate_mg_es_xx.html) | [translate_mg_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_mg_es_xx.html) |
| [xx.sv.translate_to.lua](https://nlp.johnsnowlabs.com//2021/06/04/translate_lua_sv_xx.html) | [translate_lua_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lua_sv_xx.html) |
| [xx.fr.translate_to.lg](https://nlp.johnsnowlabs.com//2021/06/04/translate_lg_fr_xx.html) | [translate_lg_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_lg_fr_xx.html) |
| [xx.fr.translate_to.kwy](https://nlp.johnsnowlabs.com//2021/06/04/translate_kwy_fr_xx.html) | [translate_kwy_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_kwy_fr_xx.html) |
| [xx.es.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_es_xx.html) | [translate_lt_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_es_xx.html) |
| [xx.sv.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_sv_xx.html) | [translate_ko_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_sv_xx.html) |
| [xx.es.translate_to.kqn](https://nlp.johnsnowlabs.com//2021/06/04/translate_kqn_es_xx.html) | [translate_kqn_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_kqn_es_xx.html) |
| [xx.fr.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_fr_xx.html) | [translate_ko_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_fr_xx.html) |
| [xx.sv.translate_to.kqn](https://nlp.johnsnowlabs.com//2021/06/04/translate_kqn_sv_xx.html) | [translate_kqn_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_kqn_sv_xx.html) |
| [xx.fi.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_fi_xx.html) | [translate_ko_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_fi_xx.html) |
| [xx.es.translate_to.mh](https://nlp.johnsnowlabs.com//2021/06/04/translate_mh_es_xx.html) | [translate_mh_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_mh_es_xx.html) |
| [xx.fr.translate_to.lua](https://nlp.johnsnowlabs.com//2021/06/04/translate_lua_fr_xx.html) | [translate_lua_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_lua_fr_xx.html) |
| [xx.it.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_it_xx.html) | [translate_lt_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_it_xx.html) |
| [xx.sv.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_sv_xx.html) | [translate_lt_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_sv_xx.html) |
| [xx.es.translate_to.lu](https://nlp.johnsnowlabs.com//2021/06/04/translate_lu_es_xx.html) | [translate_lu_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_lu_es_xx.html) |
| [xx.fi.translate_to.lua](https://nlp.johnsnowlabs.com//2021/06/04/translate_lua_fi_xx.html) | [translate_lua_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_lua_fi_xx.html) |
| [xx.fr.translate_to.kqn](https://nlp.johnsnowlabs.com//2021/06/04/translate_kqn_fr_xx.html) | [translate_kqn_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_kqn_fr_xx.html) |
| [xx.de.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_de_xx.html) | [translate_loz_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_de_xx.html) |
| [xx.fr.translate_to.ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_ms_fr_xx.html) | [translate_ms_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ms_fr_xx.html) |
| [xx.fr.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_fr_xx.html) | [translate_lt_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_fr_xx.html) |
| [xx.ru.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_ru_xx.html) | [translate_lv_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_ru_xx.html) |
| [xx.ms.translate_to.ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_ms_ms_xx.html) | [translate_ms_ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_ms_ms_xx.html) |
| [xx.sv.translate_to.lus](https://nlp.johnsnowlabs.com//2021/06/04/translate_lus_sv_xx.html) | [translate_lus_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lus_sv_xx.html) |
| [xx.fr.translate_to.lue](https://nlp.johnsnowlabs.com//2021/06/04/translate_lue_fr_xx.html) | [translate_lue_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_lue_fr_xx.html) |
| [xx.fi.translate_to.lu](https://nlp.johnsnowlabs.com//2021/06/04/translate_lu_fi_xx.html) | [translate_lu_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_lu_fi_xx.html) |
| [xx.eo.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_eo_xx.html) | [translate_lt_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_eo_xx.html) |
| [xx.fi.translate_to.mk](https://nlp.johnsnowlabs.com//2021/06/04/translate_mk_fi_xx.html) | [translate_mk_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_mk_fi_xx.html) |
| [xx.es.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_es_xx.html) | [translate_ko_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_es_xx.html) |
| [xx.sv.translate_to.lue](https://nlp.johnsnowlabs.com//2021/06/04/translate_lue_sv_xx.html) | [translate_lue_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lue_sv_xx.html) |
| [xx.pl.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_pl_xx.html) | [translate_lt_pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_pl_xx.html) |
| [xx.es.translate_to.mfe](https://nlp.johnsnowlabs.com//2021/06/04/translate_mfe_es_xx.html) | [translate_mfe_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_mfe_es_xx.html) |
| [xx.fi.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_fi_xx.html) | [translate_loz_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_fi_xx.html) |
| [xx.sv.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_sv_xx.html) | [translate_loz_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_sv_xx.html) |
| [xx.ru.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_ru_xx.html) | [translate_ko_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_ru_xx.html) |
| [xx.fi.translate_to.lg](https://nlp.johnsnowlabs.com//2021/06/04/translate_lg_fi_xx.html) | [translate_lg_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_lg_fi_xx.html) |
| [xx.fi.translate_to.mh](https://nlp.johnsnowlabs.com//2021/06/04/translate_mh_fi_xx.html) | [translate_mh_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_mh_fi_xx.html) |
| [xx.sv.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_sv_xx.html) | [translate_lv_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_sv_xx.html) |
| [xx.hu.translate_to.ko](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_hu_xx.html) | [translate_ko_hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_ko_hu_xx.html) |
| [xx.es.translate_to.lua](https://nlp.johnsnowlabs.com//2021/06/04/translate_lua_es_xx.html) | [translate_lua_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_lua_es_xx.html) |
| [xx.fi.translate_to.lv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_fi_xx.html) | [translate_lv_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_lv_fi_xx.html) |
| [xx.ru.translate_to.lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_ru_xx.html) | [translate_lt_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_lt_ru_xx.html) |
| [xx.de.translate_to.ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_ms_de_xx.html) | [translate_ms_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ms_de_xx.html) |
| [xx.fi.translate_to.lus](https://nlp.johnsnowlabs.com//2021/06/04/translate_lus_fi_xx.html) | [translate_lus_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_lus_fi_xx.html) |
| [xx.es.translate_to.lg](https://nlp.johnsnowlabs.com//2021/06/04/translate_lg_es_xx.html) | [translate_lg_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_lg_es_xx.html) |
| [xx.de.translate_to.ln](https://nlp.johnsnowlabs.com//2021/06/04/translate_ln_de_xx.html) | [translate_ln_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ln_de_xx.html) |
| [xx.es.translate_to.mfs](https://nlp.johnsnowlabs.com//2021/06/04/translate_mfs_es_xx.html) | [translate_mfs_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_mfs_es_xx.html) |
| [xx.fr.translate_to.mk](https://nlp.johnsnowlabs.com//2021/06/04/translate_mk_fr_xx.html) | [translate_mk_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_mk_fr_xx.html) |
| [xx.fr.translate_to.ln](https://nlp.johnsnowlabs.com//2021/06/04/translate_ln_fr_xx.html) | [translate_ln_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ln_fr_xx.html) |
| [xx.es.translate_to.loz](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_es_xx.html) | [translate_loz_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_loz_es_xx.html) |
| [xx.sv.translate_to.lu](https://nlp.johnsnowlabs.com//2021/06/04/translate_lu_sv_xx.html) | [translate_lu_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lu_sv_xx.html) |
| [xx.it.translate_to.ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_ms_it_xx.html) | [translate_ms_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_ms_it_xx.html) |
| [xx.sv.translate_to.lg](https://nlp.johnsnowlabs.com//2021/06/04/translate_lg_sv_xx.html) | [translate_lg_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_lg_sv_xx.html) |
| [xx.ar.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_ar_xx.html) | [translate_pl_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_ar_xx.html) |
| [xx.fr.translate_to.ro](https://nlp.johnsnowlabs.com//2021/06/04/translate_ro_fr_xx.html) | [translate_ro_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ro_fr_xx.html) |
| [xx.sv.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_sv_xx.html) | [translate_niu_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_sv_xx.html) |
| [xx.eo.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_eo_xx.html) | [translate_pl_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_eo_xx.html) |
| [xx.nl.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_nl_xx.html) | [translate_no_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_nl_xx.html) |
| [xx.es.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_es_xx.html) | [translate_no_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_es_xx.html) |
| [xx.es.translate_to.pag](https://nlp.johnsnowlabs.com//2021/06/04/translate_pag_es_xx.html) | [translate_pag_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_pag_es_xx.html) |
| [xx.ru.translate_to.rn](https://nlp.johnsnowlabs.com//2021/06/04/translate_rn_ru_xx.html) | [translate_rn_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_rn_ru_xx.html) |
| [xx.sv.translate_to.pag](https://nlp.johnsnowlabs.com//2021/06/04/translate_pag_sv_xx.html) | [translate_pag_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_pag_sv_xx.html) |
| [xx.uk.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_uk_xx.html) | [translate_pt_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_uk_xx.html) |
| [xx.uk.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_uk_xx.html) | [translate_pl_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_uk_xx.html) |
| [xx.de.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_de_xx.html) | [translate_pl_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_de_xx.html) |
| [xx.sv.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_sv_xx.html) | [translate_nl_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_sv_xx.html) |
| [xx.fr.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_fr_xx.html) | [translate_no_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_fr_xx.html) |
| [xx.es.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_es_xx.html) | [translate_niu_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_es_xx.html) |
| [xx.uk.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_uk_xx.html) | [translate_no_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_uk_xx.html) |
| [xx.lt.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_lt_xx.html) | [translate_pl_lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_lt_xx.html) |
| [xx.tl.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_tl_xx.html) | [translate_pt_tl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_tl_xx.html) |
| [xx.gl.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_gl_xx.html) | [translate_pt_gl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_gl_xx.html) |
| [xx.da.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_da_xx.html) | [translate_ru_da](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_da_xx.html) |
| [xx.da.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_da_xx.html) | [translate_no_da](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_da_xx.html) |
| [xx.uk.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_uk_xx.html) | [translate_nl_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_uk_xx.html) |
| [xx.sv.translate_to.pon](https://nlp.johnsnowlabs.com//2021/06/04/translate_pon_sv_xx.html) | [translate_pon_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_pon_sv_xx.html) |
| [xx.fr.translate_to.pis](https://nlp.johnsnowlabs.com//2021/06/04/translate_pis_fr_xx.html) | [translate_pis_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_pis_fr_xx.html) |
| [xx.fr.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_fr_xx.html) | [translate_niu_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_fr_xx.html) |
| [xx.af.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_af_xx.html) | [translate_nl_af](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_af_xx.html) |
| [xx.fi.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_fi_xx.html) | [translate_nso_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_fi_xx.html) |
| [xx.fi.translate_to.pon](https://nlp.johnsnowlabs.com//2021/06/04/translate_pon_fi_xx.html) | [translate_pon_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_pon_fi_xx.html) |
| [xx.de.translate_to.pap](https://nlp.johnsnowlabs.com//2021/06/04/translate_pap_de_xx.html) | [translate_pap_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_pap_de_xx.html) |
| [xx.de.translate_to.rn](https://nlp.johnsnowlabs.com//2021/06/04/translate_rn_de_xx.html) | [translate_rn_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_rn_de_xx.html) |
| [xx.es.translate_to.pon](https://nlp.johnsnowlabs.com//2021/06/04/translate_pon_es_xx.html) | [translate_pon_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_pon_es_xx.html) |
| [xx.es.translate_to.pis](https://nlp.johnsnowlabs.com//2021/06/04/translate_pis_es_xx.html) | [translate_pis_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_pis_es_xx.html) |
| [xx.ca.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_ca_xx.html) | [translate_pt_ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_ca_xx.html) |
| [xx.sv.translate_to.rnd](https://nlp.johnsnowlabs.com//2021/06/04/translate_rnd_sv_xx.html) | [translate_rnd_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_rnd_sv_xx.html) |
| [xx.sv.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_sv_xx.html) | [translate_pl_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_sv_xx.html) |
| [xx.ru.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_ru_xx.html) | [translate_no_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_ru_xx.html) |
| [xx.fi.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_fi_xx.html) | [translate_niu_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_fi_xx.html) |
| [xx.de.translate_to.pag](https://nlp.johnsnowlabs.com//2021/06/04/translate_pag_de_xx.html) | [translate_pag_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_pag_de_xx.html) |
| [xx.fr.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_fr_xx.html) | [translate_pl_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_fr_xx.html) |
| [xx.fi.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_fi_xx.html) | [translate_no_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_fi_xx.html) |
| [xx.pl.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_pl_xx.html) | [translate_no_pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_pl_xx.html) |
| [xx.de.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_de_xx.html) | [translate_nso_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_de_xx.html) |
| [xx.fr.translate_to.rn](https://nlp.johnsnowlabs.com//2021/06/04/translate_rn_fr_xx.html) | [translate_rn_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_rn_fr_xx.html) |
| [xx.sv.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_sv_xx.html) | [translate_nso_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_sv_xx.html) |
| [xx.sv.translate_to.ro](https://nlp.johnsnowlabs.com//2021/06/04/translate_ro_sv_xx.html) | [translate_ro_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ro_sv_xx.html) |
| [xx.no.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_no_xx.html) | [translate_pl_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_no_xx.html) |
| [xx.fr.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_fr_xx.html) | [translate_nl_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_fr_xx.html) |
| [xx.es.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_es_xx.html) | [translate_nso_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_es_xx.html) |
| [xx.no.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_no_xx.html) | [translate_nl_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_no_xx.html) |
| [xx.fi.translate_to.pis](https://nlp.johnsnowlabs.com//2021/06/04/translate_pis_fi_xx.html) | [translate_pis_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_pis_fi_xx.html) |
| [xx.ca.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_ca_xx.html) | [translate_nl_ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_ca_xx.html) |
| [xx.es.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_es_xx.html) | [translate_nl_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_es_xx.html) |
| [xx.es.translate_to.ny](https://nlp.johnsnowlabs.com//2021/06/04/translate_ny_es_xx.html) | [translate_ny_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ny_es_xx.html) |
| [xx.fr.translate_to.pap](https://nlp.johnsnowlabs.com//2021/06/04/translate_pap_fr_xx.html) | [translate_pap_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_pap_fr_xx.html) |
| [xx.fi.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_fi_xx.html) | [translate_nl_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_fi_xx.html) |
| [xx.sv.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_sv_xx.html) | [translate_no_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_sv_xx.html) |
| [xx.fr.translate_to.pon](https://nlp.johnsnowlabs.com//2021/06/04/translate_pon_fr_xx.html) | [translate_pon_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_pon_fr_xx.html) |
| [xx.fr.translate_to.rnd](https://nlp.johnsnowlabs.com//2021/06/04/translate_rnd_fr_xx.html) | [translate_rnd_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_rnd_fr_xx.html) |
| [xx.es.translate_to.pap](https://nlp.johnsnowlabs.com//2021/06/04/translate_pap_es_xx.html) | [translate_pap_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_pap_es_xx.html) |
| [xx.es.translate_to.prl](https://nlp.johnsnowlabs.com//2021/06/04/translate_prl_es_xx.html) | [translate_prl_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_prl_es_xx.html) |
| [xx.eo.translate_to.ro](https://nlp.johnsnowlabs.com//2021/06/04/translate_ro_eo_xx.html) | [translate_ro_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_ro_eo_xx.html) |
| [xx.sv.translate_to.pis](https://nlp.johnsnowlabs.com//2021/06/04/translate_pis_sv_xx.html) | [translate_pis_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_pis_sv_xx.html) |
| [xx.af.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_af_xx.html) | [translate_ru_af](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_af_xx.html) |
| [xx.fr.translate_to.nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_fr_xx.html) | [translate_nso_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_nso_fr_xx.html) |
| [xx.eo.translate_to.pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_eo_xx.html) | [translate_pt_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_pt_eo_xx.html) |
| [xx.ar.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_ar_xx.html) | [translate_ru_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_ar_xx.html) |
| [xx.fr.translate_to.mt](https://nlp.johnsnowlabs.com//2021/06/04/translate_mt_fr_xx.html) | [translate_mt_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_mt_fr_xx.html) |
| [xx.es.translate_to.rn](https://nlp.johnsnowlabs.com//2021/06/04/translate_rn_es_xx.html) | [translate_rn_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_rn_es_xx.html) |
| [xx.sv.translate_to.mt](https://nlp.johnsnowlabs.com//2021/06/04/translate_mt_sv_xx.html) | [translate_mt_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_mt_sv_xx.html) |
| [xx.de.translate_to.niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_de_xx.html) | [translate_niu_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_niu_de_xx.html) |
| [xx.es.translate_to.mt](https://nlp.johnsnowlabs.com//2021/06/04/translate_mt_es_xx.html) | [translate_mt_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_mt_es_xx.html) |
| [xx.es.translate_to.pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_es_xx.html) | [translate_pl_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_pl_es_xx.html) |
| [xx.fi.translate_to.pag](https://nlp.johnsnowlabs.com//2021/06/04/translate_pag_fi_xx.html) | [translate_pag_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_pag_fi_xx.html) |
| [xx.de.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_de_xx.html) | [translate_no_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_de_xx.html) |
| [xx.de.translate_to.ny](https://nlp.johnsnowlabs.com//2021/06/04/translate_ny_de_xx.html) | [translate_ny_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_ny_de_xx.html) |
| [xx.fi.translate_to.mt](https://nlp.johnsnowlabs.com//2021/06/04/translate_mt_fi_xx.html) | [translate_mt_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_mt_fi_xx.html) |
| [xx.no.translate_to.no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_no_xx.html) | [translate_no_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_no_no_xx.html) |
| [xx.eo.translate_to.nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_eo_xx.html) | [translate_nl_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_nl_eo_xx.html) |
| [xx.bg.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_bg_xx.html) | [translate_ru_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_bg_xx.html) |
| [xx.fi.translate_to.pap](https://nlp.johnsnowlabs.com//2021/06/04/translate_pap_fi_xx.html) | [translate_pap_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_pap_fi_xx.html) |
| [xx.fi.translate_to.ro](https://nlp.johnsnowlabs.com//2021/06/04/translate_ro_fi_xx.html) | [translate_ro_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ro_fi_xx.html) |
| [xx.sv.translate_to.st](https://nlp.johnsnowlabs.com//2021/06/04/translate_st_sv_xx.html) | [translate_st_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_st_sv_xx.html) |
| [xx.kg.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_kg_xx.html) | [translate_sv_kg](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_kg_xx.html) |
| [xx.sv.translate_to.sq](https://nlp.johnsnowlabs.com//2021/06/04/translate_sq_sv_xx.html) | [translate_sq_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sq_sv_xx.html) |
| [xx.ee.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ee_xx.html) | [translate_sv_ee](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ee_xx.html) |
| [xx.es.translate_to.srn](https://nlp.johnsnowlabs.com//2021/06/04/translate_srn_es_xx.html) | [translate_srn_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_srn_es_xx.html) |
| [xx.lv.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_lv_xx.html) | [translate_ru_lv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_lv_xx.html) |
| [xx.cs.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_cs_xx.html) | [translate_sv_cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_cs_xx.html) |
| [xx.ha.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ha_xx.html) | [translate_sv_ha](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ha_xx.html) |
| [xx.kqn.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_kqn_xx.html) | [translate_sv_kqn](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_kqn_xx.html) |
| [xx.fr.translate_to.rw](https://nlp.johnsnowlabs.com//2021/06/04/translate_rw_fr_xx.html) | [translate_rw_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_rw_fr_xx.html) |
| [xx.fr.translate_to.sn](https://nlp.johnsnowlabs.com//2021/06/04/translate_sn_fr_xx.html) | [translate_sn_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_sn_fr_xx.html) |
| [xx.eu.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_eu_xx.html) | [translate_ru_eu](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_eu_xx.html) |
| [xx.fi.translate_to.st](https://nlp.johnsnowlabs.com//2021/06/04/translate_st_fi_xx.html) | [translate_st_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_st_fi_xx.html) |
| [xx.efi.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_efi_xx.html) | [translate_sv_efi](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_efi_xx.html) |
| [xx.ho.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ho_xx.html) | [translate_sv_ho](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ho_xx.html) |
| [xx.id.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_id_xx.html) | [translate_sv_id](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_id_xx.html) |
| [xx.eo.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_eo_xx.html) | [translate_sv_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_eo_xx.html) |
| [xx.guw.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_guw_xx.html) | [translate_sv_guw](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_guw_xx.html) |
| [xx.sv.translate_to.sk](https://nlp.johnsnowlabs.com//2021/06/04/translate_sk_sv_xx.html) | [translate_sk_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sk_sv_xx.html) |
| [xx.fr.translate_to.srn](https://nlp.johnsnowlabs.com//2021/06/04/translate_srn_fr_xx.html) | [translate_srn_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_srn_fr_xx.html) |
| [xx.ceb.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ceb_xx.html) | [translate_sv_ceb](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ceb_xx.html) |
| [xx.es.translate_to.sq](https://nlp.johnsnowlabs.com//2021/06/04/translate_sq_es_xx.html) | [translate_sq_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_sq_es_xx.html) |
| [xx.sv.translate_to.rw](https://nlp.johnsnowlabs.com//2021/06/04/translate_rw_sv_xx.html) | [translate_rw_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_rw_sv_xx.html) |
| [xx.is.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_is_xx.html) | [translate_sv_is](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_is_xx.html) |
| [xx.es.translate_to.sm](https://nlp.johnsnowlabs.com//2021/06/04/translate_sm_es_xx.html) | [translate_sm_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_sm_es_xx.html) |
| [xx.bcl.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bcl_xx.html) | [translate_sv_bcl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bcl_xx.html) |
| [xx.kwy.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_kwy_xx.html) | [translate_sv_kwy](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_kwy_xx.html) |
| [xx.es.translate_to.run](https://nlp.johnsnowlabs.com//2021/06/04/translate_run_es_xx.html) | [translate_run_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_run_es_xx.html) |
| [xx.el.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_el_xx.html) | [translate_sv_el](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_el_xx.html) |
| [xx.es.translate_to.sk](https://nlp.johnsnowlabs.com//2021/06/04/translate_sk_es_xx.html) | [translate_sk_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_sk_es_xx.html) |
| [xx.iso.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_iso_xx.html) | [translate_sv_iso](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_iso_xx.html) |
| [xx.lu.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lu_xx.html) | [translate_sv_lu](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lu_xx.html) |
| [xx.af.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_af_xx.html) | [translate_sv_af](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_af_xx.html) |
| [xx.bg.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bg_xx.html) | [translate_sv_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bg_xx.html) |
| [xx.fr.translate_to.sm](https://nlp.johnsnowlabs.com//2021/06/04/translate_sm_fr_xx.html) | [translate_sm_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_sm_fr_xx.html) |
| [xx.hr.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_hr_xx.html) | [translate_sv_hr](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_hr_xx.html) |
| [xx.sv.translate_to.sn](https://nlp.johnsnowlabs.com//2021/06/04/translate_sn_sv_xx.html) | [translate_sn_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sn_sv_xx.html) |
| [xx.no.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_no_xx.html) | [translate_ru_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_no_xx.html) |
| [xx.fr.translate_to.sg](https://nlp.johnsnowlabs.com//2021/06/04/translate_sg_fr_xx.html) | [translate_sg_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_sg_fr_xx.html) |
| [xx.es.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_es_xx.html) | [translate_sl_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_es_xx.html) |
| [xx.bzs.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bzs_xx.html) | [translate_sv_bzs](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bzs_xx.html) |
| [xx.fr.translate_to.st](https://nlp.johnsnowlabs.com//2021/06/04/translate_st_fr_xx.html) | [translate_st_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_st_fr_xx.html) |
| [xx.hu.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_hu_xx.html) | [translate_sv_hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_hu_xx.html) |
| [xx.sv.translate_to.sg](https://nlp.johnsnowlabs.com//2021/06/04/translate_sg_sv_xx.html) | [translate_sg_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sg_sv_xx.html) |
| [xx.sem.translate_to.sem](https://nlp.johnsnowlabs.com//2021/06/04/translate_sem_sem_xx.html) | [translate_sem_sem](https://nlp.johnsnowlabs.com//2021/06/04/translate_sem_sem_xx.html) |
| [xx.uk.translate_to.sh](https://nlp.johnsnowlabs.com//2021/06/04/translate_sh_uk_xx.html) | [translate_sh_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_sh_uk_xx.html) |
| [xx.ln.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ln_xx.html) | [translate_sv_ln](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ln_xx.html) |
| [xx.fi.translate_to.sk](https://nlp.johnsnowlabs.com//2021/06/04/translate_sk_fi_xx.html) | [translate_sk_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_sk_fi_xx.html) |
| [xx.ht.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ht_xx.html) | [translate_sv_ht](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ht_xx.html) |
| [xx.es.translate_to.st](https://nlp.johnsnowlabs.com//2021/06/04/translate_st_es_xx.html) | [translate_st_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_st_es_xx.html) |
| [xx.fr.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_fr_xx.html) | [translate_ru_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_fr_xx.html) |
| [xx.chk.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_chk_xx.html) | [translate_sv_chk](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_chk_xx.html) |
| [xx.fr.translate_to.sk](https://nlp.johnsnowlabs.com//2021/06/04/translate_sk_fr_xx.html) | [translate_sk_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_sk_fr_xx.html) |
| [xx.lg.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lg_xx.html) | [translate_sv_lg](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lg_xx.html) |
| [xx.sv.translate_to.srn](https://nlp.johnsnowlabs.com//2021/06/04/translate_srn_sv_xx.html) | [translate_srn_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_srn_sv_xx.html) |
| [xx.crs.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_crs_xx.html) | [translate_sv_crs](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_crs_xx.html) |
| [xx.uk.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_uk_xx.html) | [translate_ru_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_uk_xx.html) |
| [xx.et.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_et_xx.html) | [translate_ru_et](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_et_xx.html) |
| [xx.et.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_et_xx.html) | [translate_sv_et](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_et_xx.html) |
| [xx.es.translate_to.rw](https://nlp.johnsnowlabs.com//2021/06/04/translate_rw_es_xx.html) | [translate_rw_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_rw_es_xx.html) |
| [xx.sla.translate_to.sla](https://nlp.johnsnowlabs.com//2021/06/04/translate_sla_sla_xx.html) | [translate_sla_sla](https://nlp.johnsnowlabs.com//2021/06/04/translate_sla_sla_xx.html) |
| [xx.ru.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_ru_xx.html) | [translate_sl_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_ru_xx.html) |
| [xx.fj.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_fj_xx.html) | [translate_sv_fj](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_fj_xx.html) |
| [xx.es.translate_to.sn](https://nlp.johnsnowlabs.com//2021/06/04/translate_sn_es_xx.html) | [translate_sn_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_sn_es_xx.html) |
| [xx.lua.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lua_xx.html) | [translate_sv_lua](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lua_xx.html) |
| [xx.hil.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_hil_xx.html) | [translate_sv_hil](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_hil_xx.html) |
| [xx.es.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_es_xx.html) | [translate_ru_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_es_xx.html) |
| [xx.lue.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lue_xx.html) | [translate_sv_lue](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lue_xx.html) |
| [xx.gaa.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_gaa_xx.html) | [translate_sv_gaa](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_gaa_xx.html) |
| [xx.hy.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_hy_xx.html) | [translate_ru_hy](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_hy_xx.html) |
| [xx.bem.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bem_xx.html) | [translate_sv_bem](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bem_xx.html) |
| [xx.sv.translate_to.run](https://nlp.johnsnowlabs.com//2021/06/04/translate_run_sv_xx.html) | [translate_run_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_run_sv_xx.html) |
| [xx.gil.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_gil_xx.html) | [translate_sv_gil](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_gil_xx.html) |
| [xx.lus.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lus_xx.html) | [translate_sv_lus](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lus_xx.html) |
| [xx.he.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_he_xx.html) | [translate_ru_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_he_xx.html) |
| [xx.vi.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_vi_xx.html) | [translate_ru_vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_vi_xx.html) |
| [xx.he.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_he_xx.html) | [translate_sv_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_he_xx.html) |
| [xx.sv.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_sv_xx.html) | [translate_ru_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_sv_xx.html) |
| [xx.fi.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_fi_xx.html) | [translate_ru_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_fi_xx.html) |
| [xx.es.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_es_xx.html) | [translate_sv_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_es_xx.html) |
| [xx.es.translate_to.sg](https://nlp.johnsnowlabs.com//2021/06/04/translate_sg_es_xx.html) | [translate_sg_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_sg_es_xx.html) |
| [xx.eo.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_eo_xx.html) | [translate_ru_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_eo_xx.html) |
| [xx.lv.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lv_xx.html) | [translate_sv_lv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_lv_xx.html) |
| [xx.fi.translate_to.sg](https://nlp.johnsnowlabs.com//2021/06/04/translate_sg_fi_xx.html) | [translate_sg_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_sg_fi_xx.html) |
| [xx.es.translate_to.ssp](https://nlp.johnsnowlabs.com//2021/06/04/translate_ssp_es_xx.html) | [translate_ssp_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ssp_es_xx.html) |
| [xx.ilo.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ilo_xx.html) | [translate_sv_ilo](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ilo_xx.html) |
| [xx.fi.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_fi_xx.html) | [translate_sv_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_fi_xx.html) |
| [xx.lt.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_lt_xx.html) | [translate_ru_lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_lt_xx.html) |
| [xx.bi.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bi_xx.html) | [translate_sv_bi](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_bi_xx.html) |
| [xx.sv.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_sv_xx.html) | [translate_sl_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_sv_xx.html) |
| [xx.fr.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_fr_xx.html) | [translate_sv_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_fr_xx.html) |
| [xx.uk.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_uk_xx.html) | [translate_sl_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_uk_xx.html) |
| [xx.fi.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_fi_xx.html) | [translate_sl_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_fi_xx.html) |
| [xx.sl.translate_to.ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_sl_xx.html) | [translate_ru_sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_ru_sl_xx.html) |
| [xx.ig.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ig_xx.html) | [translate_sv_ig](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ig_xx.html) |
| [xx.ase.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ase_xx.html) | [translate_sv_ase](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ase_xx.html) |
| [xx.eo.translate_to.sh](https://nlp.johnsnowlabs.com//2021/06/04/translate_sh_eo_xx.html) | [translate_sh_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_sh_eo_xx.html) |
| [xx.fr.translate_to.sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_fr_xx.html) | [translate_sl_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_sl_fr_xx.html) |
| [xx.es.translate_to.tl](https://nlp.johnsnowlabs.com//2021/06/04/translate_tl_es_xx.html) | [translate_tl_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_tl_es_xx.html) |
| [xx.sv.translate_to.tw](https://nlp.johnsnowlabs.com//2021/06/04/translate_tw_sv_xx.html) | [translate_tw_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tw_sv_xx.html) |
| [xx.lt.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_lt_xx.html) | [translate_tr_lt](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_lt_xx.html) |
| [xx.fi.translate_to.tll](https://nlp.johnsnowlabs.com//2021/06/04/translate_tll_fi_xx.html) | [translate_tll_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_tll_fi_xx.html) |
| [xx.sn.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sn_xx.html) | [translate_sv_sn](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sn_xx.html) |
| [xx.tn.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tn_xx.html) | [translate_sv_tn](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tn_xx.html) |
| [xx.sv.translate_to.toi](https://nlp.johnsnowlabs.com//2021/06/04/translate_toi_sv_xx.html) | [translate_toi_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_toi_sv_xx.html) |
| [xx.uk.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_uk_xx.html) | [translate_sv_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_uk_xx.html) |
| [xx.tiv.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tiv_xx.html) | [translate_sv_tiv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tiv_xx.html) |
| [xx.sk.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sk_xx.html) | [translate_sv_sk](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sk_xx.html) |
| [xx.ty.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ty_xx.html) | [translate_sv_ty](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ty_xx.html) |
| [xx.es.translate_to.toi](https://nlp.johnsnowlabs.com//2021/06/04/translate_toi_es_xx.html) | [translate_toi_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_toi_es_xx.html) |
| [xx.rw.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_rw_xx.html) | [translate_sv_rw](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_rw_xx.html) |
| [xx.ny.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ny_xx.html) | [translate_sv_ny](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ny_xx.html) |
| [xx.rnd.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_rnd_xx.html) | [translate_sv_rnd](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_rnd_xx.html) |
| [xx.es.translate_to.tn](https://nlp.johnsnowlabs.com//2021/06/04/translate_tn_es_xx.html) | [translate_tn_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_tn_es_xx.html) |
| [xx.sv.translate_to.tn](https://nlp.johnsnowlabs.com//2021/06/04/translate_tn_sv_xx.html) | [translate_tn_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tn_sv_xx.html) |
| [xx.es.translate_to.tvl](https://nlp.johnsnowlabs.com//2021/06/04/translate_tvl_es_xx.html) | [translate_tvl_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_tvl_es_xx.html) |
| [xx.pon.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_pon_xx.html) | [translate_sv_pon](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_pon_xx.html) |
| [xx.ve.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ve_xx.html) | [translate_sv_ve](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ve_xx.html) |
| [xx.fr.translate_to.tvl](https://nlp.johnsnowlabs.com//2021/06/04/translate_tvl_fr_xx.html) | [translate_tvl_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tvl_fr_xx.html) |
| [xx.es.translate_to.tum](https://nlp.johnsnowlabs.com//2021/06/04/translate_tum_es_xx.html) | [translate_tum_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_tum_es_xx.html) |
| [xx.run.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_run_xx.html) | [translate_sv_run](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_run_xx.html) |
| [xx.de.translate_to.tl](https://nlp.johnsnowlabs.com//2021/06/04/translate_tl_de_xx.html) | [translate_tl_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_tl_de_xx.html) |
| [xx.fi.translate_to.tw](https://nlp.johnsnowlabs.com//2021/06/04/translate_tw_fi_xx.html) | [translate_tw_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_tw_fi_xx.html) |
| [xx.es.translate_to.ty](https://nlp.johnsnowlabs.com//2021/06/04/translate_ty_es_xx.html) | [translate_ty_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ty_es_xx.html) |
| [xx.fr.translate_to.toi](https://nlp.johnsnowlabs.com//2021/06/04/translate_toi_fr_xx.html) | [translate_toi_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_toi_fr_xx.html) |
| [xx.sv.translate_to.tll](https://nlp.johnsnowlabs.com//2021/06/04/translate_tll_sv_xx.html) | [translate_tll_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tll_sv_xx.html) |
| [xx.sg.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sg_xx.html) | [translate_sv_sg](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sg_xx.html) |
| [xx.az.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_az_xx.html) | [translate_tr_az](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_az_xx.html) |
| [xx.es.translate_to.ts](https://nlp.johnsnowlabs.com//2021/06/04/translate_ts_es_xx.html) | [translate_ts_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ts_es_xx.html) |
| [xx.fr.translate_to.ts](https://nlp.johnsnowlabs.com//2021/06/04/translate_ts_fr_xx.html) | [translate_ts_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ts_fr_xx.html) |
| [xx.fr.translate_to.th](https://nlp.johnsnowlabs.com//2021/06/04/translate_th_fr_xx.html) | [translate_th_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_th_fr_xx.html) |
| [xx.zne.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_zne_xx.html) | [translate_sv_zne](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_zne_xx.html) |
| [xx.tw.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tw_xx.html) | [translate_sv_tw](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tw_xx.html) |
| [xx.mh.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_mh_xx.html) | [translate_sv_mh](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_mh_xx.html) |
| [xx.pag.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_pag_xx.html) | [translate_sv_pag](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_pag_xx.html) |
| [xx.fr.translate_to.tum](https://nlp.johnsnowlabs.com//2021/06/04/translate_tum_fr_xx.html) | [translate_tum_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tum_fr_xx.html) |
| [xx.no.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_no_xx.html) | [translate_sv_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_no_xx.html) |
| [xx.ts.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ts_xx.html) | [translate_sv_ts](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ts_xx.html) |
| [xx.mt.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_mt_xx.html) | [translate_sv_mt](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_mt_xx.html) |
| [xx.yo.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_yo_xx.html) | [translate_sv_yo](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_yo_xx.html) |
| [xx.fr.translate_to.to](https://nlp.johnsnowlabs.com//2021/06/04/translate_to_fr_xx.html) | [translate_to_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_to_fr_xx.html) |
| [xx.sv.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sv_xx.html) | [translate_sv_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sv_xx.html) |
| [xx.fi.translate_to.toi](https://nlp.johnsnowlabs.com//2021/06/04/translate_toi_fi_xx.html) | [translate_toi_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_toi_fi_xx.html) |
| [xx.ro.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ro_xx.html) | [translate_sv_ro](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ro_xx.html) |
| [xx.es.translate_to.tw](https://nlp.johnsnowlabs.com//2021/06/04/translate_tw_es_xx.html) | [translate_tw_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_tw_es_xx.html) |
| [xx.niu.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_niu_xx.html) | [translate_sv_niu](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_niu_xx.html) |
| [xx.uk.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_uk_xx.html) | [translate_tr_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_uk_xx.html) |
| [xx.to.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_to_xx.html) | [translate_sv_to](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_to_xx.html) |
| [xx.fi.translate_to.ts](https://nlp.johnsnowlabs.com//2021/06/04/translate_ts_fi_xx.html) | [translate_ts_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ts_fi_xx.html) |
| [xx.tll.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tll_xx.html) | [translate_sv_tll](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tll_xx.html) |
| [xx.fr.translate_to.tll](https://nlp.johnsnowlabs.com//2021/06/04/translate_tll_fr_xx.html) | [translate_tll_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tll_fr_xx.html) |
| [xx.pt.translate_to.tl](https://nlp.johnsnowlabs.com//2021/06/04/translate_tl_pt_xx.html) | [translate_tl_pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_tl_pt_xx.html) |
| [xx.nso.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_nso_xx.html) | [translate_sv_nso](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_nso_xx.html) |
| [xx.sq.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sq_xx.html) | [translate_sv_sq](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sq_xx.html) |
| [xx.sv.translate_to.tpi](https://nlp.johnsnowlabs.com//2021/06/04/translate_tpi_sv_xx.html) | [translate_tpi_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tpi_sv_xx.html) |
| [xx.yap.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_yap_xx.html) | [translate_sv_yap](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_yap_xx.html) |
| [xx.sv.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_sv_xx.html) | [translate_tr_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_sv_xx.html) |
| [xx.fr.translate_to.swc](https://nlp.johnsnowlabs.com//2021/06/04/translate_swc_fr_xx.html) | [translate_swc_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_swc_fr_xx.html) |
| [xx.nl.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_nl_xx.html) | [translate_sv_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_nl_xx.html) |
| [xx.fi.translate_to.ty](https://nlp.johnsnowlabs.com//2021/06/04/translate_ty_fi_xx.html) | [translate_ty_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_ty_fi_xx.html) |
| [xx.fr.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_fr_xx.html) | [translate_tr_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_fr_xx.html) |
| [xx.sv.translate_to.tum](https://nlp.johnsnowlabs.com//2021/06/04/translate_tum_sv_xx.html) | [translate_tum_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tum_sv_xx.html) |
| [xx.swc.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_swc_xx.html) | [translate_sv_swc](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_swc_xx.html) |
| [xx.fi.translate_to.swc](https://nlp.johnsnowlabs.com//2021/06/04/translate_swc_fi_xx.html) | [translate_swc_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_swc_fi_xx.html) |
| [xx.eo.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_eo_xx.html) | [translate_tr_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_eo_xx.html) |
| [xx.xh.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_xh_xx.html) | [translate_sv_xh](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_xh_xx.html) |
| [xx.sv.translate_to.tvl](https://nlp.johnsnowlabs.com//2021/06/04/translate_tvl_sv_xx.html) | [translate_tvl_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tvl_sv_xx.html) |
| [xx.sl.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sl_xx.html) | [translate_sv_sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sl_xx.html) |
| [xx.tum.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tum_xx.html) | [translate_sv_tum](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tum_xx.html) |
| [xx.es.translate_to.to](https://nlp.johnsnowlabs.com//2021/06/04/translate_to_es_xx.html) | [translate_to_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_to_es_xx.html) |
| [xx.fr.translate_to.tn](https://nlp.johnsnowlabs.com//2021/06/04/translate_tn_fr_xx.html) | [translate_tn_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tn_fr_xx.html) |
| [xx.sv.translate_to.ty](https://nlp.johnsnowlabs.com//2021/06/04/translate_ty_sv_xx.html) | [translate_ty_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ty_sv_xx.html) |
| [xx.sv.translate_to.swc](https://nlp.johnsnowlabs.com//2021/06/04/translate_swc_sv_xx.html) | [translate_swc_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_swc_sv_xx.html) |
| [xx.mos.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_mos_xx.html) | [translate_sv_mos](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_mos_xx.html) |
| [xx.ar.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_ar_xx.html) | [translate_tr_ar](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_ar_xx.html) |
| [xx.ru.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ru_xx.html) | [translate_sv_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_ru_xx.html) |
| [xx.srn.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_srn_xx.html) | [translate_sv_srn](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_srn_xx.html) |
| [xx.pis.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_pis_xx.html) | [translate_sv_pis](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_pis_xx.html) |
| [xx.pap.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_pap_xx.html) | [translate_sv_pap](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_pap_xx.html) |
| [xx.tvl.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tvl_xx.html) | [translate_sv_tvl](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tvl_xx.html) |
| [xx.sv.translate_to.to](https://nlp.johnsnowlabs.com//2021/06/04/translate_to_sv_xx.html) | [translate_to_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_to_sv_xx.html) |
| [xx.th.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_th_xx.html) | [translate_sv_th](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_th_xx.html) |
| [xx.war.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_war_xx.html) | [translate_sv_war](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_war_xx.html) |
| [xx.sv.translate_to.ts](https://nlp.johnsnowlabs.com//2021/06/04/translate_ts_sv_xx.html) | [translate_ts_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_ts_sv_xx.html) |
| [xx.fr.translate_to.tw](https://nlp.johnsnowlabs.com//2021/06/04/translate_tw_fr_xx.html) | [translate_tw_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tw_fr_xx.html) |
| [xx.st.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_st_xx.html) | [translate_sv_st](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_st_xx.html) |
| [xx.fr.translate_to.tiv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tiv_fr_xx.html) | [translate_tiv_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tiv_fr_xx.html) |
| [xx.tpi.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tpi_xx.html) | [translate_sv_tpi](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_tpi_xx.html) |
| [xx.fi.translate_to.tvl](https://nlp.johnsnowlabs.com//2021/06/04/translate_tvl_fi_xx.html) | [translate_tvl_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_tvl_fi_xx.html) |
| [xx.fr.translate_to.ty](https://nlp.johnsnowlabs.com//2021/06/04/translate_ty_fr_xx.html) | [translate_ty_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_ty_fr_xx.html) |
| [xx.sm.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sm_xx.html) | [translate_sv_sm](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_sm_xx.html) |
| [xx.es.translate_to.swc](https://nlp.johnsnowlabs.com//2021/06/04/translate_swc_es_xx.html) | [translate_swc_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_swc_es_xx.html) |
| [xx.sv.translate_to.tiv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tiv_sv_xx.html) | [translate_tiv_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_tiv_sv_xx.html) |
| [xx.toi.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_toi_xx.html) | [translate_sv_toi](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_toi_xx.html) |
| [xx.mfe.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_mfe_xx.html) | [translate_sv_mfe](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_mfe_xx.html) |
| [xx.wls.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_wls_xx.html) | [translate_sv_wls](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_wls_xx.html) |
| [xx.umb.translate_to.sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_umb_xx.html) | [translate_sv_umb](https://nlp.johnsnowlabs.com//2021/06/04/translate_sv_umb_xx.html) |
| [xx.es.translate_to.tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_es_xx.html) | [translate_tr_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_tr_es_xx.html) |
| [xx.es.translate_to.tll](https://nlp.johnsnowlabs.com//2021/06/04/translate_tll_es_xx.html) | [translate_tll_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_tll_es_xx.html) |
| [xx.pt.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_pt_xx.html) | [translate_uk_pt](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_pt_xx.html) |
| [xx.it.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_it_xx.html) | [translate_zh_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_it_xx.html) |
| [xx.no.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_no_xx.html) | [translate_uk_no](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_no_xx.html) |
| [xx.sh.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_sh_xx.html) | [translate_uk_sh](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_sh_xx.html) |
| [xx.sv.translate_to.wls](https://nlp.johnsnowlabs.com//2021/06/04/translate_wls_sv_xx.html) | [translate_wls_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_wls_sv_xx.html) |
| [xx.pl.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_pl_xx.html) | [translate_uk_pl](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_pl_xx.html) |
| [xx.es.translate_to.yo](https://nlp.johnsnowlabs.com//2021/06/04/translate_yo_es_xx.html) | [translate_yo_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_yo_es_xx.html) |
| [xx.es.translate_to.war](https://nlp.johnsnowlabs.com//2021/06/04/translate_war_es_xx.html) | [translate_war_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_war_es_xx.html) |
| [xx.sv.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_sv_xx.html) | [translate_zh_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_sv_xx.html) |
| [xx.tr.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_tr_xx.html) | [translate_uk_tr](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_tr_xx.html) |
| [xx.fi.translate_to.war](https://nlp.johnsnowlabs.com//2021/06/04/translate_war_fi_xx.html) | [translate_war_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_war_fi_xx.html) |
| [xx.de.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_de_xx.html) | [translate_zh_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_de_xx.html) |
| [xx.uk.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_uk_xx.html) | [translate_zh_uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_uk_xx.html) |
| [xx.eo.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_eo_xx.html) | [translate_vi_eo](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_eo_xx.html) |
| [xx.bg.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_bg_xx.html) | [translate_zh_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_bg_xx.html) |
| [xx.es.translate_to.zne](https://nlp.johnsnowlabs.com//2021/06/04/translate_zne_es_xx.html) | [translate_zne_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_zne_es_xx.html) |
| [xx.fr.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_fr_xx.html) | [translate_uk_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_fr_xx.html) |
| [xx.zls.translate_to.zls](https://nlp.johnsnowlabs.com//2021/06/04/translate_zls_zls_xx.html) | [translate_zls_zls](https://nlp.johnsnowlabs.com//2021/06/04/translate_zls_zls_xx.html) |
| [xx.fr.translate_to.yo](https://nlp.johnsnowlabs.com//2021/06/04/translate_yo_fr_xx.html) | [translate_yo_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_yo_fr_xx.html) |
| [xx.bg.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_bg_xx.html) | [translate_uk_bg](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_bg_xx.html) |
| [xx.fr.translate_to.xh](https://nlp.johnsnowlabs.com//2021/06/04/translate_xh_fr_xx.html) | [translate_xh_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_xh_fr_xx.html) |
| [xx.ca.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_ca_xx.html) | [translate_uk_ca](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_ca_xx.html) |
| [xx.fi.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_fi_xx.html) | [translate_zh_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_fi_xx.html) |
| [xx.es.translate_to.zai](https://nlp.johnsnowlabs.com//2021/06/04/translate_zai_es_xx.html) | [translate_zai_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_zai_es_xx.html) |
| [xx.es.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_es_xx.html) | [translate_uk_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_es_xx.html) |
| [xx.nl.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_nl_xx.html) | [translate_uk_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_nl_xx.html) |
| [xx.sv.translate_to.yap](https://nlp.johnsnowlabs.com//2021/06/04/translate_yap_sv_xx.html) | [translate_yap_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_yap_sv_xx.html) |
| [xx.he.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_he_xx.html) | [translate_uk_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_he_xx.html) |
| [xx.sl.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_sl_xx.html) | [translate_uk_sl](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_sl_xx.html) |
| [xx.es.translate_to.ve](https://nlp.johnsnowlabs.com//2021/06/04/translate_ve_es_xx.html) | [translate_ve_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_ve_es_xx.html) |
| [xx.zlw.translate_to.zlw](https://nlp.johnsnowlabs.com//2021/06/04/translate_zlw_zlw_xx.html) | [translate_zlw_zlw](https://nlp.johnsnowlabs.com//2021/06/04/translate_zlw_zlw_xx.html) |
| [xx.es.translate_to.tzo](https://nlp.johnsnowlabs.com//2021/06/04/translate_tzo_es_xx.html) | [translate_tzo_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_tzo_es_xx.html) |
| [xx.hu.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_hu_xx.html) | [translate_uk_hu](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_hu_xx.html) |
| [xx.de.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_de_xx.html) | [translate_vi_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_de_xx.html) |
| [xx.fi.translate_to.yo](https://nlp.johnsnowlabs.com//2021/06/04/translate_yo_fi_xx.html) | [translate_yo_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_yo_fi_xx.html) |
| [xx.ru.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_ru_xx.html) | [translate_uk_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_ru_xx.html) |
| [xx.ms.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_ms_xx.html) | [translate_zh_ms](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_ms_xx.html) |
| [xx.urj.translate_to.urj](https://nlp.johnsnowlabs.com//2021/06/04/translate_urj_urj_xx.html) | [translate_urj_urj](https://nlp.johnsnowlabs.com//2021/06/04/translate_urj_urj_xx.html) |
| [xx.it.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_it_xx.html) | [translate_uk_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_it_xx.html) |
| [xx.sv.translate_to.war](https://nlp.johnsnowlabs.com//2021/06/04/translate_war_sv_xx.html) | [translate_war_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_war_sv_xx.html) |
| [xx.fr.translate_to.wls](https://nlp.johnsnowlabs.com//2021/06/04/translate_wls_fr_xx.html) | [translate_wls_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_wls_fr_xx.html) |
| [xx.zle.translate_to.zle](https://nlp.johnsnowlabs.com//2021/06/04/translate_zle_zle_xx.html) | [translate_zle_zle](https://nlp.johnsnowlabs.com//2021/06/04/translate_zle_zle_xx.html) |
| [xx.vi.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_vi_xx.html) | [translate_zh_vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_vi_xx.html) |
| [xx.es.translate_to.vsl](https://nlp.johnsnowlabs.com//2021/06/04/translate_vsl_es_xx.html) | [translate_vsl_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_vsl_es_xx.html) |
| [xx.fi.translate_to.zne](https://nlp.johnsnowlabs.com//2021/06/04/translate_zne_fi_xx.html) | [translate_zne_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_zne_fi_xx.html) |
| [xx.fi.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_fi_xx.html) | [translate_uk_fi](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_fi_xx.html) |
| [xx.ru.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_ru_xx.html) | [translate_vi_ru](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_ru_xx.html) |
| [xx.nl.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_nl_xx.html) | [translate_zh_nl](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_nl_xx.html) |
| [xx.sv.translate_to.xh](https://nlp.johnsnowlabs.com//2021/06/04/translate_xh_sv_xx.html) | [translate_xh_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_xh_sv_xx.html) |
| [xx.es.translate_to.xh](https://nlp.johnsnowlabs.com//2021/06/04/translate_xh_es_xx.html) | [translate_xh_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_xh_es_xx.html) |
| [xx.he.translate_to.zh](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_he_xx.html) | [translate_zh_he](https://nlp.johnsnowlabs.com//2021/06/04/translate_zh_he_xx.html) |
| [xx.fr.translate_to.war](https://nlp.johnsnowlabs.com//2021/06/04/translate_war_fr_xx.html) | [translate_war_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_war_fr_xx.html) |
| [xx.fr.translate_to.zne](https://nlp.johnsnowlabs.com//2021/06/04/translate_zne_fr_xx.html) | [translate_zne_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_zne_fr_xx.html) |
| [xx.sv.translate_to.yo](https://nlp.johnsnowlabs.com//2021/06/04/translate_yo_sv_xx.html) | [translate_yo_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_yo_sv_xx.html) |
| [xx.fr.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_fr_xx.html) | [translate_vi_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_fr_xx.html) |
| [xx.it.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_it_xx.html) | [translate_vi_it](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_it_xx.html) |
| [xx.sv.translate_to.zne](https://nlp.johnsnowlabs.com//2021/06/04/translate_zne_sv_xx.html) | [translate_zne_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_zne_sv_xx.html) |
| [xx.fr.translate_to.yap](https://nlp.johnsnowlabs.com//2021/06/04/translate_yap_fr_xx.html) | [translate_yap_fr](https://nlp.johnsnowlabs.com//2021/06/04/translate_yap_fr_xx.html) |
| [xx.cs.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_cs_xx.html) | [translate_uk_cs](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_cs_xx.html) |
| [xx.es.translate_to.vi](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_es_xx.html) | [translate_vi_es](https://nlp.johnsnowlabs.com//2021/06/04/translate_vi_es_xx.html) |
| [xx.de.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_de_xx.html) | [translate_uk_de](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_de_xx.html) |
| [xx.sv.translate_to.uk](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_sv_xx.html) | [translate_uk_sv](https://nlp.johnsnowlabs.com//2021/06/04/translate_uk_sv_xx.html) |


## Bugfixes
- Fixed bugs that occured when loading a model from disk.




* [140+ NLU Tutorials](https://github.com/JohnSnowLabs/nlu/tree/master/examples)
* [Streamlit visualizations docs](https://nlu.johnsnowlabs.com/docs/en/streamlit_viz_examples)
* The complete list of all 1100+ models & pipelines in 192+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).
* [Spark NLP publications](https://medium.com/spark-nlp)
* [NLU in Action](https://nlp.johnsnowlabs.com/demo)
* [NLU documentation](https://nlu.johnsnowlabs.com/docs/en/install)
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP and NLU!

# 1 line Install NLU on Google Colab
```!wget https://setup.johnsnowlabs.com/nlu/colab.sh  -O - | bash```
# 1 line Install NLU on Kaggle
```!wget https://setup.johnsnowlabs.com/nlu/kaggle.sh  -O - | bash```
# Install via PIP
```! pip install nlu pyspark==3.0.3```





# NLU 3.0.2 Release Notes
<img width="65%" align="right" src="https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/streamlit_docs_assets/gif/start.gif">

This release contains examples and tutorials on how to visualize the 1000+ state-of-the-art NLP models provided by NLU in *just 1 line of code* in `streamlit`.
It includes simple `1-liners` you can sprinkle into your Streamlit app to for features like **Dependency Trees, Named Entities (NER), text classification results, semantic simmilarity,
embedding visualizations via ELMO, BERT, ALBERT, XLNET and much more** .  Additionally, improvements for T5, various resolvers have been added and models `Farsi`, `Hebrew`, `Korean`, and `Turkish`

This is the ultimate NLP research tool. You can visualize and compare the results of hundreds of context aware deep learning embeddings and compare them with classical vanilla embeddings like Glove
and can see with your own eyes how context is encoded by transformer models like `BERT` or `XLNET`and many more !
Besides that, you can also compare the results of the 200+ NER models John Snow Labs provides and see how peformances changes with varrying ebeddings, like Contextual, Static and Domain Specific Embeddings.

## Install
[For detailed instructions refer to the NLU install documentation here](https://nlu.johnsnowlabs.com/docs/en/install)   
You need `Open JDK 8` installed and the following python packages
```bash
pip install nlu streamlit pyspark==3.0.1 sklearn plotly 
```
Problems? [Connect with us on Slack!](https://join.slack.com/t/spark-nlp/shared_invite/zt-lutct9gm-kuUazcyFKhuGY3_0AMkxqA)

## Impatient and want some action?
Just run this Streamlit app, you can use it to generate python code for each NLU-Streamlit building block
```shell
streamlit run https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/01_dashboard.py
```

## Quick Starter cheat sheet - All you need to know in 1 picture for NLU + Streamlit
For NLU models to load, see [the NLU Namespace](https://nlu.johnsnowlabs.com/docs/en/namespace) or the [John Snow Labs Modelshub](https://modelshub.johnsnowlabs.com/models)  or go [straight to the source](https://github.com/JohnSnowLabs/nlu/blob/master/nlu/namespace.py).
![NLU Streamlit Cheatsheet](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/streamlit_docs_assets/img/NLU_Streamlit_Cheetsheet.png)


## Examples
Just try out any of these.
You can use the first example to generate python-code snippets which you can
recycle as building blocks in your streamlit apps!
### Example:  [`01_dashboard`](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/01_dashboard.py)
```shell
streamlit run https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/01_dashboard.py
```
### Example:  [`02_NER`](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/02_NER.py)
```shell
streamlit run https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/02_NER.py
```
### Example:  [`03_text_similarity_matrix`](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/03_text_similarity_matrix.py)
```shell
streamlit run https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/03_text_similarity_matrix.py
```


### Example:  [`04_dependency_tree`](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/04_dependency_tree.py)
```shell
streamlit run https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/04_dependency_tree.py
```

### Example:  [`05_classifiers`](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/05_classifiers.py)
```shell
streamlit run https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/05_classifiers.py
```

### Example:  [`06_token_features`](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/06_token_features.py)
```shell
streamlit run https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/examples/streamlit/06_token_features.py
```

## How to use NLU?
All you need to know about NLU is that there is the [`nlu.load()`](https://nlu.johnsnowlabs.com/docs/en/load_api) method which returns a `NLUPipeline` object
which has a [`.predict()`](https://nlu.johnsnowlabs.com/docs/en/predict_api) that works on most [common data types in the pydata stack like Pandas dataframes](https://nlu.johnsnowlabs.com/docs/en/predict_api#supported-data-types) .     
Ontop of that, there are various visualization methods a NLUPipeline provides easily integrate in Streamlit as re-usable components. [`viz() method`](https://nlu.johnsnowlabs.com/docs/en/viz_examples)





### Overview of NLU + Streamlit buildingblocks

|Method                                                         |               Description                 |
|---------------------------------------------------------------|-------------------------------------------|
| [`nlu.load('<Model>').predict(data)`](TODO.com)                                     | Load any of the [1000+ models](https://nlp.johnsnowlabs.com/models) by providing the model name any predict on most Pythontic [data strucutres like Pandas, strings, arrays of strings and more](https://nlu.johnsnowlabs.com/docs/en/predict_api#supported-data-types) |
| [`nlu.load('<Model>').viz_streamlit(data)`](TODO.com)                               | Display full NLU exploration dashboard, that showcases every feature avaiable with dropdown selectors for 1000+ models|
| [`nlu.load('<Model>').viz_streamlit_similarity([string1, string2])`](TODO.com)      | Display similarity matrix and scalar similarity for every word embedding loaded and 2 strings. |
| [`nlu.load('<Model>').viz_streamlit_ner(data)`](TODO.com)                           | Visualize predicted NER tags from Named Entity Recognizer model|
| [`nlu.load('<Model>').viz_streamlit_dep_tree(data)`](TODO.com)                      | Visualize Dependency Tree together with Part of Speech labels|
| [`nlu.load('<Model>').viz_streamlit_classes(data)`](TODO.com)                       | Display all extracted class features and confidences for every classifier loaded in pipeline|
| [`nlu.load('<Model>').viz_streamlit_token(data)`](TODO.com)                         | Display all detected token features and informations in Streamlit |
| [`nlu.load('<Model>').viz(data, write_to_streamlit=True)`](TODO.com)                | Display the raw visualization without any UI elements. See [viz docs for more info](https://nlu.johnsnowlabs.com/docs/en/viz_examples). By default all aplicable nlu model references will be shown. |
| [`nlu.enable_streamlit_caching()`](#test)  | Enable caching the `nlu.load()` call. Once enabled, the `nlu.load()` method will automatically cached. **This is recommended** to run first and for large peformance gans |


# Detailed visualizer information and API docs

## <kbd>function</kbd> `pipe.viz_streamlit`


Display a highly configurable UI that showcases almost every feature available for Streamlit visualization with model selection dropdowns in your applications.   
Ths includes :
- `Similarity Matrix` & `Scalars` & `Embedding Information` for any of the [100+ Word Embedding Models]()
- `NER visualizations` for any of the [200+ Named entity recognizers]()
- `Labled` & `Unlabled Dependency Trees visualizations` with `Part of Speech Tags` for any of the [100+ Part of Speech Models]()
- `Token informations`  predicted by any of the [1000+ models]()
- `Classification results`  predicted by any of the [100+ models classification models]()
- `Pipeline Configuration` & `Model Information` & `Link to John Snow Labs Modelshub` for all loaded pipelines
- `Auto generate Python code` that can be copy pasted to re-create the individual Streamlit visualization blocks.
  NlLU takes the first model specified as `nlu.load()` for the first visualization run.     
  Once the Streamlit app is running, additional models can easily be added via the UI.    
  It is recommended to run this first, since you can generate Python code snippets `to recreate individual Streamlit visualization blocks`

```python
nlu.load('ner').viz_streamlit(['I love NLU and Streamlit!','I hate buggy software'])
```



![NLU Streamlit UI Overview](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/streamlit_docs_assets/gif/ui.gif)

### <kbd>function parameters</kbd> `pipe.viz_streamlit`

| Argument              | Type                                             |                                                            Default                     |Description |
|-----------------------|--------------------------------------------------|----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `text`                  |  `Union [str, List[str], pd.DataFrame, pd.Series]` | `'NLU and Streamlit go together like peanutbutter and jelly'`                            | Default text for the `Classification`, `Named Entitiy Recognizer`, `Token Information` and `Dependency Tree` visualizations
| `similarity_texts`      |  `Union[List[str],Tuple[str,str]]`                 | `('Donald Trump Likes to part', 'Angela Merkel likes to party')`                         | Default texts for the `Text similarity` visualization. Should contain `exactly 2 strings` which will be compared `token embedding wise`. For each embedding active, a `token wise similarity matrix` and a `similarity scalar`
| `model_selection`       |  `List[str]`                                       | `[]`                                                                                         | List of nlu references to display in the model selector, see [the NLU Namespace](https://nlu.johnsnowlabs.com/docs/en/namespace) or the [John Snow Labs Modelshub](https://modelshub.johnsnowlabs.com/models)  or go [straight to the source](https://github.com/JohnSnowLabs/nlu/blob/master/nlu/namespace.py) for more info
| `title`                 |  `str`                                             | `'NLU ❤️ Streamlit - Prototype your NLP startup in 0 lines of code🚀'`                      | Title of the Streamlit app
| `sub_title`             |  `str`                                             | `'Play with over 1000+ scalable enterprise NLP models'`                                  | Sub title of the Streamlit app
| `visualizers`           |  `List[str]`                                       | `( "dependency_tree", "ner",  "similarity", "token_information", 'classification')`      | Define which visualizations should be displayed. By default all visualizations are displayed.
| `show_models_info`      |  `bool`                                            | `True`                                                                                   | Show information for every model loaded in the bottom of the Streamlit app.
| `show_model_select`   |  `bool`                                          | `True`                                                                                 | Show a model selection dropdowns that makes any of the 1000+ models avaiable in 1 click
| `show_viz_selection`    |  `bool`                                            | `False`                                                                                  | Show a selector in the sidebar which lets you configure which visualizations are displayed.
| `show_logo`             |  `bool`                                            | `True`                                                                                   | Show logo
| `display_infos`         |  `bool`                                            | `False`                                                                                  | Display additonal information about ISO codes and the NLU namespace structure.
| `set_wide_layout_CSS`     |  `bool`                                                             |  `True`                                                                                   | Whether to inject custom CSS or not.
|     `key`                 |  `str`                                                              | `"NLU_streamlit"`    | Key for the Streamlit elements drawn
| `model_select_position`   |  `str`                                                             |   `'side'`            | [Whether to output the positions of predictions or not, see `pipe.predict(positions=true`) for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-positions-parameter)  |
| `show_code_snippets`      |  `bool`                                                             |  `False`                                                                                 | Display Python code snippets above visualizations that can be used to re-create the visualization
|`num_similarity_cols`                               | `int`               |  `2`                            |  How many columns should for the layout in Streamlit when rendering the similarity matrixes.



## <kbd>function</kbd> `pipe.viz_streamlit_classes`

Visualize the predicted classes and their confidences and additional metadata to streamlit.
Aplicable with [any of the 100+ classifiers](https://nlp.johnsnowlabs.com/models?task=Text+Classification)

```python
nlu.load('sentiment').viz_streamlit_classes(['I love NLU and Streamlit!','I love buggy software', 'Sign up now get a chance to win 1000$ !', 'I am afraid of Snakes','Unicorns have been sighted on Mars!','Where is the next bus stop?'])
```
![text_class1](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/streamlit_docs_assets/gif/class.gif)


### <kbd>function parameters</kbd> `pipe.viz_streamlit_classes`

| Argument    | Type        |                                                            Default         |Description |
|--------------------------- | ---------- |-----------------------------------------------------------| ------------------------------------------------------- |
| `text`                    | `Union[str,list,pd.DataFrame, pd.Series, pyspark.sql.DataFrame ]`   |     `'I love NLU and Streamlit and sunny days!'`                  | Text to predict classes for. Will predict on each input of the iteratable or dataframe if type is not str.|
| `output_level`            | `Optional[str]`                                                     |       `document`        | [Outputlevel of NLU pipeline, see `pipe.predict()` docsmore info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-level-parameter)|
| `include_text_col`        |  `bool`                                                              |`True`               | Whether to include a e text column in the output table or just the prediction data |
| `title`                   | `Optional[str]`                                                     |   `Text Classification`            | Title of the Streamlit building block that will be visualized to screen |
| `metadata`                | `bool`                                                              |  `False`             | [whether to output addition metadata or not, see `pipe.predict(meta=true)` docs for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-metadata) |
| `positions`               |  `bool`                                                             |   `False`            | [whether to output the positions of predictions or not, see `pipe.predict(positions=true`) for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-positions-parameter)  |
| `set_wide_layout_CSS`     |  `bool`                                                             |  `True`                                                                                   | Whether to inject custom CSS or not.
|     `key`                 |  `str`                                                              | `"NLU_streamlit"`    | Key for the Streamlit elements drawn
| `model_select_position`   |  `str`                                                             |   `'side'`            | [Whether to output the positions of predictions or not, see `pipe.predict(positions=true`) for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-positions-parameter)  |
| `generate_code_sample`      |  `bool`                                                             |  `False`                                                                                 | Display Python code snippets above visualizations that can be used to re-create the visualization
| `show_model_select`   |  `bool`                                          | `True`                                                                                 | Show a model selection dropdowns that makes any of the 1000+ models avaiable in 1 click
| `show_logo`             |  `bool`                                            | `True`                                                                                   | Show logo
| `display_infos`         |  `bool`                                            | `False`                                                                                  | Display additonal information about ISO codes and the NLU namespace structure.



## <kbd>function</kbd> `pipe.viz_streamlit_ner`
Visualize the predicted classes and their confidences and additional metadata to Streamlit.
Aplicable with [any of the 250+ NER models](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition).    
You can filter which NER tags to highlight via the dropdown in the main window.

Basic usage
```python
nlu.load('ner').viz_streamlit_ner('Donald Trump from America and Angela Merkel from Germany dont share many views')
```

![NER visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/streamlit_docs_assets/gif/NER.gif)

Example for coloring
```python
# Color all entities of class GPE black
nlu.load('ner').viz_streamlit_ner('Donald Trump from America and Angela Merkel from Germany dont share many views',colors={'PERSON':'#6e992e', 'GPE':'#000000'})
```
![NER coloring](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/streamlit_docs_assets/img/NER_colored.png)

### <kbd>function parameters</kbd> `pipe.viz_streamlit_ner`

| Argument    | Type        |                                                                                      Default                                        |Description                  |
|--------------------------- | -----------------------|-----------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------ |
| `text`                     | `str`                  |     `'Donald Trump from America and Anegela Merkel from Germany do not share many views'`                 | Text to predict classes for.|
| `ner_tags`                 | `Optional[List[str]]`  |       `None`                                                                                               |Tags to display. By default all tags will be displayed|
| `show_label_select`        |  `bool`                |`True`                                                                                                      | Whether to include the label selector|
| `show_table`               | `bool`                 |   `True`                                                                                                   | Whether show to predicted pandas table or not|
| `title`                    | `Optional[str]`        |  `'Named Entities'`                                                                                        |  Title of the Streamlit building block that will be visualized to screen |
| `sub_title`                    | `Optional[str]`        |  `'"Recognize various Named Entities (NER) in text entered and filter them. You can select from over 100 languages in the dropdown. On the left side.",'`                                                                                        |  Sub-title of the Streamlit building block that will be visualized to screen |
| `colors`                   |  `Dict[str,str]`       |   `{}`                                                                                                     | Dict with `KEY=ENTITY_LABEL` and `VALUE=COLOR_AS_HEX_CODE`,which will change color of highlighted entities.[See custom color labels docs for more info.](https://nlu.johnsnowlabs.com/docs/en/viz_examples#define-custom-colors-for-labels) |
| `set_wide_layout_CSS`      |  `bool`                                                             |  `True`                                                                                   | Whether to inject custom CSS or not.
|     `key`                  |  `str`                                                              | `"NLU_streamlit"`    | Key for the Streamlit elements drawn
| `generate_code_sample`       |  `bool`                                                             |  `False`                                                                                 | Display Python code snippets above visualizations that can be used to re-create the visualization
| `show_model_select`        |  `bool`                                          | `True`                                                                                 | Show a model selection dropdowns that makes any of the 1000+ models avaiable in 1 click
| `model_select_position`    |  `str`                                                             |   `'side'`            | [Whether to output the positions of predictions or not, see `pipe.predict(positions=true`) for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-positions-parameter)  |
| `show_text_input`        |  `bool`                                                              | `True`                                                                                 | Show text input field to input text in
| `show_logo`             |  `bool`                                            | `True`                                                                                   | Show logo
| `display_infos`         |  `bool`                                            | `False`                                                                                  | Display additonal information about ISO codes and the NLU namespace structure.




## <kbd>function</kbd> `pipe.viz_streamlit_dep_tree`
Visualize a typed dependency tree, the relations between tokens and part of speech tags predicted.
Aplicable with [any of the 100+ Part of Speech(POS) models and dep tree model](https://nlp.johnsnowlabs.com/models?task=Part+of+Speech+Tagging)

```python
nlu.load('dep.typed').viz_streamlit_dep_tree('POS tags define a grammatical label for each token and the Dependency Tree classifies Relations between the tokens')
```
![Dependency Tree](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/streamlit_docs_assets/img/DEP.png)

### <kbd>function parameters</kbd> `pipe.viz_streamlit_dep_tree`

| Argument    | Type        |                                                            Default         |Description |
|--------------------------- | ---------- |-----------------------------------------------------------| ------------------------------------------------------- |
| `text`                    | `str`   |     `'Billy likes to swim'`                 | Text to predict classes for.|
| `title`                | `Optional[str]`                                                              |  `'Dependency Parse Tree & Part-of-speech tags'`             |  Title of the Streamlit building block that will be visualized to screen |
| `set_wide_layout_CSS`      |  `bool`                                                             |  `True`                                                                                   | Whether to inject custom CSS or not.
|     `key`                  |  `str`                                                              | `"NLU_streamlit"`    | Key for the Streamlit elements drawn
| `generate_code_sample`       |  `bool`                                                             |  `False`                                                                                 | Display Python code snippets above visualizations that can be used to re-create the visualization
| `set_wide_layout_CSS`      |  `bool`                                                             |  `True`                                                                                   | Whether to inject custom CSS or not.
|     `key`                  |  `str`                                                              | `"NLU_streamlit"`    | Key for the Streamlit elements drawn
| `generate_code_sample`       |  `bool`                                                             |  `False`                                                                                 | Display Python code snippets above visualizations that can be used to re-create the visualization
| `show_model_select`        |  `bool`                                          | `True`                                                                                 | Show a model selection dropdowns that makes any of the 1000+ models avaiable in 1 click
| `model_select_position`    |  `str`                                                             |   `'side'`            | [Whether to output the positions of predictions or not, see `pipe.predict(positions=true`) for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-positions-parameter)  |
| `show_logo`             |  `bool`                                            | `True`                                                                                   | Show logo
| `display_infos`         |  `bool`                                            | `False`                                                                                  | Display additonal information about ISO codes and the NLU namespace structure.





## <kbd>function</kbd> `pipe.viz_streamlit_token`
Visualize predicted token and text features for every model loaded.
You can use this with [any of the 1000+ models](https://nlp.johnsnowlabs.com/models) and select them from the left dropdown.

```python
nlu.load('stemm pos spell').viz_streamlit_token('I liek pentut buttr and jelly !')
```
![text_class1](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/streamlit_docs_assets/gif/token.gif)


### <kbd>function parameters</kbd> `pipe.viz_streamlit_token`

| Argument    | Type        |                                                            Default         |Description |
|--------------------------- | ---------- |-----------------------------------------------------------| ------------------------------------------------------- |
| `text`                    | `str`   |     `'NLU and Streamlit are great!'`                 | Text to predict token information for.|
| `title`                | `Optional[str]`                                                              |  `'Named Entities'`             |  Title of the Streamlit building block that will be visualized to screen |
| `show_feature_select`        |  `bool`                                                              |`True`               | Whether to include the token feature selector|
| `features`            | `Optional[List[str]]`                                                     |       `None`        |Features to to display. By default all Features will be displayed|
| `metadata`                | `bool`                                                              |  `False`             | [Whether to output addition metadata or not, see `pipe.predict(meta=true)` docs for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-metadata) |
| `output_level`            | `Optional[str]`                                                     |       `'token'`        | [Outputlevel of NLU pipeline, see `pipe.predict()` docsmore info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-level-parameter)|
| `positions`               |  `bool`                                                             |   `False`            | [Whether to output the positions of predictions or not, see `pipe.predict(positions=true`) for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-positions-parameter)  |
| `set_wide_layout_CSS`      |  `bool`                                                             |  `True`                                                                                   | Whether to inject custom CSS or not.
|     `key`                  |  `str`                                                              | `"NLU_streamlit"`    | Key for the Streamlit elements drawn
| `generate_code_sample`       |  `bool`                                                             |  `False`                                                                                 | Display Python code snippets above visualizations that can be used to re-create the visualization
| `show_model_select`        |  `bool`                                          | `True`                                                                                 | Show a model selection dropdowns that makes any of the 1000+ models avaiable in 1 click
| `model_select_position`    |  `str`                                                             |   `'side'`            | [Whether to output the positions of predictions or not, see `pipe.predict(positions=true`) for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-positions-parameter)  |
| `show_logo`             |  `bool`                                            | `True`                                                                                   | Show logo
| `display_infos`         |  `bool`                                            | `False`                                                                                  | Display additonal information about ISO codes and the NLU namespace structure.




## <kbd>function</kbd> `pipe.viz_streamlit_similarity`

- Displays a `similarity matrix`, where `x-axis` is every token in the first text and `y-axis` is every token in the second text.
- Index `i,j` in the matrix describes the similarity of `token-i` to `token-j` based on the loaded embeddings and distance metrics, based on [Sklearns Pariwise Metrics.](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise). [See this article for more elaboration on similarities](https://medium.com/spark-nlp/easy-sentence-similarity-with-bert-sentence-embeddings-using-john-snow-labs-nlu-ea078deb6ebf)
- Displays  a dropdown selectors from which various similarity metrics and over 100 embeddings can be selected.
  -There will be one similarity matrix per `metric` and `embedding` pair selected. `num_plots = num_metric*num_embeddings`
  Also displays embedding vector information.
  Applicable with [any of the 100+ Word Embedding models](https://nlp.johnsnowlabs.com/models?task=Embeddings)



```python
nlu.load('bert').viz_streamlit_word_similarity(['I love love loooove NLU! <3','I also love love looove  Streamlit! <3'])
```
![text_class1](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/streamlit_docs_assets/gif/SIM.gif)

### <kbd>function parameters</kbd> `pipe.viz_streamlit_similarity`

| Argument    | Type        |                                                            Default         |Description |
|--------------------------- | ---------- |-----------------------------------------------------------| ------------------------------------------------------- |
| `texts`                                 | `str`               |     `'Donald Trump from America and Anegela Merkel from Germany do not share many views.'`                 | Text to predict token information for.|
| `title`                                 | `Optional[str]`     |  `'Named Entities'`             |  Title of the Streamlit building block that will be visualized to screen |
| `similarity_matrix`                     | `bool`              |       `None`                    |Whether to display similarity matrix or not|
| `show_algo_select`                      |  `bool`             |`True`                           | Whether to show dist algo select or not |
| `show_table`                            | `bool`              |   `True`                        | Whether show to predicted pandas table or not|
| `threshold`                             | `float`             |  `0.5`                          | Threshold for displaying result red on screen |
| `set_wide_layout_CSS`                   |  `bool`             |  `True`                         | Whether to inject custom CSS or not.
|     `key`                               |  `str`              | `"NLU_streamlit"`               | Key for the Streamlit elements drawn
| `generate_code_sample`                  |  `bool`             |  `False`                        | Display Python code snippets above visualizations that can be used to re-create the visualization
| `show_model_select`                     |  `bool`             | `True`                          | Show a model selection dropdowns that makes any of the 1000+ models avaiable in 1 click
| `model_select_position`                 |  `str`              |   `'side'`                      | [Whether to output the positions of predictions or not, see `pipe.predict(positions=true`) for more info](https://nlu.johnsnowlabs.com/docs/en/predict_api#output-positions-parameter)  |
|`write_raw_pandas`                       | `bool`              |  `False`                        | Write the raw pandas similarity df to streamlit
|`display_embed_information`              | `bool`              |  `True`                         | Show additional embedding information like `dimension`, `nlu_reference`, `spark_nlp_reference`, `sotrage_reference`, `modelhub link` and more.
|`dist_metrics`                           | `List[str]`         |  `['cosine']`                   | Which distance metrics to apply. If multiple are selected, there will be multiple plots for each embedding and metric. `num_plots = num_metric*num_embeddings`. Can use multiple at the same time, any of of `cityblock`,`cosine`,`euclidean`,`l2`,`l1`,`manhattan`,`nan_euclidean`. Provided via [Sklearn `metrics.pairwise` package](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise)
|`num_cols`                               | `int`               |  `2`                            |  How many columns should for the layout in streamlit when rendering the similarity matrixes.
|`display_scalar_similarities`            | `bool`              |  `False`                        | Display scalar simmilarities in an additional field.
|`display_similarity_summary`             | `bool`              |  `False`                        | Display summary of all similarities for all embeddings and metrics.
| `show_logo`             |  `bool`                             | `True`                                                                                   | Show logo
| `display_infos`         |  `bool`                             | `False`                                                                                  | Display additonal information about ISO codes and the NLU namespace structure.











## 
In addition have added some new features to our T5 Transformer annotator to help with longer and more accurate text generation, trained some new multi-lingual models and pipelines in `Farsi`, `Hebrew`, `Korean`, and `Turkish`.



## T5 Model Improvements
* Add 6 new features to T5Transformer for longer and better text generation
  - doSample: Whether or not to use sampling; use greedy decoding otherwise
  - temperature: The value used to module the next token probabilities
  - topK: The number of highest probability vocabulary tokens to keep for top-k-filtering
  - topP: If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation
  - repetitionPenalty: The parameter for repetition penalty. 1.0 means no penalty. See [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) paper for more details
  - noRepeatNgramSize: If set to int > 0, all ngrams of that size can only occur once


## New Open Source Model in NLU 3.0.2
New multilingual models and pipelines for `Farsi`, `Hebrew`, `Korean`, and `Turkish`

| Model       |NLU Reference         | Spark NLP Reference               | Lang |  
|:----------|:-------------------------------------|:-----------------|:------|
| ClassifierDLModel       | [`tr.classify.news`](https://nlp.johnsnowlabs.com/2021/05/03/classifierdl_bert_news_tr.html) |        [classifierdl_bert_news](https://nlp.johnsnowlabs.com/2021/05/03/classifierdl_bert_news_tr.html) |  `tr`
| UniversalSentenceEncoder| [`xx.use.multi`](https://nlp.johnsnowlabs.com/2021/05/06/tfhub_use_multi_xx.html) | [tfhub_use_multi](https://nlp.johnsnowlabs.com/2021/05/06/tfhub_use_multi_xx.html) |  `xx`
| UniversalSentenceEncoder| [`xx.use.multi_lg`](https://nlp.johnsnowlabs.com/2021/05/06/tfhub_use_multi_lg_xx.html) | [tfhub_use_multi_lg](https://nlp.johnsnowlabs.com/2021/05/06/tfhub_use_multi_lg_xx.html) |  `xx`

| Pipeline                  |NLU Reference| Spark NLP Reference               | Lang |  
|:-----------------------------|:-------------------|:-----------------|:------|
| PretrainedPipeline | [`fa.ner.dl`](https://nlp.johnsnowlabs.com/2021/04/26/recognize_entities_dl_fa.html) | [recognize_entities_dl](https://nlp.johnsnowlabs.com/2021/04/26/recognize_entities_dl_fa.html) |`fa`
| PretrainedPipeline | [`he.explain_document`](https://nlp.johnsnowlabs.com/2021/04/30/explain_document_lg_he.html) | [explain_document_lg](https://nlp.johnsnowlabs.com/2021/04/30/explain_document_lg_he.html) |`he`
| PretrainedPipeline | [`ko.explain_document`](https://nlp.johnsnowlabs.com/2021/04/30/explain_document_lg_ko.html) | [explain_document_lg](https://nlp.johnsnowlabs.com/2021/04/30/explain_document_lg_ko.html) |`ko`



## New Healthcare Models in NLU 3.0.2
Five new resolver models:
- `en.resolve.umls`: This model returns CUI (concept unique identifier) codes for Clinical Findings, Medical Devices, Anatomical Structures and Injuries & Poisoning terms.
- `en.resolve.umls.findings`: This model returns CUI (concept unique identifier) codes for 200K concepts from clinical findings.
- `en.resolve.loinc`: Map clinical NER entities to LOINC codes using sbiobert.
- `en.resolve.loinc.bluebert`: Map clinical NER entities to LOINC codes using sbluebert.
- `en.resolve.HPO`: This model returns Human Phenotype Ontology (HPO) codes for phenotypic abnormalities encountered in human diseases. It also returns associated codes from the following vocabularies for each HPO code:

[Related NLU Notebook](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/release_notebooks/NLU_3_0_2_release_notebook.ipynb)

|Model| NLU Reference                           | Spark NLP Reference                                 |
|--------|-----------------------------------|-----------------------------------------|
|Resolver|[`en.resolve.umls`                    ](https://nlp.johnsnowlabs.com/2021/05/16/sbiobertresolve_umls_major_concepts_en.html)| [`sbiobertresolve_umls_major_concepts`](https://nlp.johnsnowlabs.com/2021/05/16/sbiobertresolve_umls_major_concepts_en.html)     |
|Resolver|[`en.resolve.umls.findings`           ](https://nlp.johnsnowlabs.com/2021/05/16/sbiobertresolve_umls_findings_en.html)| [`sbiobertresolve_umls_findings`](https://nlp.johnsnowlabs.com/2021/05/16/sbiobertresolve_umls_findings_en.html)           |
|Resolver|[`en.resolve.loinc`                   ](https://nlp.johnsnowlabs.com/2021/04/29/sbiobertresolve_loinc_en.html)| [`sbiobertresolve_loinc`](https://nlp.johnsnowlabs.com/2021/04/29/sbiobertresolve_loinc_en.html)                   |
|Resolver|[`en.resolve.loinc.biobert`           ](https://nlp.johnsnowlabs.com/2021/04/29/sbiobertresolve_loinc_en.html)| [`sbiobertresolve_loinc`](https://nlp.johnsnowlabs.com/2021/04/29/sbiobertresolve_loinc_en.html)                   |
|Resolver|[`en.resolve.loinc.bluebert`          ](https://nlp.johnsnowlabs.com/2021/04/29/sbluebertresolve_loinc_en.html)| [`sbluebertresolve_loinc`](https://nlp.johnsnowlabs.com/2021/04/29/sbluebertresolve_loinc_en.html)                  |
|Resolver|[`en.resolve.HPO`                     ](https://nlp.johnsnowlabs.com/2021/05/16/sbiobertresolve_HPO_en.html)| [`sbiobertresolve_HPO`](https://nlp.johnsnowlabs.com/2021/05/16/sbiobertresolve_HPO_en.html)                     |



[en.resolve.HPO](https://nlp.johnsnowlabs.com/2021/05/16/sbiobertresolve_HPO_en.html)

```python
nlu.load('med_ner.jsl.wip.clinical en.resolve.HPO').viz("""These disorders include cancer, bipolar disorder, schizophrenia, autism, Cri-du-chat syndrome,
 myopia, cortical cataract-linked Alzheimer's disease, and infectious diseases""")
```
![text_class1](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/releases/3_0_2/HPO.png)



[en.resolve.loinc.bluebert](https://nlp.johnsnowlabs.com/2021/04/29/sbluebertresolve_loinc_en.html)
```python
nlu.load('med_ner.jsl.wip.clinical en.resolve.loinc.bluebert').viz("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and
subsequent type two diabetes mellitus (TSS2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute 
hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.""")
```
![text_class1](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/releases/3_0_2/LIONC_blue.png)



[en.resolve.umls.findings](https://nlp.johnsnowlabs.com/2021/05/16/sbiobertresolve_umls_findings_en.html)
```python
nlu.load('med_ner.jsl.wip.clinical en.resolve.umls.findings').viz("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and
subsequent type two diabetes mellitus (TSS2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute 
hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting."""
)
```
![text_class1](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/releases/3_0_2/umls_finding.png)


[en.resolve.umls](https://nlp.johnsnowlabs.com/2021/05/16/sbiobertresolve_umls_major_concepts_en.html)
```python
nlu.load('med_ner.jsl.wip.clinical en.resolve.umls').viz("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and
subsequent type two diabetes mellitus (TSS2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute 
hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.""")
```
![text_class1](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/releases/3_0_2/umls.png)




[en.resolve.loinc](https://nlp.johnsnowlabs.com/2021/04/29/sbiobertresolve_loinc_en.html)
```python
nlu.load('med_ner.jsl.wip.clinical en.resolve.loinc').predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and
subsequent type two diabetes mellitus (TSS2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute 
hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.""")
```
![text_class1](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/releases/3_0_2/LIONC.png)



[en.resolve.loinc.biobert](https://nlp.johnsnowlabs.com/2021/04/29/sbiobertresolve_loinc_en.html)
```python
nlu.load('med_ner.jsl.wip.clinical en.resolve.loinc.biobert').predict("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and
subsequent type two diabetes mellitus (TSS2DM), one prior episode of HTG-induced pancreatitis three years prior to presentation, associated with an acute 
hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2, presented with a one-week history of polyuria, polydipsia, poor appetite, and vomiting.""")
```
![text_class1](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/releases/3_0_2/LIONC_BIOBERT.png)




* [140+ tutorials](https://github.com/JohnSnowLabs/nlu/tree/master/examples)
* [New Streamlit visualizations docs](https://nlu.johnsnowlabs.com/docs/en/streamlit_viz_examples)
* The complete list of all 1100+ models & pipelines in 192+ languages is available on [Models Hub](https://nlp.johnsnowlabs.com/models).
* [Spark NLP publications](https://medium.com/spark-nlp)
* [NLU in Action](https://nlp.johnsnowlabs.com/demo)
* [NLU documentation](https://nlu.johnsnowlabs.com/docs/en/install)
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP and NLU!

# 1 line Install NLU on Google Colab
```!wget https://setup.johnsnowlabs.com/nlu/colab.sh  -O - | bash```
# 1 line Install NLU on Kaggle
```!wget https://setup.johnsnowlabs.com/nlu/kaggle.sh  -O - | bash```
# Install via PIP
```! pip install nlu pyspark==3.0.1```






# NLU 3.0.1 Release Notes
We are very excited to announce NLU 3.0.1 has been released!
This is one of the most visually appealing releases, with the integration of the [Spark-NLP-Display](https://nlp.johnsnowlabs.com/docs/en/display) library and visualizations for `dependency trees`, `entity resolution`, `entity assertion`, `relationship between entities` and `named 
entity recognition`. In addition to this, the schema of how columns are named by NLU has been reworked and all 140+ tutorial notebooks have been updated to reflect the latest changes in NLU 3.0.0+
Finally, new multilingual models for `Afrikaans`, `Welsh`, `Maltese`, `Tamil`, and`Vietnamese` are now available.




# New Features and Enhancements
- 1 line to visualization for `NER`, `Dependency`, `Resolution`, `Assertion` and `Relation` via [Spark-NLP-Display](https://nlp.johnsnowlabs.com/docs/en/display) integration
- Improved column naming schema
- [Over 140 + NLU tutorial Notebooks updated](https://github.com/JohnSnowLabs/nlu/tree/master/examples) and improved to reflect latest changes in NLU 3.0.0 +
- New multilingual models for `Afrikaans`, `Welsh`, `Maltese`, `Tamil`, and`Vietnamese`
 

## Improved Column Name generation
- NLU categorized each internal component now with boolean labels for `name_deductable` and `always_name_deductable` .
- Before generating column names, NLU checks wether each component is of unique in the pipeline or not. If a component is not unique in the
  pipe and there are multiple components of same type, i.e. multiple `NER` models, NLU will deduct a base name for the final output columns from the
  NLU reference each NER model is pointing to.
- If on the other hand, there is only one `NER` model in the pipeline, only the default `ner` column prefixed will be generated.
- For some components, like `embeddings` and `classifiers` are now defined as `always_name_deductable`, for those NLU will always try to infer a meaningful base name for the output columns.
- Newly trained component output columns will now be prefixed with `trained_<type>` , for types `pos` , `ner`, `cLassifier`, `sentiment` and `multi_classifier`

## Enhanced offline mode
- You can still load a model from a path as usual with `nlu.load(path=model_path)` and output columns will be suffixed with `from_disk`
- You can now optionally also specify `request` parameter during  load a model from HDD, it will be used to deduct more meaningful column name suffixes, instead of `from_disk`, i.e. by calling `nlu.load(request ='en.embed_sentence.biobert.pubmed_pmc_base_cased', path=model_path)`

## NLU visualization
The latest NLU release integrated the beautiful Spark-NLP-Display package visualizations. You do not need to worry about installing it, when you try to visualize something, NLU will check if
Spark-NLP-Display is installed, if it is missing it will be dynamically installed into your python executable environment, so you don't need to worry about anything!

See the [visualization tutorial notebook](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/visualization/NLU_visualizations_tutorial.ipynb)  and [visualization docs](https://nlu.johnsnowlabs.com/docs/en/viz_examples) for more info.

![Cheat Sheet visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/cheat_sheet.png)

## NER visualization
Applicable to any of the [100+ NER models! See here for an overview](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition)
```python
nlu.load('ner').viz("Donald Trump from America and Angela Merkel from Germany don't share many oppinions.")
```
![NER visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/NER.png)

## Dependency tree visualization
Visualizes the structure of the labeled dependency tree and part of speech tags
```python
nlu.load('dep.typed').viz("Billy went to the mall")
```

![Dependency Tree visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/DEP.png)

```python
#Bigger Example
nlu.load('dep.typed').viz("Donald Trump from America and Angela Merkel from Germany don't share many oppinions but they both love John Snow Labs software")
```
![Dependency Tree visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/DEP_big.png)

## Assertion status visualization
Visualizes asserted statuses and entities.        
Applicable to any of the [10 + Assertion models! See here for an overview](https://nlp.johnsnowlabs.com/models?task=Assertion+Status)
```python
nlu.load('med_ner.clinical assert').viz("The MRI scan showed no signs of cancer in the left lung")
```


![Assert visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/assertion.png)

```python
#bigger example
data ='This is the case of a very pleasant 46-year-old Caucasian female, seen in clinic on 12/11/07 during which time MRI of the left shoulder showed no evidence of rotator cuff tear. She did have a previous MRI of the cervical spine that did show an osteophyte on the left C6-C7 level. Based on this, negative MRI of the shoulder, the patient was recommended to have anterior cervical discectomy with anterior interbody fusion at C6-C7 level. Operation, expected outcome, risks, and benefits were discussed with her. Risks include, but not exclusive of bleeding and infection, bleeding could be soft tissue bleeding, which may compromise airway and may result in return to the operating room emergently for evacuation of said hematoma. There is also the possibility of bleeding into the epidural space, which can compress the spinal cord and result in weakness and numbness of all four extremities as well as impairment of bowel and bladder function. However, the patient may develop deeper-seated infection, which may require return to the operating room. Should the infection be in the area of the spinal instrumentation, this will cause a dilemma since there might be a need to remove the spinal instrumentation and/or allograft. There is also the possibility of potential injury to the esophageus, the trachea, and the carotid artery. There is also the risks of stroke on the right cerebral circulation should an undiagnosed plaque be propelled from the right carotid. She understood all of these risks and agreed to have the procedure performed.'
nlu.load('med_ner.clinical assert').viz(data)
```
![Assert visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/assertion_big.png)


## Relationship between entities visualization
Visualizes the extracted entities between relationship.    
Applicable to any of the [20 + Relation Extractor models See here for an overview](https://nlp.johnsnowlabs.com/models?task=Relation+Extraction)
```python
nlu.load('med_ner.jsl.wip.clinical relation.temporal_events').viz('The patient developed cancer after a mercury poisoning in 1999 ')
```
![Entity Relation visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/relation.png)

```python
# bigger example
data = 'This is the case of a very pleasant 46-year-old Caucasian female, seen in clinic on 12/11/07 during which time MRI of the left shoulder showed no evidence of rotator cuff tear. She did have a previous MRI of the cervical spine that did show an osteophyte on the left C6-C7 level. Based on this, negative MRI of the shoulder, the patient was recommended to have anterior cervical discectomy with anterior interbody fusion at C6-C7 level. Operation, expected outcome, risks, and benefits were discussed with her. Risks include, but not exclusive of bleeding and infection, bleeding could be soft tissue bleeding, which may compromise airway and may result in return to the operating room emergently for evacuation of said hematoma. There is also the possibility of bleeding into the epidural space, which can compress the spinal cord and result in weakness and numbness of all four extremities as well as impairment of bowel and bladder function. However, the patient may develop deeper-seated infection, which may require return to the operating room. Should the infection be in the area of the spinal instrumentation, this will cause a dilemma since there might be a need to remove the spinal instrumentation and/or allograft. There is also the possibility of potential injury to the esophageus, the trachea, and the carotid artery. There is also the risks of stroke on the right cerebral circulation should an undiagnosed plaque be propelled from the right carotid. She understood all of these risks and agreed to have the procedure performed'
pipe = nlu.load('med_ner.jsl.wip.clinical relation.clinical').viz(data)
```
![Entity Relation visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/relation_big.png)


## Entity Resolution visualization for chunks
Visualizes resolutions of entities
Applicable to any of the [100+ Resolver models See here for an overview](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution)
```python
nlu.load('med_ner.jsl.wip.clinical resolve_chunk.rxnorm.in').viz("He took Prevacid 30 mg  daily")
```
![Chunk Resolution visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/resolve_chunk.png)

```python
# bigger example
data = "This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."
nlu.load('med_ner.jsl.wip.clinical resolve_chunk.rxnorm.in').viz(data)
```

![Chunk Resolution visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/resolve_chunk_big.png)


## Entity Resolution visualization for sentences
Visualizes resolutions of entities in sentences
Applicable to any of the [100+ Resolver models See here for an overview](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution)
```python
nlu.load('med_ner.jsl.wip.clinical resolve.icd10cm').viz('She was diagnosed with a respiratory congestion')
```
![Sentence Resolution visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/resolve_sentence.png)

```python
# bigger example
data = 'The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion'
nlu.load('med_ner.jsl.wip.clinical resolve.icd10cm').viz(data)
```
![Sentence Resolution visualization](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/resolve_sentence_big.png)

## Configure visualizations
### Define custom colors for labels
Some entity and relation labels will be highlighted with a pre-defined color, which you [can find here](https://github.com/JohnSnowLabs/spark-nlp-display/tree/main/sparknlp_display/label_colors).    
For labels that have no color defined, a random color will be generated.     
You can define colors for labels manually, by specifying via the `viz_colors` parameter
and defining `hex color codes` in a dictionary that maps `labels` to `colors` .
```python
data = 'Dr. John Snow suggested that Fritz takes 5mg penicilin for his cough'
# Define custom colors for labels
viz_colors={'STRENGTH':'#800080', 'DRUG_BRANDNAME':'#77b5fe', 'GENDER':'#77ffe'}
nlu.load('med_ner.jsl.wip.clinical').viz(data,viz_colors =viz_colors)
```
![define colors labels](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/define_colors.png)


### Filter entities that get highlighted
By default every entity class will be visualized.    
The `labels_to_viz` can be used to define a set of labels to highlight.       
Applicable for ner, resolution and assert.
```python
data = 'Dr. John Snow suggested that Fritz takes 5mg penicilin for his cough'
# Filter wich NER label to viz
labels_to_viz=['SYMPTOM']
nlu.load('med_ner.jsl.wip.clinical').viz(data,labels_to_viz=labels_to_viz)
```
![filter labels](https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/docs/assets/images/nlu/VizExamples/viz_module/filter_labels.png)


## New models
New multilingual models for `Afrikaans`, `Welsh`, `Maltese`, `Tamil`, and`Vietnamese`

| nlu.load() Refrence                                          | Spark NLP Refrence                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [vi.lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_vi.html) | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_vi.html) |
| [mt.lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_mt.html) | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_mt.html) |
| [ta.lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_ta.html) | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_ta.html) |
| [af.lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_af.html) | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_af.html) |
| [af.pos](https://nlp.johnsnowlabs.com/2021/04/06/pos_afribooms_af.html) | [pos_afribooms](https://nlp.johnsnowlabs.com/2021/04/06/pos_afribooms_af.html) |
| [cy.lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_cy.html) | [lemma](https://nlp.johnsnowlabs.com/2021/04/02/lemma_cy.html) |

## Reworked and updated NLU tutorial notebooks

All of the [140+ NLU tutorial Notebooks](https://github.com/JohnSnowLabs/nlu/tree/master/examples) have been updated and reworked to reflect the latest changes in NLU 3.0.0+



### Bugfixes
- Fixed a bug that caused  resolution algorithms output level to be inferred incorrectly
- Fixed a bug that caused stranger cols got dropped
- Fixed a bug that caused endings to miss when  .predict(position=True) was specified
- Fixed a bug that caused pd.Series to be converted incorrectly internally
- Fixed a bug that caused output level transformations to crash
- Fixed a bug that caused verbose mode not to turn of properly after turning it on.
- fixed a bug that caused some models to crash when loaded for HDD

* [140+ updates tutorials](https://github.com/JohnSnowLabs/nlu/tree/master/examples)
* [Updated visualization docs](https://nlu.johnsnowlabs.com/docs/en/viz_examples)
* [Models Hub](https://nlp.johnsnowlabs.com/models) with new models
* [Spark NLP publications](https://medium.com/spark-nlp)
* [NLU in Action](https://nlp.johnsnowlabs.com/demo)
* [NLU documentation](https://nlu.johnsnowlabs.com/docs/en/install)
* [Discussions](https://github.com/JohnSnowLabs/spark-nlp/discussions) Engage with other community members, share ideas, and show off how you use Spark NLP and NLU!

# 1 line Install NLU on Google Colab
```!wget https://setup.johnsnowlabs.com/nlu/colab.sh  -O - | bash```
# 1 line Install NLU on Kaggle
```!wget https://setup.johnsnowlabs.com/nlu/kaggle.sh  -O - | bash```
# Install via PIP
```! pip install nlu pyspark==3.0.1```




<div class="h3-box" markdown="1">

## 200+ State of the Art Medical Models for NER, Entity Resolution, Relation Extraction, Assertion, Spark 3 and Python 3.8 support in  NLU 3.0 Release and much more
We are incredible excited to announce the release of `NLU 3.0.0` which makes most of John Snow Labs medical healthcare model available in just 1 line of code in NLU.
These models are the most accurate in their domains and highly scalable in Spark clusters.  
In addition, `Spark 3.0.X`  and `Spark 3.1.X ` is now supported, together with Python3.8

This is enabled by the the amazing [Spark NLP3.0.1](https://nlp.johnsnowlabs.com/docs/en/release_notes#300) and [Spark NLP for Healthcare 3.0.1](https://nlp.johnsnowlabs.com/docs/en/licensed_release_notes#301) releases.

# New Features
- Over 200 new models for the `healthcare` domain
- 6 new classes of models, Assertion, Sentence/Chunk Resolvers, Relation Extractors, Medical NER models, De-Identificator Models
- Spark 3.0.X and 3.1.X support
- Python 3.8 Support
- New Output level `relation`
- 1 Line to install NLU  just run `!wget https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/scripts/colab_setup.sh -O - | bash`
- [Various new EMR and Databricks versions supported](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/3.0.0)
- GPU Mode, more then 600% speedup by enabling GPU mode.
- Authorized mode for licensed features

## New Documentation
- [NLU for Healthcare Examples](https://nlu.johnsnowlabs.com/docs/en/examples_hc#usage-examples-of-nluload)
- [Instrunctions to authorize your environment to use Licensed features](https://nlu.johnsnowlabs.com/docs/en/examples_hc#authorize-access-to-licensed-features-and-install-healthcare-dependencies)


## New Notebooks
- [Medical Named Entity Extraction (NER) notebook](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/colab/healthcare/medical_named_entity_recognition/overview_medical_entity_recognizers.ipynb)
- [Relation extraction notebook](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/colab/healthcare/relation_extraction/overview_relation.ipynb)
- [Entity Resolution overview notebook](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/colab/healthcare/entity_resolution/entity_resolvers_overview.ipynb)
- [Assertion overview notebook](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/colab/healthcare/assertion/assertion_overview.ipynb)
- [De-Identification overview notebook](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/colab/healthcare/de_identification/DeIdentification_model_overview.ipynb)
- [Graph NLU tutorial](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/graph_ai_summit/Healthcare_Graph_NLU_COVID_Tigergraph.ipynb) for the  [GRAPH+AI Summit hosted by Tigergraph ](https://www.tigergraph.com/graphaisummit/)


## AssertionDLModels

| Language | nlu.load() reference                                         | Spark NLP Model reference          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| English  | [assert](https://nlp.johnsnowlabs.com/2021/01/26/assertion_dl_en.html) | [assertion_dl](https://nlp.johnsnowlabs.com/2021/01/26/assertion_dl_en.html)                   |
| English  | [assert.biobert](https://nlp.johnsnowlabs.com/2021/01/26/assertion_dl_biobert_en.html) | [assertion_dl_biobert](https://nlp.johnsnowlabs.com/2021/01/26/assertion_dl_biobert_en.html)                   |
| English  | [assert.healthcare](https://nlp.johnsnowlabs.com/2020/09/23/assertion_dl_healthcare_en.html) | [assertion_dl_healthcare](https://nlp.johnsnowlabs.com/2020/09/23/assertion_dl_healthcare_en.html)                   |
| English  | [assert.large](https://nlp.johnsnowlabs.com/2020/05/21/assertion_dl_large_en.html) | [assertion_dl_large](https://nlp.johnsnowlabs.com/2020/05/21/assertion_dl_large_en.html)                   |

##  New Word Embeddings

| Language | nlu.load() reference                                         | Spark NLP Model reference          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| English  | [embed.glove.clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_coarse_en.html) | [embeddings_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_coarse_en.html)                   |
| English  | [embed.glove.biovec](https://nlp.johnsnowlabs.com/2020/06/02/embeddings_biovec_en.html) | [embeddings_biovec](https://nlp.johnsnowlabs.com/2020/06/02/embeddings_biovec_en.html)                   |
| English  | [embed.glove.healthcare](https://nlp.johnsnowlabs.com/2020/03/26/embeddings_healthcare_en.html) | [embeddings_healthcare](https://nlp.johnsnowlabs.com/2020/03/26/embeddings_healthcare_en.html)                   |
| English  | [embed.glove.healthcare_100d](https://nlp.johnsnowlabs.com/2020/05/29/embeddings_healthcare_100d_en.html) | [embeddings_healthcare_100d](https://nlp.johnsnowlabs.com/2020/05/29/embeddings_healthcare_100d_en.html)                   |
| English  | en.embed.glove.icdoem | embeddings_icdoem          |
| English  | en.embed.glove.icdoem_2ng | embeddings_icdoem_2ng          |

## Sentence Entity resolvers

| Language | nlu.load() reference                                         | Spark NLP Model reference          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| English  | embed_sentence.biobert.mli | sbiobert_base_cased_mli          |
| English  | resolve | sbiobertresolve_cpt          |
| English  | resolve.cpt | sbiobertresolve_cpt          |
| English  | resolve.cpt.augmented | sbiobertresolve_cpt_augmented          |
| English  | resolve.cpt.procedures_augmented | sbiobertresolve_cpt_procedures_augmented          |
| English  | resolve.hcc.augmented | sbiobertresolve_hcc_augmented          |
| English  | [resolve.icd10cm](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_icd10cm_en.html) | [sbiobertresolve_icd10cm](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_icd10cm_en.html)                   |
| English  | [resolve.icd10cm.augmented](https://nlp.johnsnowlabs.com/2020/12/13/sbiobertresolve_icd10cm_augmented_en.html) | [sbiobertresolve_icd10cm_augmented](https://nlp.johnsnowlabs.com/2020/12/13/sbiobertresolve_icd10cm_augmented_en.html)                   |
| English  | [resolve.icd10cm.augmented_billable](https://nlp.johnsnowlabs.com/2021/02/06/sbiobertresolve_icd10cm_augmented_billable_hcc_en.html) | [sbiobertresolve_icd10cm_augmented_billable_hcc](https://nlp.johnsnowlabs.com/2021/02/06/sbiobertresolve_icd10cm_augmented_billable_hcc_en.html)                   |
| English  | [resolve.icd10pcs](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_icd10pcs_en.html) | [sbiobertresolve_icd10pcs](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_icd10pcs_en.html)                   |
| English  | [resolve.icdo](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_icdo_en.html) | [sbiobertresolve_icdo](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_icdo_en.html)                   |
| English  | [resolve.rxcui](https://nlp.johnsnowlabs.com/2020/12/11/sbiobertresolve_rxcui_en.html) | [sbiobertresolve_rxcui](https://nlp.johnsnowlabs.com/2020/12/11/sbiobertresolve_rxcui_en.html)                   |
| English  | [resolve.rxnorm](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_rxnorm_en.html) | [sbiobertresolve_rxnorm](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_rxnorm_en.html)                   |
| English  | [resolve.snomed](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_auxConcepts_en.html) | [sbiobertresolve_snomed_auxConcepts](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_auxConcepts_en.html)                   |
| English  | [resolve.snomed.aux_concepts](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_auxConcepts_en.html) | [sbiobertresolve_snomed_auxConcepts](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_auxConcepts_en.html)                   |
| English  | [resolve.snomed.aux_concepts_int](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_auxConcepts_int_en.html) | [sbiobertresolve_snomed_auxConcepts_int](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_auxConcepts_int_en.html)                   |
| English  | [resolve.snomed.findings](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_findings_en.html) | [sbiobertresolve_snomed_findings](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_findings_en.html)                   |
| English  | [resolve.snomed.findings_int](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_findings_int_en.html) | [sbiobertresolve_snomed_findings_int](https://nlp.johnsnowlabs.com/2020/11/27/sbiobertresolve_snomed_findings_int_en.html)                   |

## RelationExtractionModel

| Language | nlu.load() reference                                         | Spark NLP Model reference          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| English  | relation.posology | posology_re          |
| English  | [relation](https://nlp.johnsnowlabs.com/2021/02/04/redl_bodypart_direction_biobert_en.html) | [redl_bodypart_direction_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_bodypart_direction_biobert_en.html)                   |
| English  | [relation.bodypart.direction](https://nlp.johnsnowlabs.com/2021/02/04/redl_bodypart_direction_biobert_en.html) | [redl_bodypart_direction_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_bodypart_direction_biobert_en.html)                   |
| English  | [relation.bodypart.problem](https://nlp.johnsnowlabs.com/2021/02/04/redl_bodypart_problem_biobert_en.html) | [redl_bodypart_problem_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_bodypart_problem_biobert_en.html)                   |
| English  | [relation.bodypart.procedure](https://nlp.johnsnowlabs.com/2021/02/04/redl_bodypart_procedure_test_biobert_en.html) | [redl_bodypart_procedure_test_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_bodypart_procedure_test_biobert_en.html)                   |
| English  | [relation.chemprot](https://nlp.johnsnowlabs.com/2021/02/04/redl_chemprot_biobert_en.html) | [redl_chemprot_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_chemprot_biobert_en.html)                   |
| English  | [relation.clinical](https://nlp.johnsnowlabs.com/2021/02/04/redl_clinical_biobert_en.html) | [redl_clinical_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_clinical_biobert_en.html)                   |
| English  | [relation.date](https://nlp.johnsnowlabs.com/2021/02/04/redl_date_clinical_biobert_en.htmls) | [redl_date_clinical_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_date_clinical_biobert_en.htmls)                   |
| English  | [relation.drug_drug_interaction](https://nlp.johnsnowlabs.com/2021/02/04/redl_drug_drug_interaction_biobert_en.html) | [redl_drug_drug_interaction_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_drug_drug_interaction_biobert_en.html)                   |
| English  | [relation.humen_phenotype_gene](https://nlp.johnsnowlabs.com/2021/02/04/redl_human_phenotype_gene_biobert_en.html) | [redl_human_phenotype_gene_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_human_phenotype_gene_biobert_en.html)                   |
| English  | [relation.temporal_events](https://nlp.johnsnowlabs.com/2021/02/04/redl_temporal_events_biobert_en.html) | [redl_temporal_events_biobert](https://nlp.johnsnowlabs.com/2021/02/04/redl_temporal_events_biobert_en.html)                   |



## NERDLModels

| Language | nlu.load() reference                                         | Spark NLP Model reference          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
|English  | [med_ner.ade.clinical](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_clinical_en.html) | [ner_ade_clinical](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_clinical_en.html)                   |
| English  | [med_ner.ade.clinical_bert](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_clinicalbert_en.html) | [ner_ade_clinicalbert](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_clinicalbert_en.html)                   |
| English  | [med_ner.ade.ade_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_healthcare_en.html) | [ner_ade_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_ade_healthcare_en.html)                   |
| English  | [med_ner.anatomy](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_en.html) | [ner_anatomy](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_en.html)                   |
| English  | [med_ner.anatomy.biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_anatomy_biobert_en.html) | [ner_anatomy_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_anatomy_biobert_en.html)                   |
| English  | [med_ner.anatomy.coarse](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_coarse_en.html) | [ner_anatomy_coarse](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_coarse_en.html)                   |
| English  | [med_ner.anatomy.coarse_biobert](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_coarse_biobert_en.html) | [ner_anatomy_coarse_biobert](https://nlp.johnsnowlabs.com/2021/03/31/ner_anatomy_coarse_biobert_en.html)                   |
| English  | [med_ner.aspect_sentiment](https://nlp.johnsnowlabs.com/2021/03/31/ner_aspect_based_sentiment_en.html) | [ner_aspect_based_sentiment](https://nlp.johnsnowlabs.com/2021/03/31/ner_aspect_based_sentiment_en.html)                   |
| English  | [med_ner.bacterial_species](https://nlp.johnsnowlabs.com/2021/04/01/ner_bacterial_species_en.html) | [ner_bacterial_species](https://nlp.johnsnowlabs.com/2021/04/01/ner_bacterial_species_en.html)                   |
| English  | [med_ner.bionlp](https://nlp.johnsnowlabs.com/2021/03/31/ner_bionlp_en.html) | [ner_bionlp](https://nlp.johnsnowlabs.com/2021/03/31/ner_bionlp_en.html)                   |
| English  | [med_ner.bionlp.biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_bionlp_biobert_en.html) | [ner_bionlp_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_bionlp_biobert_en.html)                   |
| English  | [med_ner.cancer](https://nlp.johnsnowlabs.com/2021/03/31/ner_cancer_genetics_en.html) | [ner_cancer_genetics](https://nlp.johnsnowlabs.com/2021/03/31/ner_cancer_genetics_en.html)                   |
| Englishs | [med_ner.cellular](https://nlp.johnsnowlabs.com/2021/03/31/ner_cellular_en.html) | [ner_cellular](https://nlp.johnsnowlabs.com/2021/03/31/ner_cellular_en.html)                   |
| English  | [med_ner.cellular.biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_cellular_biobert_en.html) | [ner_cellular_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_cellular_biobert_en.html)                   |
| English  | [med_ner.chemicals](https://nlp.johnsnowlabs.com/2021/04/01/ner_chemicals_en.html) | [ner_chemicals](https://nlp.johnsnowlabs.com/2021/04/01/ner_chemicals_en.html)                   |
| English  | [med_ner.chemprot](https://nlp.johnsnowlabs.com/2021/04/01/ner_chemprot_biobert_en.html) | [ner_chemprot_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_chemprot_biobert_en.html)           |
| English  | [med_ner.chemprot.clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_chemprot_clinical_en.html) | [ner_chemprot_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_chemprot_clinical_en.html)           |
| English  | [med_ner.clinical](https://nlp.johnsnowlabs.com/2020/01/30/ner_clinical_en.html) | [ner_clinical](https://nlp.johnsnowlabs.com/2020/01/30/ner_clinical_en.html)           |
| English  | [med_ner.clinical.biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_clinical_biobert_en.html) | [ner_clinical_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_clinical_biobert_en.html)           |
| English  | med_ner.clinical.noncontrib | ner_clinical_noncontrib          |
| English  | [med_ner.diseases](https://nlp.johnsnowlabs.com/2021/03/31/ner_diseases_en.html) | [ner_diseases](https://nlp.johnsnowlabs.com/2021/03/31/ner_diseases_en.html)           |
| English  | [med_ner.diseases.biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_diseases_biobert_en.html) | [ner_diseases_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_diseases_biobert_en.html)           |
| English  | [med_ner.diseases.large](https://nlp.johnsnowlabs.com/2021/04/01/ner_diseases_large_en.html) | [ner_diseases_large](https://nlp.johnsnowlabs.com/2021/04/01/ner_diseases_large_en.html)           |
| English  | [med_ner.drugs](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_en.html) | [ner_drugs](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_en.html)           |
| English  | [med_ner.drugsgreedy](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_greedy_en.html) | [ner_drugs_greedy](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_greedy_en.html)           |
| English  | [med_ner.drugs.large](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_large_en.html) | [ner_drugs_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_drugs_large_en.html)           |
| English  | [med_ner.events_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_events_biobert_en.html) | [ner_events_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_events_biobert_en.html)           |
| English  | [med_ner.events_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_events_clinical_en.html) | [ner_events_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_events_clinical_en.html)           |
| English  | [med_ner.events_healthcre](https://nlp.johnsnowlabs.com/2021/04/01/ner_events_healthcare_en.html) | [ner_events_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_events_healthcare_en.html)           |
| English  | [med_ner.financial_contract](https://nlp.johnsnowlabs.com/2021/04/01/ner_financial_contract_en.html) | [ner_financial_contract](https://nlp.johnsnowlabs.com/2021/04/01/ner_financial_contract_en.html)           |
| English  | [med_ner.healthcare](https://nlp.johnsnowlabs.com/2021/03/31/ner_healthcare_de.html) | [ner_healthcare](https://nlp.johnsnowlabs.com/2021/03/31/ner_healthcare_de.html)           |
| English  | [med_ner.human_phenotype.gene_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_human_phenotype_gene_biobert_en.html) | [ner_human_phenotype_gene_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_human_phenotype_gene_biobert_en.html)           |
| English  | [med_ner.human_phenotype.gene_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_human_phenotype_gene_clinical_en.html) | [ner_human_phenotype_gene_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_human_phenotype_gene_clinical_en.html)           |
| English  | [med_ner.human_phenotype.go_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_human_phenotype_go_biobert_en.html) | [ner_human_phenotype_go_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_human_phenotype_go_biobert_en.html)           |
| English  | [med_ner.human_phenotype.go_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_human_phenotype_go_clinical_en.html) | [ner_human_phenotype_go_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_human_phenotype_go_clinical_en.html)           |
| English  | [med_ner.jsl](https://nlp.johnsnowlabs.com/2021/03/31/ner_jsl_en.html) | [ner_jsl](https://nlp.johnsnowlabs.com/2021/03/31/ner_jsl_en.html)           |
| English  | [med_ner.jsl.biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_jsl_biobert_en.html) | [ner_jsl_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_jsl_biobert_en.html)           |
| English  | [med_ner.jsl.enriched](https://nlp.johnsnowlabs.com/2021/03/31/ner_jsl_enriched_en.html) | [ner_jsl_enriched](https://nlp.johnsnowlabs.com/2021/03/31/ner_jsl_enriched_en.html)           |
| English  | [med_ner.jsl.enriched_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_jsl_enriched_biobert_en.html) | [ner_jsl_enriched_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_jsl_enriched_biobert_en.html)           |
| English  | [med_ner.measurements](https://nlp.johnsnowlabs.com/2021/04/01/ner_measurements_clinical_en.html) | [ner_measurements_clinical](https://nlp.johnsnowlabs.com/2021/04/01/ner_measurements_clinical_en.html)           |
| English  | [med_ner.medmentions](https://nlp.johnsnowlabs.com/2021/04/01/ner_medmentions_coarse_en.html) | [ner_medmentions_coarse](https://nlp.johnsnowlabs.com/2021/04/01/ner_medmentions_coarse_en.html)           |
| English  | [med_ner.posology](https://nlp.johnsnowlabs.com/2020/04/15/ner_posology_en.html) | [ner_posology](https://nlp.johnsnowlabs.com/2020/04/15/ner_posology_en.html)           |
| English  | [med_ner.posology.biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_biobert_en.html) | [ner_posology_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_biobert_en.html)           |
| English  | [med_ner.posology.greedy](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_greedy_en.html) | [ner_posology_greedy](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_greedy_en.html)           |
| English  | [med_ner.posology.healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_healthcare_en.html) | [ner_posology_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_healthcare_en.html)           |
| English  | [med_ner.posology.large](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_large_en.html) | [ner_posology_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_large_en.html)           |
| English  | [med_ner.posology.large_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_large_biobert_en.html) | [ner_posology_large_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_large_biobert_en.html)           |
| English  | [med_ner.posology.small](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_small_en.html) | [ner_posology_small](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_small_en.html)           |
| English  | [med_ner.radiology](https://nlp.johnsnowlabs.com/2021/03/31/ner_radiology_en.html) | [ner_radiology](https://nlp.johnsnowlabs.com/2021/03/31/ner_radiology_en.html)           |
| English  | [med_ner.radiology.wip_clinical](https://nlp.johnsnowlabs.com/2021/04/01/ner_radiology_wip_clinical_en.html) | [ner_radiology_wip_clinical](https://nlp.johnsnowlabs.com/2021/04/01/ner_radiology_wip_clinical_en.html)           |
| English  | [med_ner.risk_factors](https://nlp.johnsnowlabs.com/2021/03/31/ner_risk_factors_en.html) | [ner_risk_factors](https://nlp.johnsnowlabs.com/2021/03/31/ner_risk_factors_en.html)           |
| English  | [med_ner.risk_factors.biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_risk_factors_biobert_en.html) | [ner_risk_factors_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_risk_factors_biobert_en.html)           |
| English  | med_ner.i2b2 | nerdl_i2b2          |
| English  | [med_ner.tumour](https://nlp.johnsnowlabs.com/2021/04/01/nerdl_tumour_demo_en.html) | [nerdl_tumour_demo](https://nlp.johnsnowlabs.com/2021/04/01/nerdl_tumour_demo_en.html)           |
| English  | med_ner.jsl.wip.clinical | jsl_ner_wip_clinical          |
| English  | [med_ner.jsl.wip.clinical.greedy](https://nlp.johnsnowlabs.com/2021/03/31/jsl_ner_wip_clinical_en.html) | [jsl_ner_wip_greedy_clinical](https://nlp.johnsnowlabs.com/2021/03/31/jsl_ner_wip_clinical_en.html)           |
| English  | [med_ner.jsl.wip.clinical.modifier](https://nlp.johnsnowlabs.com/2021/04/01/jsl_ner_wip_modifier_clinical_en.html) | [jsl_ner_wip_modifier_clinical](https://nlp.johnsnowlabs.com/2021/04/01/jsl_ner_wip_modifier_clinical_en.html)           |
| English  | [med_ner.jsl.wip.clinical.rd](https://nlp.johnsnowlabs.com/2021/04/01/jsl_rd_ner_wip_greedy_clinical_en.html) | [jsl_rd_ner_wip_greedy_clinical](https://nlp.johnsnowlabs.com/2021/04/01/jsl_rd_ner_wip_greedy_clinical_en.html)           |


## De-Identification Models

| Language | nlu.load() reference                                         | Spark NLP Model reference          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| English  | [med_ner.deid.augmented](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_augmented_en.html) | [ner_deid_augmented](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_augmented_en.html)           |
| English  | [med_ner.deid.biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_biobert_en.html) | [ner_deid_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_biobert_en.html)           |
| English  | [med_ner.deid.enriched](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_enriched_en.html) | [ner_deid_enriched](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_enriched_en.html)           |
| English  | [med_ner.deid.enriched_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_enriched_biobert_en.html) | [ner_deid_enriched_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_enriched_biobert_en.html)           |
| English  | [med_ner.deid.large](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_large_en.html) | [ner_deid_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_large_en.html)           |
| English  | [med_ner.deid.sd](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_sd_en.html) | [ner_deid_sd](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_sd_en.html)           |
| English  | [med_ner.deid.sd_large](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_sd_large_en.html) | [ner_deid_sd_large](https://nlp.johnsnowlabs.com/2021/04/01/ner_deid_sd_large_en.html)           |
| English  | med_ner.deid | nerdl_deid          |
| English  | med_ner.deid.synthetic | ner_deid_synthetic          |
| English  | [med_ner.deid.dl](https://nlp.johnsnowlabs.com/2021/03/31/ner_deidentify_dl_en.html) | [ner_deidentify_dl](https://nlp.johnsnowlabs.com/2021/03/31/ner_deidentify_dl_en.html)           |
| English  | [en.de_identify](https://nlp.johnsnowlabs.com/2019/06/04/deidentify_rb_en.html) | [deidentify_rb](https://nlp.johnsnowlabs.com/2019/06/04/deidentify_rb_en.html)           |
| English  | de_identify.rules | deid_rules          |
| English  | [de_identify.clinical](https://nlp.johnsnowlabs.com/2021/01/29/deidentify_enriched_clinical_en.html) | [deidentify_enriched_clinical](https://nlp.johnsnowlabs.com/2021/01/29/deidentify_enriched_clinical_en.html)           |
| English  | [de_identify.large](https://nlp.johnsnowlabs.com/2020/08/04/deidentify_large_en.html) | [deidentify_large](https://nlp.johnsnowlabs.com/2020/08/04/deidentify_large_en.html)           |
| English  | [de_identify.rb](https://nlp.johnsnowlabs.com/2019/06/04/deidentify_rb_en.html) | [deidentify_rb](https://nlp.johnsnowlabs.com/2019/06/04/deidentify_rb_en.html)           |
| English  | de_identify.rb_no_regex | deidentify_rb_no_regex          |



# Chunk resolvers

| Language | nlu.load() reference                                         | Spark NLP Model reference          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| English  | [resolve_chunk.athena_conditions](https://nlp.johnsnowlabs.com/2020/09/16/chunkresolve_athena_conditions_healthcare_en.html) | [chunkresolve_athena_conditions_healthcare](https://nlp.johnsnowlabs.com/2020/09/16/chunkresolve_athena_conditions_healthcare_en.html)           |
| English  | [resolve_chunk.cpt_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_cpt_clinical_en.html) | [chunkresolve_cpt_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_cpt_clinical_en.html)           |
| English  | [resolve_chunk.icd10cm.clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_clinical_en.html) | [chunkresolve_icd10cm_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_clinical_en.html)           |
| English  | [resolve_chunk.icd10cm.diseases_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_diseases_clinical_en.html) | [chunkresolve_icd10cm_diseases_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_diseases_clinical_en.html)           |
| English  | resolve_chunk.icd10cm.hcc_clinical | chunkresolve_icd10cm_hcc_clinical          |
| English  | resolve_chunk.icd10cm.hcc_healthcare | chunkresolve_icd10cm_hcc_healthcare          |
| English  | [resolve_chunk.icd10cm.injuries](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_injuries_clinical_en.html) | [chunkresolve_icd10cm_injuries_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_injuries_clinical_en.html)           |
| English  | [resolve_chunk.icd10cm.musculoskeletal](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_musculoskeletal_clinical_en.html) | [chunkresolve_icd10cm_musculoskeletal_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_musculoskeletal_clinical_en.html)           |
| English  | [resolve_chunk.icd10cm.neoplasms](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_neoplasms_clinical_en.html) | [chunkresolve_icd10cm_neoplasms_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10cm_neoplasms_clinical_en.html)           |
| English  | [resolve_chunk.icd10cm.poison](https://nlp.johnsnowlabs.com/2020/04/28/chunkresolve_icd10cm_poison_ext_clinical_en.html) | [chunkresolve_icd10cm_poison_ext_clinical](https://nlp.johnsnowlabs.com/2020/04/28/chunkresolve_icd10cm_poison_ext_clinical_en.html)           |
| English  | [resolve_chunk.icd10cm.puerile](https://nlp.johnsnowlabs.com/2020/04/28/chunkresolve_icd10cm_puerile_clinical_en.html) | [chunkresolve_icd10cm_puerile_clinical](https://nlp.johnsnowlabs.com/2020/04/28/chunkresolve_icd10cm_puerile_clinical_en.html)           |
| English  | resolve_chunk.icd10pcs.clinical | chunkresolve_icd10pcs_clinical          |
| English  | [resolve_chunk.icdo.clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10pcs_clinical_en.html) | [chunkresolve_icdo_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_icd10pcs_clinical_en.html)           |
| English  | [resolve_chunk.loinc](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_loinc_clinical_en.html) | [chunkresolve_loinc_clinical](https://nlp.johnsnowlabs.com/2021/04/02/chunkresolve_loinc_clinical_en.html)           |
| English  | [resolve_chunk.rxnorm.cd](https://nlp.johnsnowlabs.com/2020/07/27/chunkresolve_rxnorm_cd_clinical_en.html) | [chunkresolve_rxnorm_cd_clinical](https://nlp.johnsnowlabs.com/2020/07/27/chunkresolve_rxnorm_cd_clinical_en.html)           |
| English  | resolve_chunk.rxnorm.in | chunkresolve_rxnorm_in_clinical          |
| English  | resolve_chunk.rxnorm.in_healthcare | chunkresolve_rxnorm_in_healthcare          |
| English  | [resolve_chunk.rxnorm.sbd](https://nlp.johnsnowlabs.com/2020/07/27/chunkresolve_rxnorm_sbd_clinical_en.html) | [chunkresolve_rxnorm_sbd_clinical](https://nlp.johnsnowlabs.com/2020/07/27/chunkresolve_rxnorm_sbd_clinical_en.html)           |
| English  | [resolve_chunk.rxnorm.scd](https://nlp.johnsnowlabs.com/2020/07/27/chunkresolve_rxnorm_scd_clinical_en.html) | [chunkresolve_rxnorm_scd_clinical](https://nlp.johnsnowlabs.com/2020/07/27/chunkresolve_rxnorm_scd_clinical_en.html)           |
| English  | resolve_chunk.rxnorm.scdc | chunkresolve_rxnorm_scdc_clinical          |
| English  | resolve_chunk.rxnorm.scdc_healthcare | chunkresolve_rxnorm_scdc_healthcare          |
| English  | [resolve_chunk.rxnorm.xsmall.clinical](https://nlp.johnsnowlabs.com/2020/06/24/chunkresolve_rxnorm_xsmall_clinical_en.html) | [chunkresolve_rxnorm_xsmall_clinical](https://nlp.johnsnowlabs.com/2020/06/24/chunkresolve_rxnorm_xsmall_clinical_en.html)           |
| English  | [resolve_chunk.snomed.findings](https://nlp.johnsnowlabs.com/2020/06/20/chunkresolve_snomed_findings_clinical_en.html) | [chunkresolve_snomed_findings_clinical](https://nlp.johnsnowlabs.com/2020/06/20/chunkresolve_snomed_findings_clinical_en.html)           |


# New Classifiers

| Language | nlu.load() reference                                         | Spark NLP Model reference          |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| English  | classify.icd10.clinical | classifier_icd10cm_hcc_clinical          |
| English  | classify.icd10.healthcare | classifier_icd10cm_hcc_healthcare          |
| English  | [classify.ade.biobert](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_ade_biobert_en.html) | [classifierdl_ade_biobert](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_ade_biobert_en.html)           |
| English  | [classify.ade.clinical](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_ade_clinicalbert_en.html) | [classifierdl_ade_clinicalbert](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_ade_clinicalbert_en.html)           |
| English  | [classify.ade.conversational](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_ade_conversational_biobert_en.html) | [classifierdl_ade_conversational_biobert](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_ade_conversational_biobert_en.html)           |
| English  | [classify.gender.biobert](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_gender_biobert_en.html) | [classifierdl_gender_biobert](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_gender_biobert_en.html)           |
| English  | [classify.gender.sbert](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_gender_sbert_en.html) | [classifierdl_gender_sbert](https://nlp.johnsnowlabs.com/2021/01/21/classifierdl_gender_sbert_en.html)           |
| English  | classify.pico | classifierdl_pico_biobert          |


# German Medical models

| nlu.load() reference                                         | Spark NLP Model reference          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [embed]    | w2v_cc_300d|
| [embed.w2v]    | w2v_cc_300d|
| [resolve_chunk]    | chunkresolve_ICD10GM|
| [resolve_chunk.icd10gm]    | chunkresolve_ICD10GM|
| resolve_chunk.icd10gm.2021    | chunkresolve_ICD10GM_2021|
| med_ner.legal   | ner_legal|
| med_ner    | ner_healthcare|
| med_ner.healthcare    | ner_healthcare|
| med_ner.healthcare_slim    | ner_healthcare_slim|
| med_ner.traffic    | ner_traffic|

# Spanish Medical models
| nlu.load() reference                                         | Spark NLP Model reference          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [embed.scielo.150d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielo_150d_es.html) | [embeddings_scielo_150d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielo_150d_es.html)| 
| [embed.scielo.300d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielo_300d_es.html)   | [embeddings_scielo_300d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielo_300d_es.html)| 
| [embed.scielo.50d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielo_50d_es.html)  | [embeddings_scielo_50d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielo_50d_es.html)| 
| [embed.scielowiki.150d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielowiki_150d_es.html)   | [embeddings_scielowiki_150d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielowiki_150d_es.html)| 
| [embed.scielowiki.300d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielowiki_300d_es.html)   | [embeddings_scielowiki_300d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielowiki_300d_es.html)| 
| [embed.scielowiki.50d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielowiki_50d_es.html)   | [embeddings_scielowiki_50d](https://nlp.johnsnowlabs.com/2020/05/26/embeddings_scielowiki_50d_es.html)| 
| [embed.sciwiki.150d](https://nlp.johnsnowlabs.com/2020/05/27/embeddings_sciwiki_150d_es.html)   | [embeddings_sciwiki_150d](https://nlp.johnsnowlabs.com/2020/05/27/embeddings_sciwiki_150d_es.html)| 
| [embed.sciwiki.300d](https://nlp.johnsnowlabs.com/2020/05/27/embeddings_sciwiki_300d_es.html)   | [embeddings_sciwiki_300d](https://nlp.johnsnowlabs.com/2020/05/27/embeddings_sciwiki_300d_es.html)| 
| [embed.sciwiki.50d](https://nlp.johnsnowlabs.com/2020/05/27/embeddings_sciwiki_50d_es.html)   | [embeddings_sciwiki_50d](https://nlp.johnsnowlabs.com/2020/05/27/embeddings_sciwiki_50d_es.html)| 
| [med_ner](https://nlp.johnsnowlabs.com/2021/03/31/ner_diag_proc_es.html)   |  [ner_diag_proc](https://nlp.johnsnowlabs.com/2021/03/31/ner_diag_proc_es.html)| 
| [med_ner.neoplasm](https://nlp.johnsnowlabs.com/2021/03/31/ner_neoplasms_es.html)  | [ner_neoplasms](https://nlp.johnsnowlabs.com/2021/03/31/ner_neoplasms_es.html)| 
| [med_ner.diag_proc](https://nlp.johnsnowlabs.com/2021/03/31/ner_diag_proc_es.html)  | [ner_diag_proc](https://nlp.johnsnowlabs.com/2021/03/31/ner_diag_proc_es.html)| 

# GPU Mode
You can now enable NLU GPU mode by setting `gpu=true` while loading a model. I.e. `nlu.load('train.sentiment' gpu=True)` . If must resart you kernel, if you already loaded a nlu pipeline withouth GPU mode.

# Output Level Relation
This new output level is used for relation extractors and will give you 1 row per relation extracted.


# Bug fixes
- Fixed a bug that caused loading NLU models in offline mode not to work in some occasions


# 1 line Install NLU
```!wget https://raw.githubusercontent.com/JohnSnowLabs/nlu/master/scripts/colab_setup.sh -O - | bash```

# Install via PIP 
```! pip install nlu pyspark==3.0.1```


## Additional NLU ressources

- [NLU Website](https://nlu.johnsnowlabs.com/)
- [All NLU Tutorial Notebooks](https://nlu.johnsnowlabs.com/docs/en/notebooks)
- [NLU Videos and Blogposts on NLU](https://nlp.johnsnowlabs.com/learn#pythons-nlu-library)
- [NLU on Github](https://github.com/JohnSnowLabs/nlu)
- [Suggestions or Questions? Contact us in Slack!](https://join.slack.com/t/spark-nlp/shared_invite/zt-lutct9gm-kuUazcyFKhuGY3_0AMkxqA)








<div class="h3-box" markdown="1">

# Intent and Action Classification,  analyze Chinese News and the Crypto market, train a classifier that understands 100+ languages, translate between 200 + languages, answer questions, summarize text, and much more in NLU 1.1.3

## NLU 1.1.3 Release Notes
We are very excited to announce that the latest NLU release comes with a new pretrained Intent Classifier and NER Action Extractor for text related to
music, restaurants, and movies trained on the SNIPS dataset. Make sure to check out the models hub and the easy 1-liners for more info!

In addition to that, new NER and Embedding models for Bengali are now available

Finally, there is a new NLU Webinar with 9 accompanying tutorial notebooks which teach you  a lot of things and is segmented into the following parts :

- Part1: Easy 1 Liners
  - Spell checking/Sentiment/POS/NER/ BERTtology embeddings
- Part2: Data analysis and NLP tasks on [Crypto News Headline dataset](https://www.kaggle.com/kashnitsky/news-about-major-cryptocurrencies-20132018-40k)
  - Preprocessing and extracting Emotions, Keywords, Named Entities and visualize them
- Part3: NLU Multi-Lingual 1 Liners with [Microsoft's Marian Models](https://marian-nmt.github.io/publications/)
  - Translate between 200+ languages (and classify lang afterward)
- Part 4: Data analysis and NLP tasks on [Chinese News Article Dataset](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/4_Unsupervise_Chinese_Keyword_Extraction_NER_and_Translation_from_Chinese_News.ipynb)
  - Word Segmentation, Lemmatization, Extract Keywords, Named Entities and translate to english
- Part 5: Train a sentiment Classifier that understands 100+ Languages
  - Train on a french sentiment dataset and predict the sentiment of 100+ languages with [language-agnostic BERT Sentence Embedding](https://arxiv.org/abs/2007.01852)
- Part 6: Question answering, Summarization, Squad and more with [Google's T5](https://arxiv.org/abs/1910.10683)
  - T5 Question answering and 18 + other NLP tasks ([SQUAD](https://arxiv.org/abs/1606.05250) / [GLUE](https://arxiv.org/abs/1804.07461) / [SUPER GLUE](https://super.gluebenchmark.com/))


### New Models

#### NLU 1.1.3 New Non-English Models

| Language | nlu.load() reference                                         | Spark NLP Model reference                                    | Type                  |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------- |
| Bengali  | [bn.ner.cc_300d](https://nlp.johnsnowlabs.com/2021/02/10/bengali_cc_300d_bn.html) | [ bengaliner_cc_300d](https://nlp.johnsnowlabs.com/2021/02/10/bengali_cc_300d_bn.html) | NerDLModel    |
| Bengali  | [bn.embed](https://nlp.johnsnowlabs.com/2021/02/10/bengaliner_cc_300d_bn.html) | [bengali_cc_300d](https://nlp.johnsnowlabs.com/2021/02/10/bengaliner_cc_300d_bn.html) | NerDLModel            |
| Bengali  | [bn.embed.cc_300d](https://nlp.johnsnowlabs.com/2021/02/10/bengaliner_cc_300d_bn.html) | [bengali_cc_300d](https://nlp.johnsnowlabs.com/2021/02/10/bengaliner_cc_300d_bn.html) | Word Embeddings Model (Alias)    |
| Bengali  | [bn.embed.glove](https://nlp.johnsnowlabs.com/2021/02/10/bengaliner_cc_300d_bn.html) | [bengali_cc_300d](https://nlp.johnsnowlabs.com/2021/02/10/bengaliner_cc_300d_bn.html) |  Word Embeddings Model (Alias)|





#### NLU 1.1.3 New English Models

|Language | nlu.load() reference | Spark NLP Model reference | Type |
|---------|---------------------|----------------------------|------|
| English | [en.classify.snips](https://nlp.johnsnowlabs.com/2021/02/15/nerdl_snips_100d_en.html) |[nerdl_snips_100d](https://nlp.johnsnowlabs.com/2021/02/15/nerdl_snips_100d_en.html)     | NerDLModel |
| English | [en.ner.snips](https://nlp.johnsnowlabs.com/2021/02/15/classifierdl_use_snips_en.html) |[classifierdl_use_snips](https://nlp.johnsnowlabs.com/2021/02/15/classifierdl_use_snips_en.html)|ClassifierDLModel|




### New NLU Webinar
#### [State-of-the-art Natural Language Processing for 200+ Languages with 1 Line of code](https://events.johnsnowlabs.com/state-of-the-art-natural-language-processing-for-200-languages-with-1-line-of-code)


##### Talk Abstract
Learn to harness the power of 1,000+ production-grade & scalable NLP models for 200+ languages - all available with just 1 line of Python code by leveraging the open-source NLU library, which is powered by the widely popular Spark NLP.

John Snow Labs has delivered over 80 releases of Spark NLP to date, making it the most widely used NLP library in the enterprise and providing the AI community with state-of-the-art accuracy and scale for a variety of common NLP tasks. The most recent releases include pre-trained models for over 200 languages - including languages that do not use spaces for word segmentation algorithms like Chinese, Japanese, and Korean, and languages written from right to left like Arabic, Farsi, Urdu, and Hebrew. All software and models are free and open source under an Apache 2.0 license.

This webinar will show you how to leverage the multi-lingual capabilities of Spark NLP & NLU - including automated language detection for up to 375 languages, and the ability to perform translation, named entity recognition, stopword removal, lemmatization, and more in a variety of language families. We will create Python code in real-time and solve these problems in just 30 minutes. The notebooks will then be made freely available online.

You can watch the [video here,](https://events.johnsnowlabs.com/state-of-the-art-natural-language-processing-for-200-languages-with-1-line-of-code)

### NLU 1.1.3 New Notebooks and tutorials


#### New Webinar Notebooks

1. [NLU basics, easy 1-liners (Spellchecking, sentiment, NER, POS, BERT](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/0_liners_intro.ipynb)
2. [Analyze Crypto News dataset with Keyword extraction, NER, Emotional distribution, and stemming](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/1_NLU_base_features_on_dataset_with_YAKE_Lemma_Stemm_classifiers_NER_.ipynb)
3. [Translate Crypto News dataset between 300 Languages with the Marian Model (German, French, Hebrew examples)](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/2_multilingual_translation_with_marian_intro.ipynb)
4. [Translate Crypto News dataset between 300 Languages with the Marian Model (Hindi, Russian, Chinese examples)](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/3_more_multi_lingual_NLP_translation_Asian_languages_with_Marian.ipynb)
5. [Analyze Chinese News Headlines with Chinese Word Segmentation, Lemmatization, NER, and Keyword extraction](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/4_Unsupervise_Chinese_Keyword_Extraction_NER_and_Translation_from_Chinese_News.ipynb)
6. [Train a Sentiment Classifier that will understand 100+ languages on just a French Dataset with the powerful Language Agnostic Bert Embeddings](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/5_multi_lingual_sentiment_classifier_training_for_over_100_languages.ipynb)
7. [Summarize text and Answer Questions with T5](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/6_T5_question_answering_and_Text_summarization.ipynb)
8. [Solve any task in 1 line from SQUAD, GLUE and SUPER GLUE with T5](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/7_T5_SQUAD_GLUE_SUPER_GLUE_TASKS.ipynb)
9. [Overview of models for various languages](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/multi_lingual_webinar/8_Multi_lingual_ner_pos_stop_words_sentiment_pretrained.ipynb)





#### New easy NLU 1-liners in NLU 1.1.3

####  [Detect actions in general commands related to music, restaurant, movies.](https://nlp.johnsnowlabs.com/2021/02/15/nerdl_snips_100d_en.html)


```python
nlu.load("en.classify.snips").predict("book a spot for nona gray  myrtle and alison at a top-rated brasserie that is distant from wilson av on nov  the 4th  2030 that serves ouzeri",output_level = "document")
```

outputs :

|                                               ner_confidence | entities                                                     | document                                                     | Entities_Classes                                             |
| -----------------------------------------------------------: | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [1.0, 1.0, 0.9997000098228455, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9990000128746033, 1.0, 1.0, 1.0, 0.9965000152587891, 0.9998999834060669, 0.9567000269889832, 1.0, 1.0, 1.0, 0.9980000257492065, 0.9991999864578247, 0.9988999962806702, 1.0, 1.0, 0.9998999834060669] | ['nona gray myrtle and alison', 'top-rated', 'brasserie', 'distant', 'wilson av', 'nov the 4th 2030', 'ouzeri'] | book a spot for nona gray myrtle and alison at a top-rated brasserie that is distant from wilson av on nov the 4th 2030 that serves ouzeri | ['party_size_description', 'sort', 'restaurant_type', 'spatial_relation', 'poi', 'timeRange', 'cuisine'] |

####  [Named Entity Recognition (NER) Model in Bengali (bengaliner_cc_300d)](https://nlp.johnsnowlabs.com/2021/02/10/bengaliner_cc_300d_bn.html)


```python
# Bengali for: 'Iajuddin Ahmed passed Matriculation from Munshiganj High School in 1947 and Intermediate from Munshiganj Horganga College in 1950.'
nlu.load("bn.ner.cc_300d").predict("১৯৪৮ সালে ইয়াজউদ্দিন আহম্মেদ মুন্সিগঞ্জ উচ্চ বিদ্যালয় থেকে মেট্রিক পাশ করেন এবং ১৯৫০ সালে মুন্সিগঞ্জ হরগঙ্গা কলেজ থেকে ইন্টারমেডিয়েট পাশ করেন",output_level = "document")
```

outputs :

| ner_confidence                                                                                                                                                                                                                                                                                                                                                                                                                       | entities                                                                           | Entities_Classes   | document                                                                                                                         |
|---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:--------------------------------------|
| [0.9987999796867371, 0.9854000210762024, 0.8604000210762024, 0.6686999797821045, 0.5289999842643738, 0.7009999752044678, 0.7684999704360962, 0.9979000091552734, 0.9976000189781189, 0.9930999875068665, 0.9994000196456909, 0.9879000186920166, 0.7407000064849854, 0.9215999841690063, 0.7657999992370605, 0.39419999718666077, 0.9124000072479248, 0.9932000041007996, 0.9919999837875366, 0.995199978351593, 0.9991999864578247] | ['সালে', 'ইয়াজউদ্দিন আহম্মেদ', 'মুন্সিগঞ্জ উচ্চ বিদ্যালয়', 'সালে', 'মুন্সিগঞ্জ হরগঙ্গা কলেজ'] | ['TIME', 'PER', 'ORG', 'TIME', 'ORG'] | ১৯৪৮ সালে ইয়াজউদ্দিন আহম্মেদ মুন্সিগঞ্জ উচ্চ বিদ্যালয় থেকে মেট্রিক পাশ করেন এবং ১৯৫০ সালে মুন্সিগঞ্জ হরগঙ্গা কলেজ থেকে ইন্টারমেডিয়েট পাশ করেন |

#### [Identify intent in general text - SNIPS dataset](https://nlp.johnsnowlabs.com/2021/02/15/classifierdl_use_snips_en.html)


```python
nlu.load("en.ner.snips").predict("I want to bring six of us to a bistro in town that serves hot chicken sandwich that is within the same area",output_level = "document")
```

outputs :


| document | snips | snips_confidence|
|----------|------|------------------|
| I want to bring six of us to a bistro in town that serves hot chicken sandwich that is within the same area | BookRestaurant |                  1 |


#### [Word Embeddings for Bengali (bengali_cc_300d)](https://nlp.johnsnowlabs.com/2021/02/10/bengali_cc_300d_bn.html)




```python
# Bengali for : 'Iajuddin Ahmed passed Matriculation from Munshiganj High School in 1947 and Intermediate from Munshiganj Horganga College in 1950.'
nlu.load("bn.embed").predict("১৯৪৮ সালে ইয়াজউদ্দিন আহম্মেদ মুন্সিগঞ্জ উচ্চ বিদ্যালয় থেকে মেট্রিক পাশ করেন এবং ১৯৫০ সালে মুন্সিগঞ্জ হরগঙ্গা কলেজ থেকে ইন্টারমেডিয়েট পাশ করেন",output_level = "document")
```

outputs :

|                                                     document | bn_embed_embeddings                                          |
| -----------------------------------------------------------: | :----------------------------------------------------------- |
| ১৯৪৮ সালে ইয়াজউদ্দিন আহম্মেদ মুন্সিগঞ্জ উচ্চ বিদ্যালয় থেকে মেট্রিক পাশ করেন এবং ১৯৫০ সালে মুন্সিগঞ্জ হরগঙ্গা কলেজ থেকে ইন্টারমেডিয়েট পাশ করেন | [-0.0828      0.0683      0.0215     ...  0.0679     -0.0484...] |



### NLU 1.1.3 Enhancements
- Added automatic conversion  to Sentence Embeddings of Word Embeddings when there is no Sentence Embedding Avaiable and a model needs the converted version to run.


### NLU 1.1.3 Bug Fixes
- Fixed a bug that caused `ur.sentiment` NLU pipeline to build incorrectly
- Fixed a bug that caused `sentiment.imdb.glove` NLU pipeline to build incorrectly
- Fixed a bug that caused `en.sentiment.glove.imdb` NLU pipeline to build incorrectly
- Fixed a bug that caused Spark 2.3.X environments to crash.

### NLU Installation

```bash
# PyPi
!pip install nlu pyspark==2.4.7
#Conda
# Install NLU from Anaconda/Conda
conda install -c johnsnowlabs nlu
```

### Additional NLU ressources

- [NLU Website](https://nlu.johnsnowlabs.com/)
- [All NLU Tutorial Notebooks](https://nlu.johnsnowlabs.com/docs/en/notebooks)
- [NLU Videos and Blogposts on NLU](https://nlp.johnsnowlabs.com/learn#pythons-nlu-library)
- [NLU on Github](https://github.com/JohnSnowLabs/nlu)
- [Suggestions or Questions? Contact us in Slack!](https://join.slack.com/t/spark-nlp/shared_invite/zt-lutct9gm-kuUazcyFKhuGY3_0AMkxqA)




## NLU 1.1.2 Release Notes
### Hindi  WordEmbeddings , Bengali Named Entity Recognition (NER), 30+ new models, analyze Crypto news with John Snow Labs NLU 1.1.2 

We are very happy to announce NLU 1.1.2 has been released with the integration of 30+ models and pipelines Bengali Named Entity Recognition, Hindi Word Embeddings,
and state-of-the-art transformer based OntoNotes models and pipelines from the [incredible Spark NLP 2.7.3 Release](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.3) in addition to a few bugfixes.  
In addition to that, there is a [new NLU Webinar video](https://www.youtube.com/watch?t=2141&v=hJR9m3NYnwk&feature=youtu.be) showcasing in detail 
how to use NLU to analyze a crypto news dataset to extract keywords unsupervised and predict sentimential/emotional distributions of the dataset and much more!

### [Python's NLU library: 1,000+ models, 200+ Languages, State of the Art Accuracy, 1 Line of code - NLU NYC/DC NLP Meetup Webinar](https://www.youtube.com/watch?t=2141&v=hJR9m3NYnwk&feature=youtu.be)
Using just 1 line of Python code by leveraging the NLU library, which is powered by the award-winning Spark NLP.

This webinar covers, using live coding in real-time,
how to deliver summarization, translation, unsupervised keyword extraction, emotion analysis,
question answering, spell checking, named entity recognition, document classification, and other common NLP tasks. T
his is all done with a single line of code, that works directly on Python strings or pandas data frames.
Since NLU is based on Spark NLP, no code changes are required to scale processing to multi-core or cluster environment - integrating natively with Ray, Dask, or Spark data frames.

The recent releases for Spark NLP and NLU include pre-trained models for over 200 languages and language detection for 375 languages.
This includes 20 languages families; non-Latin alphabets; languages that do not use spaces for word segmentation like
Chinese, Japanese, and Korean; and languages written from right to left like Arabic, Farsi, Urdu, and Hebrew.
We'll also cover some of the algorithms and models that are included. The code notebooks will be freely available online.

 

### NLU 1.1.2 New Models  and Pipelines

#### NLU 1.1.2 New Non-English Models

|Language | nlu.load() reference | Spark NLP Model reference | Type |
|---------|---------------------|----------------------------|------|
|Bengali | [bn.ner](https://nlp.johnsnowlabs.com/2021/01/27/ner_jifs_glove_840B_300d_bn.html) |[ner_jifs_glove_840B_300d](https://nlp.johnsnowlabs.com/2021/01/27/ner_jifs_glove_840B_300d_bn.html) | Word Embeddings Model (Alias) |
| Bengali  | [bn.ner.glove](https://nlp.johnsnowlabs.com/2021/01/27/ner_jifs_glove_840B_300d_bn.html) | [ner_jifs_glove_840B_300d](https://nlp.johnsnowlabs.com/2021/01/27/ner_jifs_glove_840B_300d_bn.html) | Word Embeddings Model (Alias) |
|Hindi|[hi.embed](https://nlp.johnsnowlabs.com/2021/02/03/hindi_cc_300d_hi.html)|[hindi_cc_300d](https://nlp.johnsnowlabs.com/2021/02/03/hindi_cc_300d_hi.html)|NerDLModel|
|Bengali | [bn.lemma](https://nlp.johnsnowlabs.com/2021/01/20/lemma_bn.html) |[lemma](https://nlp.johnsnowlabs.com/2021/01/20/lemma_bn.html) | Lemmatizer                    |
|Japanese | [ja.lemma](https://nlp.johnsnowlabs.com/2021/01/15/lemma_ja.html) |[lemma](https://nlp.johnsnowlabs.com/2021/01/15/lemma_ja.html) | Lemmatizer                    |
|Bihari | [bh.lemma](https://nlp.johnsnowlabs.com/2021/01/18/lemma_bh.html) |[lemma](https://nlp.johnsnowlabs.com/2021/01/18/lemma_bh.html) | Lemma                    |
|Amharic | [am.lemma](https://nlp.johnsnowlabs.com/2021/01/20/lemma_am.html) |[lemma](https://nlp.johnsnowlabs.com/2021/01/20/lemma_am.html) | Lemma                    |

#### NLU 1.1.2 New English Models and Pipelines

|Language | nlu.load() reference | Spark NLP Model reference | Type |
|---------|---------------------|----------------------------|------|
| English | [en.ner.onto.bert.small_l2_128](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L2_128_en.html) |[onto_small_bert_L2_128](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L2_128_en.html)     | NerDLModel |
| English | [en.ner.onto.bert.small_l4_256](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L4_256_en.html) |[onto_small_bert_L4_256](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L4_256_en.html)     | NerDLModel |
| English | [en.ner.onto.bert.small_l4_512](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L4_512_en.html) |[onto_small_bert_L4_512](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L4_512_en.html)     | NerDLModel |
| English | [en.ner.onto.bert.small_l8_512](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L8_512_en.html) |[onto_small_bert_L8_512](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L8_512_en.html)     | NerDLModel |
| English | [en.ner.onto.bert.cased_base](https://nlp.johnsnowlabs.com/2020/12/05/onto_bert_base_cased_en.html) |[onto_bert_base_cased](https://nlp.johnsnowlabs.com/2020/12/05/onto_bert_base_cased_en.html)     | NerDLModel |
| English | [en.ner.onto.bert.cased_large](https://nlp.johnsnowlabs.com/2020/12/05/onto_bert_large_cased_en.html) |[onto_bert_large_cased](https://nlp.johnsnowlabs.com/2020/12/05/onto_bert_large_cased_en.html)     | NerDLModel |
| English | [en.ner.onto.electra.uncased_small](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_small_uncased_en.html) |[onto_electra_small_uncased](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_small_uncased_en.html)     | NerDLModel |
| English  | [en.ner.onto.electra.uncased_base](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_base_uncased_en.html) |[onto_electra_base_uncased](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_base_uncased_en.html)     | NerDLModel |
| English | [en.ner.onto.electra.uncased_large](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_large_uncased_en.html) |[onto_electra_large_uncased](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_large_uncased_en.html)     | NerDLModel |
| English | [en.ner.onto.bert.tiny](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_tiny_en.html) | [onto_recognize_entities_bert_tiny](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_tiny_en.html) | Pipeline |
| English | [en.ner.onto.bert.mini](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_mini_en.html) |[onto_recognize_entities_bert_mini](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_mini_en.html)     | Pipeline |
| English | [en.ner.onto.bert.small](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_small_en.html) | [onto_recognize_entities_bert_small](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_small_en.html) | Pipeline |
| English | [en.ner.onto.bert.medium](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_medium_en.html) |[onto_recognize_entities_bert_medium](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_medium_en.html)     | Pipeline |
| English | [en.ner.onto.bert.base](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_base_en.html) |[onto_recognize_entities_bert_base](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_base_en.html)     | Pipeline |
|English|[en.ner.onto.bert.large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_large_en.html)|[onto_recognize_entities_bert_large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_large_en.html)|Pipeline|
|English|[en.ner.onto.electra.small](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_small_en.html)|[onto_recognize_entities_electra_small](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_small_en.html)|Pipeline|
|English|[en.ner.onto.electra.base](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_base_en.html)|[onto_recognize_entities_electra_base](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_base_en.html)|Pipeline|
|English|[en.ner.onto.large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_large_en.html)|[onto_recognize_entities_electra_large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_large_en.html)|Pipeline|



### New Tutorials and Notebooks

- [NYC/DC NLP Meetup Webinar video analyze Crypto News, Unsupervised Keywords, Translate between 300 Languages, Question Answering, Summerization, POS, NER in 1 line of code in almost just 20 minutes](https://www.youtube.com/watch?t=2141&v=hJR9m3NYnwk&feature=youtu.be)
- [NLU basics POS/NER/Sentiment Classification/BERTology Embeddings](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/NYC_DC_NLP_MEETUP/0_liners_intro.ipynb)
- [Explore Crypto Newsarticle dataset, unsupervised Keyword extraction, Stemming, Emotion/Sentiment distribution Analysis](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/NYC_DC_NLP_MEETUP/1_NLU_base_features_on_dataset_with_YAKE_Lemma_Stemm_classifiers_NER_.ipynb)
- [Translate between more than 300 Languages in 1 line of code with the Marian Models](https://github.com/JohnSnowLabs/nlu/blob/master/examples/webinars_conferences_etc/NYC_DC_NLP_MEETUP/2_multilingual_translation_with_marian.ipynb)
- [New NLU 1.1.2 Models Showcase Notebooks, Bengali NER, Hindi Embeddings, 30 new_models](https://colab.research.google.com/github/JohnSnowLabs/nlu/blob/master/examples/release_notebooks/NLU1.1.2_Bengali_ner_Hindi_Embeddings_30_new_models.ipynb)


### NLU 1.1.2 Bug Fixes

- Fixed a bug that caused NER confidences not beeing extracted
- Fixed a bug that caused nlu.load('spell') to crash
- Fixed a bug that caused Uralic/Estonian/ET language models not to be loaded properly


### New  Easy NLU 1-liners in 1.1.2


#### [Named Entity Recognition for Bengali (GloVe 840B 300d)](https://nlp.johnsnowlabs.com/2021/01/27/ner_jifs_glove_840B_300d_bn.html)


```python
#Bengali for :  It began to be widely used in the United States in the early '90s.
nlu.load("bn.ner").predict("৯০ এর দশকের শুরুর দিকে বৃহৎ আকারে মার্কিন যুক্তরাষ্ট্রে এর প্রয়োগের প্রক্রিয়া শুরু হয়'")
```
output :

|   entities             | token     | Entities_classes   |   ner_confidence |
|:---------------------|:----------|:----------------------|-----------------:|
| ['মার্কিন যুক্তরাষ্ট্রে'] | ৯০        | ['LOC']               |           1      |
| ['মার্কিন যুক্তরাষ্ট্রে'] | এর        | ['LOC']               |           0.9999 |
| ['মার্কিন যুক্তরাষ্ট্রে'] | দশকের     | ['LOC']               |           1      |
| ['মার্কিন যুক্তরাষ্ট্রে'] | শুরুর       | ['LOC']               |           0.9969 |
| ['মার্কিন যুক্তরাষ্ট্রে'] | দিকে      | ['LOC']               |           1      |
| ['মার্কিন যুক্তরাষ্ট্রে'] | বৃহৎ       | ['LOC']               |           0.9994 |
| ['মার্কিন যুক্তরাষ্ট্রে'] | আকারে     | ['LOC']               |           1      |
| ['মার্কিন যুক্তরাষ্ট্রে'] | মার্কিন    | ['LOC']               |           0.9602 |
| ['মার্কিন যুক্তরাষ্ট্রে'] | যুক্তরাষ্ট্রে | ['LOC']               |           0.4134 |
| ['মার্কিন যুক্তরাষ্ট্রে'] | এর        | ['LOC']               |           1      |
| ['মার্কিন যুক্তরাষ্ট্রে'] | প্রয়োগের   | ['LOC']               |           1      |
| ['মার্কিন যুক্তরাষ্ট্রে'] | প্রক্রিয়া   | ['LOC']               |           1      |
| ['মার্কিন যুক্তরাষ্ট্রে'] | শুরু        | ['LOC']               |           0.9999 |
| ['মার্কিন যুক্তরাষ্ট্রে'] | হয়        | ['LOC']               |           1      |
| ['মার্কিন যুক্তরাষ্ট্রে'] | '         | ['LOC']               |           1      |


#### [Bengali Lemmatizer](https://nlp.johnsnowlabs.com/2021/01/20/lemma_bn.html)


```python
#Bengali for :  One morning in the marble-decorated building of Vaidyanatha, an obese monk was engaged in the enchantment of Duis and the milk service of one and a half Vaidyanatha. Give me two to eat
nlu.load("bn.lemma").predict("একদিন প্রাতে বৈদ্যনাথের মার্বলমণ্ডিত দালানে একটি স্থূলোদর সন্ন্যাসী দুইসের মোহনভোগ এবং দেড়সের দুগ্ধ সেবায় নিযুক্ত আছে বৈদ্যনাথ গায়ে একখানি চাদর দিয়া জোড়করে একান্ত বিনীতভাবে ভূতলে বসিয়া ভক্তিভরে পবিত্র ভোজনব্যাপার নিরীক্ষণ করিতেছিলেন এমন সময় কোনোমতে দ্বারীদের দৃষ্টি এড়াইয়া জীর্ণদেহ বালক সহিত একটি অতি শীর্ণকায়া রমণী গৃহে প্রবেশ করিয়া ক্ষীণস্বরে কহিল বাবু দুটি খেতে দাও")

```
output :

| lemma                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | document                                                                                                                                                                                                                                                                                                                                          |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ['একদিন', 'প্রাতঃ', 'বৈদ্যনাথ', 'মার্বলমণ্ডিত', 'দালান', 'এক', 'স্থূলউদর', 'সন্ন্যাসী', 'দুইসের', 'মোহনভোগ', 'এবং', 'দেড়সের', 'দুগ্ধ', 'সেবা', 'নিযুক্ত', 'আছে', 'বৈদ্যনাথ', 'গা', 'একখান', 'চাদর', 'দেওয়া', 'জোড়কর', 'একান্ত', 'বিনীতভাব', 'ভূতল', 'বসা', 'ভক্তিভরা', 'পবিত্র', 'ভোজনব্যাপার', 'নিরীক্ষণ', 'করা', 'এমন', 'সময়', 'কোনোমত', 'দ্বারী', 'দৃষ্টি', 'এড়ানো', 'জীর্ণদেহ', 'বালক', 'সহিত', 'এক', 'অতি', 'শীর্ণকায়া', 'রমণী', 'গৃহ', 'প্রবেশ', 'বিশ্বাস', 'ক্ষীণস্বর', 'কহা', 'বাবু', 'দুই', 'খাওয়া', 'দাওয়া'] | একদিন প্রাতে বৈদ্যনাথের মার্বলমণ্ডিত দালানে একটি স্থূলোদর সন্ন্যাসী দুইসের মোহনভোগ এবং দেড়সের দুগ্ধ সেবায় নিযুক্ত আছে বৈদ্যনাথ গায়ে একখানি চাদর দিয়া জোড়করে একান্ত বিনীতভাবে ভূতলে বসিয়া ভক্তিভরে পবিত্র ভোজনব্যাপার নিরীক্ষণ করিতেছিলেন এমন সময় কোনোমতে দ্বারীদের দৃষ্টি এড়াইয়া জীর্ণদেহ বালক সহিত একটি অতি শীর্ণকায়া রমণী গৃহে প্রবেশ করিয়া ক্ষীণস্বরে কহিল বাবু দুটি খেতে দাও |


#### [Japanese Lemmatizer](https://nlp.johnsnowlabs.com/2021/01/15/lemma_ja.html)


```python
#Japanese for :  Some residents were uncomfortable with this, but it seems that no one is now openly protesting or protesting.
nlu.load("ja.lemma").predict("これに不快感を示す住民はいましたが,現在,表立って反対や抗議の声を挙げている住民はいないようです。")

```
output :

| lemma                                                                                                                                                                                                                                                          | document                                                                                         |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------|
| ['これ', 'にる', '不快', '感', 'を', '示す', '住民', 'はる', 'いる', 'まする', 'たる', 'がる', ',', '現在', ',', '表立つ', 'てる', '反対', 'やる', '抗議', 'のる', '声', 'を', '挙げる', 'てる', 'いる', '住民', 'はる', 'いる', 'なぐ', 'よう', 'です', '。'] | これに不快感を示す住民はいましたが,現在,表立って反対や抗議の声を挙げている住民はいないようです。 |

#### [Aharic Lemmatizer](https://nlp.johnsnowlabs.com/2021/01/20/lemma_am.html)


```python
#Aharic for :  Bookmark the permalink.
nlu.load("am.lemma").predict("መጽሐፉን መጽሐፍ ኡ ን አስያዛት አስያዝ ኧ ኣት ።")

```
output  :

| lemma                                                | document                         |
|:-----------------------------------------------------|:---------------------------------|
| ['_', 'መጽሐፍ', 'ኡ', 'ን', '_', 'አስያዝ', 'ኧ', 'ኣት', '።'] | መጽሐፉን መጽሐፍ ኡ ን አስያዛት አስያዝ ኧ ኣት ። |

#### [Bhojpuri Lemmatizer](https://nlp.johnsnowlabs.com/2021/01/18/lemma_bh.html)


```python
#Bhojpuri for : In this event, participation of World Bhojpuri Conference, Purvanchal Ekta Manch, Veer Kunwar Singh Foundation, Purvanchal Bhojpuri Mahasabha, and Herf - Media.
nlu.load("bh.lemma").predict("एह आयोजन में विश्व भोजपुरी सम्मेलन , पूर्वांचल एकता मंच , वीर कुँवर सिंह फाउन्डेशन , पूर्वांचल भोजपुरी महासभा , अउर हर्फ - मीडिया के सहभागिता बा ।")
```

output :

| lemma                                                                                                                                                                                                                               | document                                                                                                                      |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------|
| ['एह', 'आयोजन', 'में', 'विश्व', 'भोजपुरी', 'सम्मेलन', 'COMMA', 'पूर्वांचल', 'एकता', 'मंच', 'COMMA', 'वीर', 'कुँवर', 'सिंह', 'फाउन्डेशन', 'COMMA', 'पूर्वांचल', 'भोजपुरी', 'महासभा', 'COMMA', 'अउर', 'हर्फ', '-', 'मीडिया', 'को', 'सहभागिता', 'बा', '।'] | एह आयोजन में विश्व भोजपुरी सम्मेलन , पूर्वांचल एकता मंच , वीर कुँवर सिंह फाउन्डेशन , पूर्वांचल भोजपुरी महासभा , अउर हर्फ - मीडिया के सहभागिता बा । |

#### [Named Entity Recognition - BERT Tiny (OntoNotes)](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L2_128_en.html)
```python
nlu.load("en.ner.onto.bert.small_l2_128").predict("""William Henry Gates III (born October 28, 1955) is an American business magnate,
 software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft,
  Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect,
   while also being the largest individual shareholder until May 2014.
    He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico;
     it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect.
     During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time
      role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.
 He gradually transferred his duties to Ray Ozzie and Craig Mundie.
  He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella.""",output_level = "document")
```

output  :

| ner_confidence | entities | Entities_classes                                          |
| :------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| [0.8536999821662903, 0.7195000052452087, 0.746...] | ['PERSON', 'DATE', 'NORP', 'ORG', 'ORG', 'PERSON', 'DATE', 'CARDINAL', 'DATE', 'DATE', 'GPE', 'GPE', 'PERSON', 'DATE', 'GPE', 'GPE'] | ['William Henry Gates III', 'October 28, 1955', 'American', 'Microsoft Corporation', 'Microsoft', 'Gates', 'May 2014', 'one', '1970s', '1980s', 'Seattle', 'Washington', 'Paul Allen', '1975', 'Albuquerque', 'New Mexico'] |

####  [Named Entity Recognition - BERT Mini (OntoNotes)](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L4_256_en.html)
```python
nlu.load("en.ner.onto.bert.small_l4_256").predict("""William Henry Gates III (born October 28, 1955) is an American business magnate,
 software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft,
  Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect,
   while also being the largest individual shareholder until May 2014.
    He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico;
     it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect.
     During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time
      role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.
 He gradually transferred his duties to Ray Ozzie and Craig Mundie.
  He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella.""",output_level = "document")
```

output :

|  ner_confidence	  | entities                                                                                                                                                                                                                                           | Entities_classes                                                                                                                     |
|---------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------|
|          [0.835099995136261, 0.40450000762939453, 0.331...] | ['William Henry Gates III', 'October 28, 1955', 'American', 'Microsoft Corporation', 'Microsoft', 'Gates', 'May 2014', 'one', '1970s and 1980s', 'Seattle', 'Washington', 'Gates', 'Microsoft', 'Paul Allen', '1975', 'Albuquerque', 'New Mexico'] | ['PERSON', 'DATE', 'NORP', 'ORG', 'ORG', 'ORG', 'DATE', 'CARDINAL', 'DATE', 'GPE', 'GPE', 'ORG', 'ORG', 'PERSON', 'DATE', 'GPE', 'GPE'] |




#### [Named Entity Recognition - BERT Small (OntoNotes)](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L4_512_en.html)

```python
nlu.load("en.ner.onto.bert.small_l4_512").predict("""William Henry Gates III (born October 28, 1955) is an American business magnate,
 software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft,
  Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect,
   while also being the largest individual shareholder until May 2014.
    He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico;
     it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect.
     During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time
      role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.
 He gradually transferred his duties to Ray Ozzie and Craig Mundie.
  He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella.""",output_level = "document")
```
output :

|   ner_confidence | entities                                                                                                                                                                                                                                               | Entities_classes                                                                                                                           |
|---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------|
|              [0.964900016784668, 0.8299000263214111, 0.9607...]| ['William Henry Gates III', 'October 28, 1955', 'American', 'Microsoft Corporation', 'Microsoft', 'Gates', 'May 2014', 'one', 'the 1970s and 1980s', 'Seattle', 'Washington', 'Gates', 'Microsoft', 'Paul Allen', '1975', 'Albuquerque', 'New Mexico'] | ['PERSON', 'DATE', 'NORP', 'ORG', 'ORG', 'PERSON', 'DATE', 'CARDINAL', 'DATE', 'GPE', 'GPE', 'PERSON', 'ORG', 'PERSON', 'DATE', 'GPE', 'GPE'] |


#### [Named Entity Recognition - BERT Medium (OntoNotes)](https://nlp.johnsnowlabs.com/2020/12/05/onto_small_bert_L8_512_en.html)

```python
nlu.load("en.ner.onto.bert.small_l8_512").predict("""William Henry Gates III (born October 28, 1955) is an American business magnate,
 software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft,
  Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect,
   while also being the largest individual shareholder until May 2014.
    He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico;
     it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect.
     During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time
      role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.
 He gradually transferred his duties to Ray Ozzie and Craig Mundie.
  He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella.""",output_level = "document")
```
output :

| ner_confidence   | entities                                                                                                                                                                                                                           | Entities_classes                                                                                                        |
|---------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------|
|        [0.916700005531311, 0.5873000025749207, 0.8816...] | ['William Henry Gates III', 'October 28, 1955', 'American', 'Microsoft Corporation', 'Microsoft', 'Gates', 'May 2014', 'the 1970s and 1980s', 'Seattle', 'Washington', 'Gates', 'Paul Allen', '1975', 'Albuquerque', 'New Mexico'] | ['PERSON', 'DATE', 'NORP', 'ORG', 'ORG', 'PERSON', 'DATE', 'DATE', 'GPE', 'GPE', 'PERSON', 'PERSON', 'DATE', 'GPE', 'GPE'] |



#### [Named Entity Recognition - BERT Base (OntoNotes)](https://nlp.johnsnowlabs.com/2020/12/05/onto_bert_base_cased_en.html)

```python
nlu.load("en.ner.onto.bert.cased_base").predict("""William Henry Gates III (born October 28, 1955) is an American business magnate,
 software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft,
  Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect,
   while also being the largest individual shareholder until May 2014.
    He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico;
     it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect.
     During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time
      role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.
 He gradually transferred his duties to Ray Ozzie and Craig Mundie.
  He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella.""",output_level = "document")
```
output :

|   ner_confidence | entities                                                                                                                                                                                                                                               | Entities_classes                                                                                                                           |
|---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------|
|              [0.504800021648407, 0.47290000319480896, 0.462...] | ['William Henry Gates III', 'October 28, 1955', 'American', 'Microsoft Corporation', 'Microsoft', 'Gates', 'May 2014', 'one', 'the 1970s and 1980s', 'Seattle', 'Washington', 'Gates', 'Microsoft', 'Paul Allen', '1975', 'Albuquerque', 'New Mexico'] | ['PERSON', 'DATE', 'NORP', 'ORG', 'ORG', 'PERSON', 'DATE', 'CARDINAL', 'DATE', 'GPE', 'GPE', 'PERSON', 'ORG', 'PERSON', 'DATE', 'GPE', 'GPE'] |



#### [Named Entity Recognition - BERT Large (OntoNotes)](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_small_uncased_en.html)
```python
nlu.load("en.ner.onto.electra.uncased_small").predict("""William Henry Gates III (born October 28, 1955) is an American business magnate,
 software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft,
  Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect,
   while also being the largest individual shareholder until May 2014.
    He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico;
     it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect.
     During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time
      role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.
 He gradually transferred his duties to Ray Ozzie and Craig Mundie.
  He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella.""",output_level = "document")
```
output :

|   ner_confidence | entities                                                                                                                                                                                                                                          | Entities_classes                                                                                                                                   |
|---------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
|            [0.7213000059127808, 0.6384000182151794, 0.731...]  | ['William Henry Gates III', 'October 28, 1955', 'American', 'Microsoft Corporation', 'Microsoft', 'Gates', 'May 2014', 'one', '1970s', '1980s', 'Seattle', 'Washington', 'Gates', 'Microsoft', 'Paul Allen', '1975', 'Albuquerque', 'New Mexico'] | ['PERSON', 'DATE', 'NORP', 'ORG', 'ORG', 'PERSON', 'DATE', 'CARDINAL', 'DATE', 'DATE', 'GPE', 'GPE', 'PERSON', 'ORG', 'PERSON', 'DATE', 'GPE', 'GPE'] |

#### [Named Entity Recognition - ELECTRA Small (OntoNotes)](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_small_uncased_en.html)

```python
nlu.load("en.ner.onto.electra.uncased_small").predict("""William Henry Gates III (born October 28, 1955) is an American business magnate,
 software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft,
  Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect,
   while also being the largest individual shareholder until May 2014.
    He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico;
     it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect.
     During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time
      role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.
 He gradually transferred his duties to Ray Ozzie and Craig Mundie.
  He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella.""",output_level = "document")
```

output :

|   ner_confidence | Entities_classes                                                                                                                                   | entities                                                                                                                                                                                                                                          |
|---------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|            [0.8496000170707703, 0.4465999901294708, 0.568...]  | ['PERSON', 'DATE', 'NORP', 'ORG', 'ORG', 'PERSON', 'DATE', 'CARDINAL', 'DATE', 'DATE', 'GPE', 'GPE', 'PERSON', 'ORG', 'PERSON', 'DATE', 'GPE', 'GPE'] | ['William Henry Gates III', 'October 28, 1955', 'American', 'Microsoft Corporation', 'Microsoft', 'Gates', 'May 2014', 'one', '1970s', '1980s', 'Seattle', 'Washington', 'Gates', 'Microsoft', 'Paul Allen', '1975', 'Albuquerque', 'New Mexico'] |



#### [Named Entity Recognition - ELECTRA Base (OntoNotes)](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_base_uncased_en.html)

```python
nlu.load("en.ner.onto.electra.uncased_base").predict("""William Henry Gates III (born October 28, 1955) is an American business magnate,
 software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft,
  Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect,
   while also being the largest individual shareholder until May 2014.
    He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico;
     it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect.
     During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time
      role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.
 He gradually transferred his duties to Ray Ozzie and Craig Mundie.
  He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadellabase.""",output_level = "document")

```

output :

|   ner_confidence | entities                                                                                                                                                                                                                                              | Entities_classes                                                                                                                                   |
|---------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------|
|              [0.5134000182151794, 0.9419000148773193, 0.802...]| ['William Henry Gates III', 'October 28, 1955', 'American', 'Microsoft Corporation', 'Microsoft', 'Gates', 'May 2014', 'one', 'the 1970s', '1980s', 'Seattle', 'Washington', 'Gates', 'Microsoft', 'Paul Allen', '1975', 'Albuquerque', 'New Mexico'] | ['PERSON', 'DATE', 'NORP', 'ORG', 'ORG', 'PERSON', 'DATE', 'CARDINAL', 'DATE', 'DATE', 'GPE', 'GPE', 'PERSON', 'ORG', 'PERSON', 'DATE', 'GPE', 'GPE'] |


#### [Named Entity Recognition - ELECTRA Large (OntoNotes)](https://nlp.johnsnowlabs.com/2020/12/05/onto_electra_large_uncased_en.html)

```python

nlu.load("en.ner.onto.electra.uncased_large").predict("""William Henry Gates III (born October 28, 1955) is an American business magnate,
 software developer, investor, and philanthropist. He is best known as the co-founder of Microsoft Corporation. During his career at Microsoft,
  Gates held the positions of chairman, chief executive officer (CEO), president and chief software architect,
   while also being the largest individual shareholder until May 2014.
    He is one of the best-known entrepreneurs and pioneers of the microcomputer revolution of the 1970s and 1980s. Born and raised in Seattle, Washington, Gates co-founded Microsoft with childhood friend Paul Allen in 1975, in Albuquerque, New Mexico;
     it went on to become the world's largest personal computer software company. Gates led the company as chairman and CEO until stepping down as CEO in January 2000, but he remained chairman and became chief software architect.
     During the late 1990s, Gates had been criticized for his business tactics, which have been considered anti-competitive. This opinion has been upheld by numerous court rulings. In June 2006, Gates announced that he would be transitioning to a part-time
      role at Microsoft and full-time work at the Bill & Melinda Gates Foundation, the private charitable foundation that he and his wife, Melinda Gates, established in 2000.
 He gradually transferred his duties to Ray Ozzie and Craig Mundie.
  He stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadellabase.""",output_level = "document")
```

output :

|   ner_confidence | entities                                                                                                                                                                                                                                                            | Entities_classes                                                                                                                                          |
|---------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|              [0.8442000150680542, 0.26840001344680786, 0.57...] | ['William Henry Gates', 'October 28, 1955', 'American', 'Microsoft Corporation', 'Microsoft', 'Gates', 'May 2014', 'one', '1970s', '1980s', 'Seattle', 'Washington', 'Gates co-founded', 'Microsoft', 'Paul Allen', '1975', 'Albuquerque', 'New Mexico', 'largest'] | ['PERSON', 'DATE', 'NORP', 'ORG', 'ORG', 'PERSON', 'DATE', 'CARDINAL', 'DATE', 'DATE', 'GPE', 'GPE', 'PERSON', 'ORG', 'PERSON', 'DATE', 'GPE', 'GPE', 'GPE'] |


#### [Recognize Entities OntoNotes - BERT Tiny](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_tiny_en.html)

```python

nlu.load("en.ner.onto.bert.tiny").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.",output_level="document")
```

output :

|   ner_confidence | entities                                                                            | Entities_classes                                         |
|---------------:|:------------------------------------------------------------------------------------|:------------------------------------------------------------|
|              [0.994700014591217, 0.9412999749183655, 0.9685...] | ['Johnson', 'first', '2001', 'Parliament', 'eight years', 'London', '2008 to 2016'] | ['PERSON', 'ORDINAL', 'DATE', 'ORG', 'DATE', 'GPE', 'DATE'] |

#### [Recognize Entities OntoNotes - BERT Mini](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_mini_en.html)

```python
nlu.load("en.ner.onto.bert.mini").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.",output_level="document")
```
output :

|   ner_confidence | entities                                                              | Entities_classes                                  |
|---------------:|:----------------------------------------------------------------------|:-----------------------------------------------------|
|              [0.996399998664856, 0.9733999967575073, 0.8766...]| ['Johnson', 'first', '2001', 'eight years', 'London', '2008 to 2016'] | ['PERSON', 'ORDINAL', 'DATE', 'DATE', 'GPE', 'DATE'] |

#### [Recognize Entities OntoNotes - BERT Small](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_small_en.html)


```python
nlu.load("en.ner.onto.bert.small").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.",output_level="document")
```
output :

|   ner_confidence | entities                                                                            | Entities_classes                                         |
|---------------:|:------------------------------------------------------------------------------------|:------------------------------------------------------------|
|              [0.9987999796867371, 0.9610000252723694, 0.998...]| ['Johnson', 'first', '2001', 'eight years', 'London', '2008 to 2016', 'Parliament'] | ['PERSON', 'ORDINAL', 'DATE', 'DATE', 'GPE', 'DATE', 'ORG'] |

#### [Recognize Entities OntoNotes - BERT Medium](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_medium_en.html)

```python

nlu.load("en.ner.onto.bert.medium").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.",output_level="document")
```
output :

|   ner_confidence | entities                                                              | Entities_classes                                  |
|---------------:|:----------------------------------------------------------------------|:-----------------------------------------------------|
|              [0.9969000220298767, 0.8575999736785889, 0.995...] | ['Johnson', 'first', '2001', 'eight years', 'London', '2008 to 2016'] | ['PERSON', 'ORDINAL', 'DATE', 'DATE', 'GPE', 'DATE'] |

#### [Recognize Entities OntoNotes - BERT Base](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_base_en.html)

```python
nlu.load("en.ner.onto.bert.base").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.",output_level="document")
```


output :

|   ner_confidence | entities                                                                                          | Entities_classes                                                |
|---------------:|:--------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
|              [0.996999979019165, 0.933899998664856, 0.99930...] | ['Johnson', 'first', '2001', 'Parliament', 'eight years', 'London', '2008 to 2016', 'Parliament'] | ['PERSON', 'ORDINAL', 'DATE', 'ORG', 'DATE', 'GPE', 'DATE', 'ORG'] |

#### [Recognize Entities OntoNotes - BERT Large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_bert_large_en.html)


```python
nlu.load("en.ner.onto.bert.large").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.",output_level="document")
```

output :

|   ner_confidence | entities                                                                                          | Entities_classes                                                |
|---------------:|:--------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
|              [0.9786999821662903, 0.9549000263214111, 0.998...] | ['Johnson', 'first', '2001', 'Parliament', 'eight years', 'London', '2008 to 2016', 'Parliament'] | ['PERSON', 'ORDINAL', 'DATE', 'ORG', 'DATE', 'GPE', 'DATE', 'ORG'] |

#### [Recognize Entities OntoNotes - ELECTRA Small](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_small_en.html)

```pythone
nlu.load("en.ner.onto.electra.small").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.",output_level="document")
```
output :

|   ner_confidence | entities                                                              | Entities_classes                                  |
|---------------:|:----------------------------------------------------------------------|:-----------------------------------------------------|
|              [0.9952999949455261, 0.8589000105857849, 0.996...] | ['Johnson', 'first', '2001', 'eight years', 'London', '2008 to 2016'] | ['PERSON', 'ORDINAL', 'DATE', 'DATE', 'GPE', 'DATE'] |

#### [Recognize Entities OntoNotes - ELECTRA Base](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_base_en.html)
```python
nlu.load("en.ner.onto.electra.base").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.",output_level="document")
```
output :

|   ner_confidence | entities                                                                            | Entities_classes                                                 |
|---------------:|:------------------------------------------------------------------------------------|:--------------------------------------------------------------------|
|              [0.9987999796867371, 0.9474999904632568, 0.999...] | ['Johnson', 'first', '2001', 'Parliament', 'eight years', 'London', '2008', '2016'] | ['PERSON', 'ORDINAL', 'DATE', 'ORG', 'DATE', 'GPE', 'DATE', 'DATE'] |

#### [Recognize Entities OntoNotes - ELECTRA Large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_large_en.html)

```python
nlu.load("en.ner.onto.large").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London, from 2008 to 2016, before rejoining Parliament.",output_level="document")
```
output :

|   ner_confidence | entities                                                              | Entities_classes                                  |
|---------------:|:----------------------------------------------------------------------|:-----------------------------------------------------|
|              [0.9998000264167786, 0.9613999724388123, 0.998...] | ['Johnson', 'first', '2001', 'eight years', 'London', '2008 to 2016'] | ['PERSON', 'ORDINAL', 'DATE', 'DATE', 'GPE', 'DATE'] |

### NLU Installation

```bash
# PyPi
!pip install nlu pyspark==2.4.7
#Conda
# Install NLU from Anaconda/Conda
conda install -c johnsnowlabs nlu
```

### Additional NLU ressources
- [NLU Website](https://nlu.johnsnowlabs.com/)
- [All NLU Tutorial Notebooks](https://nlu.johnsnowlabs.com/docs/en/notebooks)
- [NLU Videos and Blogposts on NLU](https://nlp.johnsnowlabs.com/learn#pythons-nlu-library)
- [NLU on Github](https://github.com/JohnSnowLabs/nlu)


## NLU 1.1.1 Release Notes

We are very excited to release NLU 1.1.1!
This release features 3 new tutorial notebooks for Open/Closed book question answering with Google's T5, Intent classification and Aspect Based NER.
In Addition NLU 1.1.0 comes with  25+ pretrained models and pipelines in Amharic, Bengali, Bhojpuri, Japanese, and Korean languages from the [amazing Spark2.7.2 release](https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.7.2)
Finally NLU now supports running on Spark 2.3 clusters.


### NLU 1.1.0 New Non-English Models

|Language | nlu.load() reference | Spark NLP Model reference | Type |
|---------|---------------------|----------------------------|------|
|Arabic | [ar.ner](https://nlp.johnsnowlabs.com/2020/12/05/aner_cc_300d_ar.html) |[arabic_w2v_cc_300d](https://nlp.johnsnowlabs.com/2020/12/05/aner_cc_300d_ar.html) | Named Entity Recognizer                    |
|Arabic | [ar.embed.aner](https://nlp.johnsnowlabs.com/2020/12/05/aner_cc_300d_ar.html) |[aner_cc_300d](https://nlp.johnsnowlabs.com/2020/12/05/aner_cc_300d_ar.html) | Word Embedding                    |
|Arabic | [ar.embed.aner.300d](https://nlp.johnsnowlabs.com/2020/12/05/aner_cc_300d_ar.html) |[aner_cc_300d](https://nlp.johnsnowlabs.com/2020/12/05/aner_cc_300d_ar.html) | Word Embedding (Alias)                    |
|Bengali | [bn.stopwords](https://nlp.johnsnowlabs.com/2020/07/14/stopwords_bn.html) |[stopwords_bn](https://nlp.johnsnowlabs.com/2020/07/14/stopwords_bn.html) | Stopwords Cleaner                    |
|Bengali | [bn.pos](https://nlp.johnsnowlabs.com/2021/01/20/pos_msri_bn.html) |[pos_msri](https://nlp.johnsnowlabs.com/2021/01/20/pos_msri_bn.html) | Part of Speech                    |
|Thai | [th.segment_words](https://nlp.johnsnowlabs.com/2021/01/11/ner_lst20_glove_840B_300d_th.html) |[wordseg_best](https://nlp.johnsnowlabs.com/2021/01/11/ner_lst20_glove_840B_300d_th.html) | Word Segmenter                    |
|Thai | [th.pos](https://nlp.johnsnowlabs.com/2021/01/13/pos_lst20_th.html) |[pos_lst20](https://nlp.johnsnowlabs.com/2021/01/13/pos_lst20_th.html) | Part of Speech                    |
|Thai |   [th.sentiment](https://nlp.johnsnowlabs.com/2021/01/14/sentiment_jager_use_th.html) |[sentiment_jager_use](https://nlp.johnsnowlabs.com/2021/01/14/sentiment_jager_use_th.html) | Sentiment Classifier                     |
|Thai |    [th.classify.sentiment](https://nlp.johnsnowlabs.com/2021/01/14/sentiment_jager_use_th.html) |[sentiment_jager_use](https://nlp.johnsnowlabs.com/2021/01/14/sentiment_jager_use_th.html) | Sentiment Classifier (Alias)                    |
|Chinese | [zh.pos.ud_gsd_trad](https://nlp.johnsnowlabs.com/2021/01/25/pos_ud_gsd_trad_zh.html) |[pos_ud_gsd_trad](https://nlp.johnsnowlabs.com/2021/01/25/pos_ud_gsd_trad_zh.html) | Part of Speech                    |
|Chinese | [zh.segment_words.gsd](https://nlp.johnsnowlabs.com/2021/01/25/wordseg_gsd_ud_trad_zh.html) |[wordseg_gsd_ud_trad](https://nlp.johnsnowlabs.com/2021/01/25/wordseg_gsd_ud_trad_zh.html) | Word Segmenter                    |
|Bihari | [bh.pos](https://nlp.johnsnowlabs.com/2021/01/18/pos_ud_bhtb_bh.html) |[pos_ud_bhtb](https://nlp.johnsnowlabs.com/2021/01/18/pos_ud_bhtb_bh.html) | Part of Speech                    |
|Amharic | [am.pos](https://nlp.johnsnowlabs.com/2021/01/20/pos_ud_att_am.html) |[pos_ud_att](https://nlp.johnsnowlabs.com/2021/01/20/pos_ud_att_am.html) | Part of Speech                    |



### NLU 1.1.1 New English Models and Pipelines

|Language | nlu.load() reference | Spark NLP Model reference | Type |
|---------|---------------------|----------------------------|------|
| English | [en.sentiment.glove](https://nlp.johnsnowlabs.com/2021/01/15/analyze_sentimentdl_glove_imdb_en.html) |[analyze_sentimentdl_glove_imdb](https://nlp.johnsnowlabs.com/2021/01/15/analyze_sentimentdl_glove_imdb_en.html)     | Sentiment Classifier |
| English | [en.sentiment.glove.imdb](https://nlp.johnsnowlabs.com/2021/01/15/analyze_sentimentdl_glove_imdb_en.html) |[analyze_sentimentdl_glove_imdb](https://nlp.johnsnowlabs.com/2021/01/15/analyze_sentimentdl_glove_imdb_en.html)     | Sentiment Classifier (Alias) |
| English | [en.classify.sentiment.glove.imdb](https://nlp.johnsnowlabs.com/2021/01/15/analyze_sentimentdl_glove_imdb_en.html) |[analyze_sentimentdl_glove_imdb](https://nlp.johnsnowlabs.com/2021/01/15/analyze_sentimentdl_glove_imdb_en.html)     | Sentiment Classifier (Alias) |
| English | [en.classify.sentiment.glove](https://nlp.johnsnowlabs.com/2021/01/15/analyze_sentimentdl_glove_imdb_en.html) |[analyze_sentimentdl_glove_imdb](https://nlp.johnsnowlabs.com/2021/01/15/analyze_sentimentdl_glove_imdb_en.html)     | Sentiment Classifier (Alias) |
| English | [en.classify.trec50.pipe](https://nlp.johnsnowlabs.com/2021/01/08/classifierdl_use_trec50_pipeline_en.html) |[classifierdl_use_trec50_pipeline](https://nlp.johnsnowlabs.com/2021/01/08/classifierdl_use_trec50_pipeline_en.html)     | Language Classifier |
| English | [en.ner.onto.large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_large_en.html) |[onto_recognize_entities_electra_large](https://nlp.johnsnowlabs.com/2020/12/09/onto_recognize_entities_electra_large_en.html)     | Named Entity Recognizer |
| English | [en.classify.questions.atis](https://nlp.johnsnowlabs.com/2021/01/25/classifierdl_use_atis_en.html) |[classifierdl_use_atis](https://nlp.johnsnowlabs.com/2021/01/25/classifierdl_use_atis_en.html)     | Intent Classifier |
| English  | [en.classify.questions.airline](https://nlp.johnsnowlabs.com/2021/01/25/classifierdl_use_atis_en.html) |[classifierdl_use_atis](https://nlp.johnsnowlabs.com/2021/01/25/classifierdl_use_atis_en.html)     | Intent Classifier (Alias) |
| English | [en.classify.intent.atis](https://nlp.johnsnowlabs.com/2021/01/25/classifierdl_use_atis_en.html) |[classifierdl_use_atis](https://nlp.johnsnowlabs.com/2021/01/25/classifierdl_use_atis_en.html)     | Intent Classifier (Alias) |
| English | [en.classify.intent.airline](https://nlp.johnsnowlabs.com/2021/01/25/classifierdl_use_atis_en.html) |[classifierdl_use_atis](https://nlp.johnsnowlabs.com/2021/01/25/classifierdl_use_atis_en.html)     | Intent Classifier (Alias) |
| English | [en.ner.atis](https://nlp.johnsnowlabs.com/2021/01/25/nerdl_atis_840b_300d_en.html) |[nerdl_atis_840b_300d](https://nlp.johnsnowlabs.com/2021/01/25/nerdl_atis_840b_300d_en.html)     | Aspect based NER |
| English | [en.ner.airline](https://nlp.johnsnowlabs.com/2021/01/25/nerdl_atis_840b_300d_en.html) |[nerdl_atis_840b_300d](https://nlp.johnsnowlabs.com/2021/01/25/nerdl_atis_840b_300d_en.html)     | Aspect based NER (Alias) |
| English | [en.ner.aspect.airline](https://nlp.johnsnowlabs.com/2021/01/25/nerdl_atis_840b_300d_en.html) |[nerdl_atis_840b_300d](https://nlp.johnsnowlabs.com/2021/01/25/nerdl_atis_840b_300d_en.html)     | Aspect based NER (Alias) |
| English | [en.ner.aspect.atis](https://nlp.johnsnowlabs.com/2021/01/25/nerdl_atis_840b_300d_en.html) |[nerdl_atis_840b_300d](https://nlp.johnsnowlabs.com/2021/01/25/nerdl_atis_840b_300d_en.html)     | Aspect based NER (Alias) |

### New Easy NLU 1-liner Examples : 

#### Extract aspects and entities from airline questions (ATIS dataset)

```python
	
nlu.load("en.ner.atis").predict("i want to fly from baltimore to dallas round trip")
output:  ["baltimore"," dallas", "round trip"]
```



#### Intent Classification for Airline Traffic Information System queries (ATIS dataset)

```python

nlu.load("en.classify.questions.atis").predict("what is the price of flight from newyork to washington")
output:  "atis_airfare"	
```



#### Recognize Entities OntoNotes - ELECTRA Large

```python

nlu.load("en.ner.onto.large").predict("Johnson first entered politics when elected in 2001 as a member of Parliament. He then served eight years as the mayor of London.")	
output:  ["Johnson", "first", "2001", "eight years", "London"]	
```

#### Question classification of open-domain and fact-based questions Pipeline - TREC50

```python
nlu.load("en.classify.trec50.pipe").predict("When did the construction of stone circles begin in the UK? ")
output:  LOC_other
```

#### Traditional Chinese Word Segmentation

```python
# 'However, this treatment also creates some problems' in Chinese
nlu.load("zh.segment_words.gsd").predict("然而，這樣的處理也衍生了一些問題。")
output:  ["然而",",","這樣","的","處理","也","衍生","了","一些","問題","。"]

```


#### Part of Speech for Traditional Chinese

```python
# 'However, this treatment also creates some problems' in Chinese
nlu.load("zh.pos.ud_gsd_trad").predict("然而，這樣的處理也衍生了一些問題。")
```

Output:

|Token |  POS   |
| ----- | ----- |
| 然而  | ADV   |
| ，    | PUNCT |
| 這樣  | PRON  |
| 的    | PART  |
| 處理  | NOUN  |
| 也    | ADV   |
| 衍生  | VERB  |
| 了    | PART  |
| 一些  | ADJ   |
| 問題  | NOUN  |
| 。    | PUNCT |

#### Thai Word Segment Recognition


```python
# 'Mona Lisa is a 16th-century oil painting created by Leonardo held at the Louvre in Paris' in Thai
nlu.loadnlu.load("th.segment_words").predict("Mona Lisa เป็นภาพวาดสีน้ำมันในศตวรรษที่ 16 ที่สร้างโดย Leonardo จัดขึ้นที่พิพิธภัณฑ์ลูฟร์ในปารีส")

```

Output:

| token |
| --------- |
| M         |
| o         |
| n         |
| a         |
| Lisa      |
| เป็น       |
| ภาพ       |
| ว         |
| า         |
| ด         |
| สีน้ำ       |
| มัน        |
| ใน        |
| ศตวรรษ    |
| ที่         |
| 16        |
| ที่         |
| สร้าง      |
| โ         |
| ด         |
| ย         |
| L         |
| e         |
| o         |
| n         |
| a         |
| r         |
| d         |
| o         |
| จัด        |
| ขึ้น        |
| ที่         |
| พิพิธภัณฑ์    |
| ลูฟร์       |
| ใน        |
| ปารีส      |

#### Part of Speech for Bengali (POS)

```python
# 'The village is also called 'Mod' in Tora language' in Behgali 
nlu.load("bn.pos").predict("বাসস্থান-ঘরগৃহস্থালি তোড়া ভাষায় গ্রামকেও বলে ` মোদ ' ৷")
```

Output:

| token             | pos  |
| ----------------- | ---- |
| বাসস্থান-ঘরগৃহস্থালি | NN   |
| তোড়া              | NNP  |
| ভাষায়             | NN   |
| গ্রামকেও           | NN   |
| বলে               | VM   |
| `                 | SYM  |
| মোদ               | NN   |
| '                 | SYM  |
| ৷                 | SYM  |



#### Stop Words Cleaner for Bengali


```python
# 'This language is not enough' in Bengali 
df = nlu.load("bn.stopwords").predict("এই ভাষা যথেষ্ট নয়")

```

Output:

| cleanTokens | token |
| :---------- | :---- |
| ভাষা        | এই    |
| যথেষ্ট       | ভাষা  |
| নয়          | যথেষ্ট |
| None        | নয়    |


#### Part of Speech for Bengali
```python

# 'The people of Ohu know that the foundation of Bhojpuri was shaken' in Bengali
nlu.load('bh.pos').predict("ओहु लोग के मालूम बा कि श्लील होखते भोजपुरी के नींव हिल जाई")
```

Output:

| pos   | token   |
| :---- | :------ |
| DET   | ओहु     |
| NOUN  | लोग     |
| ADP   | के      |
| NOUN  | मालूम   |
| VERB  | बा      |
| SCONJ | कि      |
| ADJ   | श्लील   |
| VERB  | होखते   |
| PROPN | भोजपुरी |
| ADP   | के      |
| NOUN  | नींव    |
| VERB  | हिल     |
| AUX   | जाई     |


#### Amharic Part of Speech (POS)
```python
# ' "Son, finish the job," he said.' in Amharic
nlu.load('am.pos').predict('ልጅ ኡ ን ሥራ ው ን አስጨርስ ኧው ኣል ኧሁ ።"')
```

Output:

| pos   | token   |
|:------|:--------|
| NOUN  | ልጅ      |
| DET   | ኡ       |
| PART  | ን       |
| NOUN  | ሥራ      |
| DET   | ው       |
| PART  | ን       |
| VERB  | አስጨርስ   |
| PRON  | ኧው      |
| AUX   | ኣል      |
| PRON  | ኧሁ      |
| PUNCT | ።       |
| NOUN  | "       |


#### Thai Sentiment Classification
```python
#  'I love peanut butter and jelly!' in thai
nlu.load('th.classify.sentiment').predict('ฉันชอบเนยถั่วและเยลลี่!')[['sentiment','sentiment_confidence']]
```

Output:

| sentiment   |   sentiment_confidence |
|:------------|-----------------------:|
| positive    |               0.999998 |


#### Arabic Named Entity Recognition (NER)
```python
# 'In 1918, the forces of the Arab Revolt liberated Damascus with the help of the British' in Arabic
nlu.load('ar.ner').predict('في عام 1918 حررت قوات الثورة العربية دمشق بمساعدة من الإنكليز',output_level='chunk')[['entities_confidence','ner_confidence','entities']]
```

Output:

| entity_class   | ner_confidence                                                                                                                                                                  | entities            |
|:----------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|
| ORG                   | [1.0, 1.0, 1.0, 0.9997000098228455, 0.9840999841690063, 0.9987999796867371, 0.9990000128746033, 0.9998999834060669, 0.9998999834060669, 0.9993000030517578, 0.9998999834060669] | قوات الثورة العربية |
| LOC                   | [1.0, 1.0, 1.0, 0.9997000098228455, 0.9840999841690063, 0.9987999796867371, 0.9990000128746033, 0.9998999834060669, 0.9998999834060669, 0.9993000030517578, 0.9998999834060669] | دمشق                |
| PER                   | [1.0, 1.0, 1.0, 0.9997000098228455, 0.9840999841690063, 0.9987999796867371, 0.9990000128746033, 0.9998999834060669, 0.9998999834060669, 0.9993000030517578, 0.9998999834060669] | الإنكليز            |



### NLU 1.1.0 Enhancements : 
-  Spark 2.3 compatibility

### New NLU Notebooks and Tutorials 
- [Open and Closed book question Ansering](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_question_answering.ipynb)
- [Aspect based NER for Airline ATIS](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/classifiers/intent_classification_airlines_ATIS.ipynb)
- [Intent Classification for Airline emssages ATIS](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/named_entity_recognition_(NER)/NER_aspect_airline_ATIS.ipynb)

### Installation

```bash
# PyPi
!pip install nlu pyspark==2.4.7
#Conda
# Install NLU from Anaconda/Conda
conda install -c johnsnowlabs nlu
```

### Additional NLU ressources
- [NLU Website](https://nlu.johnsnowlabs.com/)
- [All NLU Tutorial Notebooks](https://nlu.johnsnowlabs.com/docs/en/notebooks)
- [NLU Videos and Blogposts on NLU](https://nlp.johnsnowlabs.com/learn#pythons-nlu-library)
- [NLU on Github](https://github.com/JohnSnowLabs/nlu)




##  NLU 1.1.0 Release Notes 
We are incredibly excited to release NLU 1.1.0!
This release it integrates the 720+ new models from the latest [Spark-NLP 2.7.0 + releases](https://github.com/JohnSnowLabs/spark-nlp/releases)
You can now achieve state-of-the-art results with Sequence2Sequence transformers like for problems text summarization, question answering, translation between  192+ languages and extract Named Entity in various Right to Left written languages like Koreas, Japanese, Chinese and many more in 1 line of code!     
These new features are possible because of the integration of the [Google's T5 models](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) and [Microsoft's Marian models](https://marian-nmt.github.io/publications/)  transformers

NLU 1.1.0 has over 720+ new pretrained models and pipelines while extending the support of multi-lingual models to 192+ languages such as Chinese, Japanese, Korean, Arabic, Persian, Urdu, and Hebrew.     



### NLU 1.1.0  New Features
* **720+** new models you can find an overview of all NLU models [here](https://nlu.johnsnowlabs.com/docs/en/spellbook) and further documentation in the [models hub](https://nlp.johnsnowlabs.com/models)
* **NEW:** Introducing MarianTransformer annotator for machine translation based on MarianNMT models. Marian is an efficient, free Neural Machine Translation framework mainly being developed by the Microsoft Translator team (646+ pretrained models & pipelines in 192+ languages)
* **NEW:** Introducing T5Transformer annotator for Text-To-Text Transfer Transformer (Google T5) models to achieve state-of-the-art results on multiple NLP tasks such as Translation, Summarization, Question Answering, Sentence Similarity, and so on
* **NEW:** Introducing brand new and refactored language detection and identification models. The new LanguageDetectorDL is faster, more accurate, and supports up to 375 languages
* **NEW:** Introducing WordSegmenter model for word segmentation of languages without any rule-based tokenization such as Chinese, Japanese, or Korean
* **NEW:** Introducing DocumentNormalizer component for cleaning content from HTML or XML documents, applying either data cleansing using an arbitrary number of custom regular expressions either data extraction following the different parameters


### NLU 1.1.0  New Notebooks, Tutorials and Articles
- [Translate between 192+ languages with marian](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/translation_demo.ipynb)
- [Try out the 18 Tasks like Summarization Question Answering and more on T5](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more)
- [Tokenize, extract POS and NER in Chinese](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/multilingual/chinese_ner_pos_and_tokenization.ipynb)
- [Tokenize, extract POS and NER in Korean](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/multilingual/korean_ner_pos_and_tokenization.ipynb)
- [Tokenize, extract POS and NER in Japanese](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/multilingual/japanese_ner_pos_and_tokenization.ipynb)
- [Normalize documents](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/text_pre_processing_and_cleaning/document_normalizer_demo.ipynb)
- [Aspect based sentiment NER sentiment for restaurants](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/named_entity_recognition_(NER)/aspect_based_ner_sentiment_restaurants.ipynb)

### NLU 1.1.0 New Training Tutorials
#### Binary Classifier training Jupyter tutorials
- [2 class Finance News sentiment classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_apple_twitter.ipynb)
- [2 class Reddit comment sentiment classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_reddit.ipynb)
- [2 class Apple Tweets sentiment classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_IMDB.ipynb)
- [2 class IMDB Movie sentiment classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_IMDB.ipynb)
- [2 class twitter classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/binary_text_classification/NLU_training_sentiment_classifier_demo_twitter.ipynb)

#### Multi Class text Classifier training Jupyter tutorials
- [5 class WineEnthusiast Wine review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_wine.ipynb) 
- [3 class Amazon Phone review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_amazon.ipynb)
- [5 class Amazon Musical Instruments review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_musical_instruments.ipynb)
- [5 class Tripadvisor Hotel review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_hotel_reviews.ipynb)
- [5 class Phone review classifier training](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/Training/multi_class_text_classification/NLU_training_multi_class_text_classifier_demo_hotel_reviews.ipynb)


### NLU 1.1.0 New Medium Tutorials

- [1 line to Glove Word Embeddings with NLU     with t-SNE plots](https://medium.com/spark-nlp/1-line-to-glove-word-embeddings-with-nlu-in-python-baed152fff4d)     
- [1 line to Xlnet Word Embeddings with NLU     with t-SNE plots](https://medium.com/spark-nlp/1-line-to-xlnet-word-embeddings-with-nlu-in-python-5efc57d7ac79)     
- [1 line to AlBERT Word Embeddings with NLU    with t-SNE plots](https://medium.com/spark-nlp/1-line-to-albert-word-embeddings-with-nlu-in-python-1691bc048ed1)     
- [1 line to CovidBERT Word Embeddings with NLU with t-SNE plots](https://medium.com/spark-nlp/1-line-to-covidbert-word-embeddings-with-nlu-in-python-e67396da2f78)     
- [1 line to Electra Word Embeddings with NLU   with t-SNE plots](https://medium.com/spark-nlp/1-line-to-electra-word-embeddings-with-nlu-in-python-25f749bf3e92)     
- [1 line to BioBERT Word Embeddings with NLU   with t-SNE plots](https://medium.com/spark-nlp/1-line-to-biobert-word-embeddings-with-nlu-in-python-7224ab52e131)     




## Translation
[Translation example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/translation_demo.ipynb)       
You can translate between more than 192 Languages pairs with the [Marian Models](https://marian-nmt.github.io/publications/)
You need to specify the language your data is in as `start_language` and the language you want to translate to as `target_language`.    
The language references must be [ISO language codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)

`nlu.load('<start_language>.translate.<target_language>')`

**Translate Turkish to English:**     
`nlu.load('tr.translate_to.fr')`

**Translate English to French:**     
`nlu.load('en.translate_to.fr')`


**Translate French to Hebrew:**     
`nlu.load('en.translate_to.fr')`

```python
translate_pipe = nlu.load('en.translate_to.fr')
df = translate_pipe.predict('Billy likes to go to the mall every sunday')
df
```

|	sentence|	translation|
|-----------|--------------|
|Billy likes to go to the mall every sunday	| Billy geht gerne jeden Sonntag ins Einkaufszentrum|






## T5
[Example of every T5 task](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more)
### Overview of every task available with T5
[The T5 model](https://arxiv.org/pdf/1910.10683.pdf) is trained on various datasets for 17 different tasks which fall into 8 categories.


1. Text summarization
2. Question answering
3. Translation
4. Sentiment analysis
5. Natural Language inference
6. Coreference resolution
7. Sentence Completion
8. Word sense disambiguation

### Every T5 Task with explanation:

|Task Name | Explanation | 
|----------|--------------|
|[1.CoLA](https://nyu-mll.github.io/CoLA/)                   | Classify if a sentence is gramaticaly correct|
|[2.RTE](https://dl.acm.org/doi/10.1007/11736790_9)                    | Classify whether if a statement can be deducted from a sentence|
|[3.MNLI](https://arxiv.org/abs/1704.05426)                   | Classify for a hypothesis and premise whether they contradict or contradict each other or neither of both (3 class).|
|[4.MRPC](https://www.aclweb.org/anthology/I05-5002.pdf)                   | Classify whether a pair of sentences is a re-phrasing of each other (semantically equivalent)|
|[5.QNLI](https://arxiv.org/pdf/1804.07461.pdf)                   | Classify whether the answer to a question can be deducted from an answer candidate.|
|[6.QQP](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)                    | Classify whether a pair of questions is a re-phrasing of each other (semantically equivalent)|
|[7.SST2](https://www.aclweb.org/anthology/D13-1170.pdf)                   | Classify the sentiment of a sentence as positive or negative|
|[8.STSB](https://www.aclweb.org/anthology/S17-2001/)                   | Classify the sentiment of a sentence on a scale from 1 to 5 (21 Sentiment classes)|
|[9.CB](https://ojs.ub.uni-konstanz.de/sub/index.php/sub/article/view/601)                     | Classify for a premise and a hypothesis whether they contradict each other or not (binary).|
|[10.COPA](https://www.aaai.org/ocs/index.php/SSS/SSS11/paper/view/2418/0)                   | Classify for a question, premise, and 2 choices which choice the correct choice is (binary).|
|[11.MultiRc](https://www.aclweb.org/anthology/N18-1023.pdf)                | Classify for a question, a paragraph of text, and an answer candidate, if the answer is correct (binary),|
|[12.WiC](https://arxiv.org/abs/1808.09121)                    | Classify for a pair of sentences and a disambigous word if the word has the same meaning in both sentences.|
|[13.WSC/DPR](https://www.aaai.org/ocs/index.php/KR/KR12/paper/view/4492/0)       | Predict for an ambiguous pronoun in a sentence what it is referring to.  |
|[14.Summarization](https://arxiv.org/abs/1506.03340)          | Summarize text into a shorter representation.|
|[15.SQuAD](https://arxiv.org/abs/1606.05250)                  | Answer a question for a given context.|
|[16.WMT1.](https://arxiv.org/abs/1706.03762)                  | Translate English to German|
|[17.WMT2.](https://arxiv.org/abs/1706.03762)                   | Translate English to French|
|[18.WMT3.](https://arxiv.org/abs/1706.03762)                   | Translate English to Romanian|

[refer to this notebook](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more) to see how to use every T5 Task.




## Question Answering
[Question answering example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more))

Predict an `answer` to a `question` based on input `context`.    
This is based on [SQuAD - Context based question answering](https://arxiv.org/abs/1606.05250)


|Predicted Answer | Question | Context | 
|-----------------|----------|------|
|carbon monoxide| What does increased oxygen concentrations in the patient’s lungs displace? | Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
|pie| What did Joey eat for breakfast?| Once upon a time, there was a squirrel named Joey. Joey loved to go outside and play with his cousin Jimmy. Joey and Jimmy played silly games together, and were always laughing. One day, Joey and Jimmy went swimming together 50 at their Aunt Julie’s pond. Joey woke up early in the morning to eat some food before they left. Usually, Joey would eat cereal, fruit (a pear), or oatmeal for breakfast. After he ate, he and Jimmy went to the pond. On their way there they saw their friend Jack Rabbit. They dove into the water and swam for several hours. The sun was out, but the breeze was cold. Joey and Jimmy got out of the water and started walking home. Their fur was wet, and the breeze chilled them. When they got home, they dried off, and Jimmy put on his favorite purple shirt. Joey put on a blue shirt with red and green dots. The two squirrels ate some food that Joey’s mom, Jasmine, made and went off to bed,'|  

```python
# Set the task on T5
t5['t5'].setTask('question ') 


# define Data, add additional tags between sentences
data = ['''
What does increased oxygen concentrations in the patient’s lungs displace? 
context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
''']


#Predict on text data with T5
t5.predict(data)
```

### How to configure T5 task parameter for Squad Context based question answering and pre-process data
`.setTask('question:)` and prefix the context which can be made up of multiple sentences with `context:`

### Example pre-processed input for T5 Squad Context based question answering:
```
question: What does increased oxygen concentrations in the patient’s lungs displace? 
context: Hyperbaric (high-pressure) medicine uses special oxygen chambers to increase the partial pressure of O 2 around the patient and, when needed, the medical staff. Carbon monoxide poisoning, gas gangrene, and decompression sickness (the ’bends’) are sometimes treated using these devices. Increased O 2 concentration in the lungs helps to displace carbon monoxide from the heme group of hemoglobin. Oxygen gas is poisonous to the anaerobic bacteria that cause gas gangrene, so increasing its partial pressure helps kill them. Decompression sickness occurs in divers who decompress too quickly after a dive, resulting in bubbles of inert gas, mostly nitrogen and helium, forming in their blood. Increasing the pressure of O 2 as soon as possible is part of the treatment.
```



## Text Summarization
[Summarization example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more)

`Summarizes` a paragraph into a shorter version with the same semantic meaning, based on [Text summarization](https://arxiv.org/abs/1506.03340)

```python
# Set the task on T5
pipe = nlu.load('summarize')

# define Data, add additional tags between sentences
data = [
'''
The belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth .
''',
'''  Calculus, originally called infinitesimal calculus or "the calculus of infinitesimals", is the mathematical study of continuous change, in the same way that geometry is the study of shape and algebra is the study of generalizations of arithmetic operations. It has two major branches, differential calculus and integral calculus; the former concerns instantaneous rates of change, and the slopes of curves, while integral calculus concerns accumulation of quantities, and areas under or between curves. These two branches are related to each other by the fundamental theorem of calculus, and they make use of the fundamental notions of convergence of infinite sequences and infinite series to a well-defined limit.[1] Infinitesimal calculus was developed independently in the late 17th century by Isaac Newton and Gottfried Wilhelm Leibniz.[2][3] Today, calculus has widespread uses in science, engineering, and economics.[4] In mathematics education, calculus denotes courses of elementary mathematical analysis, which are mainly devoted to the study of functions and limits. The word calculus (plural calculi) is a Latin word, meaning originally "small pebble" (this meaning is kept in medicine – see Calculus (medicine)). Because such pebbles were used for calculation, the meaning of the word has evolved and today usually means a method of computation. It is therefore used for naming specific methods of calculation and related theories, such as propositional calculus, Ricci calculus, calculus of variations, lambda calculus, and process calculus.'''
]


#Predict on text data with T5
pipe.predict(data)
```

| Predicted summary| Text | 
|------------------|-------|
| manchester united face newcastle in the premier league on wednesday . louis van gaal's side currently sit two points clear of liverpool in fourth . the belgian duo took to the dance floor on monday night with some friends .            | the belgian duo took to the dance floor on monday night with some friends . manchester united face newcastle in the premier league on wednesday . red devils will be looking for just their second league away win in seven . louis van gaal’s side currently sit two points clear of liverpool in fourth . | 


## Binary Sentence similarity/ Paraphrasing
[Binary sentence similarity example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more)
Classify whether one sentence is a re-phrasing or similar to another sentence      
This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf) and based on [MRPC - Binary Paraphrasing/ sentence similarity classification ](https://www.aclweb.org/anthology/I05-5002.pdf)

```
t5 = nlu.load('en.t5.base')
# Set the task on T5
t5['t5'].setTask('mrpc ')

# define Data, add additional tags between sentences
data = [
''' sentence1: We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said .
sentence2: Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11 "
'''
,
'''  
sentence1: I like to eat peanutbutter for breakfast
sentence2: 	I like to play football.
'''
]

#Predict on text data with T5
t5.predict(data)
```
| Sentence1 | Sentence2 | prediction|
|------------|------------|----------|
|We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said .| Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11 " . | equivalent | 
| I like to eat peanutbutter for breakfast| I like to play football | not_equivalent | 


### How to configure T5 task for MRPC and pre-process text
`.setTask('mrpc sentence1:)` and prefix second sentence with `sentence2:`

### Example pre-processed input for T5 MRPC - Binary Paraphrasing/ sentence similarity

```
mrpc 
sentence1: We acted because we saw the existing evidence in a new light , through the prism of our experience on 11 September , " Rumsfeld said . 
sentence2: Rather , the US acted because the administration saw " existing evidence in a new light , through the prism of our experience on September 11",
```



## Regressive Sentence similarity/ Paraphrasing

Measures how similar two sentences are on a scale from 0 to 5 with 21 classes representing a regressive label.     
This is a sub-task of [GLUE](https://arxiv.org/pdf/1804.07461.pdf) and based on[STSB - Regressive semantic sentence similarity](https://www.aclweb.org/anthology/S17-2001/) .

```python
t5 = nlu.load('en.t5.base')
# Set the task on T5
t5['t5'].setTask('stsb ') 

# define Data, add additional tags between sentences
data = [
             
              ''' sentence1:  What attributes would have made you highly desirable in ancient Rome?  
                  sentence2:  How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?'
              '''
             ,
             '''  
              sentence1: What was it like in Ancient rome?
              sentence2: 	What was Ancient rome like?
              ''',
              '''  
              sentence1: What was live like as a King in Ancient Rome??
              sentence2: 	What was Ancient rome like?
              '''

             ]



#Predict on text data with T5
t5.predict(data)

```

| Question1 | Question2 | prediction|
|------------|------------|----------|
|What attributes would have made you highly desirable in ancient Rome?        | How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER? | 0 | 
|What was it like in Ancient rome?  | What was Ancient rome like?| 5.0 | 
|What was live like as a King in Ancient Rome??       | What is it like to live in Rome? | 3.2 | 


### How to configure T5 task for stsb and pre-process text
`.setTask('stsb sentence1:)` and prefix second sentence with `sentence2:`




### Example pre-processed input for T5 STSB - Regressive semantic sentence similarity

```
stsb
sentence1: What attributes would have made you highly desirable in ancient Rome?        
sentence2: How I GET OPPERTINUTY TO JOIN IT COMPANY AS A FRESHER?',
```





## Grammar Checking
[Grammar checking with T5 example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/sequence2sequence/T5_tasks_summarize_question_answering_and_more))
Judges if a sentence is grammatically acceptable.    
Based on [CoLA - Binary Grammatical Sentence acceptability classification](https://nyu-mll.github.io/CoLA/)

```python
pipe = nlu.load('grammar_correctness')
# Set the task on T5
pipe['t5'].setTask('cola sentence: ')
# define Data
data = ['Anna and Mike is going skiing and they is liked is','Anna and Mike like to dance']
#Predict on text data with T5
pipe.predict(data)
```
|sentence  | prediction|
|------------|------------|
| Anna and Mike is going skiing and they is liked is | unacceptable |      
| Anna and Mike like to dance | acceptable | 


## Document Normalization
[Document Normalizer example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/text_pre_processing_and_cleaning/document_normalizer_demo.ipynb)     
The DocumentNormalizer extracts content from HTML or XML documents, applying either data cleansing using an arbitrary number of custom regular expressions either data extraction following the different parameters

```python
pipe = nlu.load('norm_document')
data = '<!DOCTYPE html> <html> <head> <title>Example</title> </head> <body> <p>This is an example of a simple HTML page with one paragraph.</p> </body> </html>'
df = pipe.predict(data,output_level='document')
df
```
|text|normalized_text|
|------|-------------|
| `<!DOCTYPE html> <html> <head> <title>Example</title> </head> <body> <p>This is an example of a simple HTML page with one paragraph.</p> </body> </html>`       |Example This is an example of a simple HTML page with one paragraph.|

## Word Segmenter
[Word Segmenter Example](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/component_examples/multilingual/japanese_ner_pos_and_tokenization.ipynb)     
The WordSegmenter segments languages without any rule-based tokenization such as Chinese, Japanese, or Korean
```python
pipe = nlu.load('ja.segment_words')
# japanese for 'Donald Trump and Angela Merkel dont share many opinions'
ja_data = ['ドナルド・トランプとアンゲラ・メルケルは多くの意見を共有していません']
df = pipe.predict(ja_data, output_level='token')
df

```

|	token|
|--------|
|	ドナルド|
|	・|
|	トランプ|
|	と|
|	アンゲラ|
|	・|
|	メルケル|
|	は|
|	多く|
|	の|
|	意見|
|	を|
|	共有|
|	し|
|	て|
|	い|
|	ませ|
|	ん|


### Installation

```bash
# PyPi
!pip install nlu pyspark==2.4.7
#Conda
# Install NLU from Anaconda/Conda
conda install -c johnsnowlabs nlu
```


### Additional NLU ressources
- [NLU Website](https://nlu.johnsnowlabs.com/)
- [All NLU Tutorial Notebooks](https://nlu.johnsnowlabs.com/docs/en/notebooks)
- [NLU Videos and Blogposts on NLU](https://nlp.johnsnowlabs.com/learn#pythons-nlu-library)
- [NLU on Github](https://github.com/JohnSnowLabs/nlu)


##  NLU 1.0.6 Release Notes
### Trainable Multi Label Classifiers, predict Stackoverflow Tags and much more in 1 Line of with NLU 1.0.6
We are glad to announce NLU 1.0.6 has been released!
NLU 1.0.6 comes with the Multi Label classifier, it can learn to map strings to multiple labels.
The Multi Label Classifier is using Bidirectional GRU and CNNs inside TensorFlow and supports up to 100 classes.

### NLU 1.0.6 New Features
- Multi Label Classifier
   - The Multi Label Classifier learns a 1 to many mapping between text and labels. This means it can predict multiple labels at the same time for a given input string. This is very helpful for tasks similar to content tag prediction (HashTags/RedditTags/YoutubeTags/Toxic/E2e etc..)
   - Support up to 100 classes
   - Pre-trained Multi Label Classifiers are already avaiable as [Toxic](https://nlu.johnsnowlabs.com/docs/en/examples#toxic-classifier) and [E2E](https://nlu.johnsnowlabs.com/docs/en/examples#e2e-classifier) classifiers

####  Multi Label Classifier
- [ Train Multi Label Classifier on E2E dataset Demo](https://colab.research.google.com/drive/15ZqfNUqliRKP4UgaFcRg5KOSTkqrtDXy?usp=sharing)
- [Train Multi Label  Classifier on Stack Overflow Question Tags dataset Demo](https://colab.research.google.com/drive/1Y0pYdUMKSs1ZP0NDcKgVECqkKD9ShIdc?usp=sharing)       
  This model can predict multiple labels for one sentence.
  To train the Multi Label text classifier model, you must pass a dataframe with a ```text``` column and a ```y``` column for the label.   
  The ```y``` label must be a string column where each label is seperated with a seperator.     
  By default, ```,``` is assumed as line seperator.      
  If your dataset is using a different label seperator, you must configure the ```label_seperator``` parameter while calling the ```fit()``` method.

By default *Universal Sentence Encoder Embeddings (USE)* are used as sentence embeddings for training.

```python
fitted_pipe = nlu.load('train.multi_classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

If you add a nlu sentence embeddings reference, before the train reference, NLU will use that Sentence embeddings instead of the default USE.
```python
#Train on BERT sentence emebddings
fitted_pipe = nlu.load('embed_sentence.bert train.multi_classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

Configure a custom line seperator
```python
#Use ; as label seperator
fitted_pipe = nlu.load('embed_sentence.electra train.multi_classifier').fit(train_df, label_seperator=';')
preds = fitted_pipe.predict(train_df)
```


### NLU 1.0.6 Enhancements
- Improved outputs for Toxic and E2E Classifier.
  - by default, all predicted classes and their confidences which are above the threshold will be returned inside of a list in the Pandas dataframe
  - by configuring meta=True, the confidences for all classes will be returned.


### NLU 1.0.6 New Notebooks and Tutorials

- [ Train Multi Label Classifier on E2E dataset](https://colab.research.google.com/drive/15ZqfNUqliRKP4UgaFcRg5KOSTkqrtDXy?usp=sharing)
- [Train Multi Label  Classifier on Stack Overflow Question Tags dataset](https://drive.google.com/file/d/1Nmrncn-y559od3AKJglwfJ0VmZKjtMAF/view?usp=sharing)

### NLU 1.0.6 Bug-fixes
- Fixed a bug that caused ```en.ner.dl.bert``` to be inaccessible
- Fixed a bug that caused ```pt.ner.large``` to be inaccessible
- Fixed a bug that caused USE embeddings not properly beeing configured to document level output when using multiple embeddings at the same time


##  NLU 1.0.5 Release Notes 

### Trainable Part of Speech Tagger (POS), Sentiment Classifier with BERT/USE/ELECTRA sentence embeddings in 1 Line of code! Latest NLU Release 1.0.5
We are glad to announce NLU 1.0.5 has been released!       
This release comes with a **trainable Sentiment classifier** and a **Trainable Part of Speech (POS)** models!       
These Neural Network Architectures achieve the state of the art (SOTA) on most **binary Sentiment analysis** and **Part of Speech Tagging** tasks!       
You can train the Sentiment Model on any of the **100+ Sentence Embeddings** which include **BERT, ELECTRA, USE, Multi Lingual BERT Sentence Embeddings** and many more!       
Leverage this and achieve the state of the art in any of your datasets, all of this in **just 1 line of Python code**

### NLU 1.0.5 New Features
- Trainable Sentiment DL classifier
- Trainable POS

### NLU 1.0.5 New Notebooks and Tutorials 
- [Sentiment Classification Training Demo](https://colab.research.google.com/drive/1f-EORjO3IpvwRAktuL4EvZPqPr2IZ_g8?usp=sharing)
- [Part Of Speech Tagger Training demo](https://colab.research.google.com/drive/1CZqHQmrxkDf7y3rQHVjO-97tCnpUXu_3?usp=sharing)

### Sentiment Classifier Training
[Sentiment Classification Training Demo](https://colab.research.google.com/drive/1f-EORjO3IpvwRAktuL4EvZPqPr2IZ_g8?usp=sharing)

To train the Binary Sentiment classifier model, you must pass a dataframe with a 'text' column and a 'y' column for the label.

By default *Universal Sentence Encoder Embeddings (USE)* are used as sentence embeddings.

```python
fitted_pipe = nlu.load('train.sentiment').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

If you add a nlu sentence embeddings reference, before the train reference, NLU will use that Sentence embeddings instead of the default USE.

```python
#Train Classifier on BERT sentence embeddings
fitted_pipe = nlu.load('embed_sentence.bert train.classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

```python
#Train Classifier on ELECTRA sentence embeddings
fitted_pipe = nlu.load('embed_sentence.electra train.classifier').fit(train_df)
preds = fitted_pipe.predict(train_df)
```

### Part Of Speech Tagger Training 
[Part Of Speech Tagger Training demo](https://colab.research.google.com/drive/1CZqHQmrxkDf7y3rQHVjO-97tCnpUXu_3?usp=sharing)

```python
fitted_pipe = nlu.load('train.pos').fit(train_df)
preds = fitted_pipe.predict(train_df)
```



### NLU 1.0.5 Installation changes
Starting from version 1.0.5 NLU will not automatically install pyspark for users anymore.      
This enables easier customizing the Pyspark version which makes it easier to use in various cluster enviroments.

To install NLU from now on, please run
```bash
pip install nlu pyspark==2.4.7 
```
or install any pyspark>=2.4.0 with pyspark<3

### NLU 1.0.5 Improvements
- Improved Databricks path handling for loading and storing models.




## NLU  1.0.4 Release Notes 
##  John Snow Labs NLU 1.0.4 : Trainable Named Entity Recognizer (NER) , achieve SOTA in 1 line of code and easy scaling to 100's of Spark nodes
We are glad to announce NLU 1.0.4 releases the State of the Art breaking Neural Network architecture for NER, Char CNNs - BiLSTM - CRF!

```python
#fit and predict in 1 line!
nlu.load('train.ner').fit(dataset).predict(dataset)


#fit and predict in 1 line with BERT!
nlu.load('bert train.ner').fit(dataset).predict(dataset)


#fit and predict in 1 line with ALBERT!
nlu.load('albert train.ner').fit(dataset).predict(dataset)


#fit and predict in 1 line with ELMO!
nlu.load('elmo train.ner').fit(dataset).predict(dataset)

```



Any NLU pipeline stored can now be loaded as pyspark ML pipeline
```python
# Ready for big Data with Spark distributed computing
import pyspark
nlu_pipe.save(path)
pyspark_pipe = pyspark.ml.PipelineModel.load(stored_model_path)
pyspark_pipe.transform(spark_df)
```


### NLU 1.0.4 New Features
- Trainable  [Named Entity Recognizer](https://nlp.johnsnowlabs.com/docs/en/annotators#ner-dl-named-entity-recognition-deep-learning-annotator)
- NLU pipeline loadable as Spark pipelines

### NLU 1.0.4 New Notebooks,Tutorials and Docs
- [NER training demo](https://colab.research.google.com/drive/1_GwhdXULq45GZkw3157fAOx4Wqo-fmFV?usp=sharing)        
- [Multi Class Text Classifier Training Demo updated to showcase usage of different Embeddings](https://colab.research.google.com/drive/12FA2TVvvRWw4pRhxDnK32WAzl9dbF6Qw?usp=sharing)         
- [New Documentation Page on how to train Models with NLU](https://nlu.johnsnowlabs.com/docs/en/training)
- Databricks Notebook showcasing Scaling with NLU


## NLU 1.0.4 Bug Fixes
- Fixed a bug that NER token confidences do not appear. They now appear when nlu.load('ner').predict(df, meta=True) is called.
- Fixed a bug that caused some Spark NLP models to not be loaded properly in offline mode



## 1.0.3 Release Notes 
We are happy to announce NLU 1.0.3 comes with a lot new features, training classifiers, saving them and loading them offline, enabling running NLU with no internet connection, new notebooks and articles!

### NLU 1.0.3 New Features
- Train a Deep Learning classifier in 1 line! The popular [ClassifierDL](https://nlp.johnsnowlabs.com/docs/en/annotators#classifierdl-multi-class-text-classification)
which can achieve state of the art results on any multi class text classification problem is now trainable!
All it takes is just nlu.load('train.classifier).fit(dataset) . Your dataset can be a Pandas/Spark/Modin/Ray/Dask dataframe and needs to have a column named x for text data and a column named y for labels
- Saving pipelines to HDD is now possible with nlu.save(path)
- Loading pipelines from disk now possible with nlu.load(path=path). 
- NLU offline mode: Loading from disk makes running NLU offline now possible, since you can load pipelines/models from your local hard drive instead of John Snow Labs AWS servers.

### NLU 1.0.3 New Notebooks and Tutorials
- New colab notebook showcasing nlu training, saving and loading from disk
- [Sentence Similarity with BERT, Electra and Universal Sentence Encoder Medium Tutorial](https://medium.com/spark-nlp/easy-sentence-similarity-with-bert-sentence-embeddings-using-john-snow-labs-nlu-ea078deb6ebf)
- [Sentence Similarity with BERT, Electra and Universal Sentence Encoder](https://colab.research.google.com/drive/1LtOdtXtRJ3_N8kYywPd5k2AJMCGcgAdN?usp=sharing)
- [Train a Deep Learning Classifier ](https://colab.research.google.com/drive/12FA2TVvvRWw4pRhxDnK32WAzl9dbF6Qw?usp=sharing)
- [Sentence Detector Notebook Updated](https://colab.research.google.com/drive/1CAXEdRk_q3U5qbMXsxoVyZRwvonKthhF?usp=sharing)
- [New Workshop video](https://events.johnsnowlabs.com/cs/c/?cta_guid=8b2b188b-92a3-48ba-ad7e-073b384425b0&signature=AAH58kFAHrVT-HfvWFxdTg_lm8reKUdTBw&pageId=25538044150&placement_guid=c659363c-2188-4c86-945f-5cfb7b42fcfc&click=8cd42d22-2f03-4358-a9e8-0d8f9aa33139&hsutk=c7a000001cda197314f90175e307161f&canon=https%3A%2F%2Fevents.johnsnowlabs.com%2Fwebinars&utm_referrer=https%3A%2F%2Fwww.johnsnowlabs.com%2F&portal_id=1794529&redirect_url=APefjpGh4Q9Hy0Mg9Ezy0_kJOOLC3l5QYyJsCSfZc1Lf61qrn2Bk6OQIJj65atZ9zzzrNrxuDPk5EHt94G0ZcIJaP_QMuD_E7fnMeJs4bQrEdLl7HE2MC4WNHGB6t1cqABfjZntS_TYSaj02yJNDf6p7Zaj9OYy0qQCmM8bbeuVgxUe6s5946UqHDsVHrpY0Oa2Fs7DJXIahZsB08hGkVj3qSHIM5vpjsA)


### NLU 1.0.3 Bug fixes
- Sentence Detector bugfix 




## NLU 1.0.2 Release Notes 

We are glad to announce nlu 1.0.2 is released!

### NLU 1.0.2  Enhancements
- More semantically concise output levels sentence and document enforced : 
  - If a pipe is set to output_level='document' : 
    -  Every Sentence Embedding will generate 1 Embedding per Document/row in the input Dataframe, instead of 1 embedding per sentence. 
    - Every  Classifier will classify an entire Document/row 
    - Each row in the output DF is a 1 to 1 mapping of the original input DF. 1 to 1 mapping from input to output.
  - If a pipe is set to output_level='sentence' : 
    -  Every Sentence Embedding will generate 1 Embedding per Sentence, 
    - Every  Classifier will classify exactly one sentence
    - Each row in the output DF can is mapped to one row in the input DF, but one row in the input DF can have multiple corresponding rows in the output DF. 1 to N mapping from input to output.
- Improved generation of column names for classifiers. based on input nlu reference
- Improved generation of column names for embeddings, based on input nlu reference
- Improved automatic output level inference
- Various test updates
- Integration of CI pipeline with Github Actions 

### New  Documentation is out!
Check it out here :  http://nlu.johnsnowlabs.com/


## NLU 1.0.1 Release Notes 

### NLU 1.0.1 Bugfixes
- Fixed bug that caused NER pipelines to crash in NLU when input string caused the NER model to predict without additional metadata

## 1.0 Release Notes 
- Automatic to Numpy conversion of embeddings
- Added various testing classes
- [New 6 embeddings at once notebook with t-SNE and Medium article](https://medium.com/spark-nlp/1-line-of-code-for-bert-albert-elmo-electra-xlnet-glove-part-of-speech-with-nlu-and-t-sne-9ebcd5379cd)
 <img src="https://miro.medium.com/max/1296/1*WI4AJ78hwPpT_2SqpRpolA.png" >
- Integration of Spark NLP 2.6.2 enhancements and bugfixes https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.2
- Updated old T-SNE notebooks with more elegant and simpler generation of t-SNE embeddings 

</div><div class="h3-box" markdown="1">

## 0.2.1 Release Notes 
- Various bugfixes
- Improved output column names when using multiple classifirs at once

</div><div class="h3-box" markdown="1">

## 0.2 Release Notes 
-   Improved output column names  classifiers

</div><div class="h3-box" markdown="1">
    
## 0.1 Release Notes



# 1.0 Release Notes 
- Automatic to Numpy conversion of embeddings
- Added various testing classes
- [New 6 embeddings at once notebook with t-SNE and Medium article](https://medium.com/spark-nlp/1-line-of-code-for-bert-albert-elmo-electra-xlnet-glove-part-of-speech-with-nlu-and-t-sne-9ebcd5379cd)
 <img src="https://miro.medium.com/max/1296/1*WI4AJ78hwPpT_2SqpRpolA.png" >
- Integration of Spark NLP 2.6.2 enhancements and bugfixes https://github.com/JohnSnowLabs/spark-nlp/releases/tag/2.6.2
- Updated old T-SNE notebooks with more elegant and simpler generation of t-SNE embeddings 

# 0.2.1 Release Notes 
- Various bugfixes
- Improved output column names when using multiple classifirs at once

# 0.2 Release Notes 
-   Improved output column names  classifiers
    
# 0.1 Release Notes
We are glad to announce that NLU 0.0.1 has been released!
NLU makes the 350+ models and annotators in Spark NLPs arsenal available in just 1 line of python code and it works with Pandas dataframes!
A picture says more than a 1000 words, so here is a demo clip of the 12 coolest features in NLU, all just in 1 line!

</div><div class="h3-box" markdown="1">

## NLU in action 
<img src="http://ckl-it.de/wp-content/uploads/2020/08/My-Video6.gif" width="1800" height="500"/>

</div><div class="h3-box" markdown="1">

## What does NLU 0.1 include?

## NLU in action 
<img src="http://ckl-it.de/wp-content/uploads/2020/08/My-Video6.gif" width="1800" height="500"/>

# What does NLU 0.1 include?
 - NLU provides everything a data scientist might want to wish for in one line of code!
 - 350 + pre-trained models
 - 100+ of the latest NLP word embeddings ( BERT, ELMO, ALBERT, XLNET, GLOVE, BIOBERT, ELECTRA, COVIDBERT) and different variations of them
 - 50+ of the latest NLP sentence embeddings ( BERT, ELECTRA, USE) and different variations of them
 - 50+ Classifiers (NER, POS, Emotion, Sarcasm, Questions, Spam)
 - 40+ Supported Languages
 - Labeled and Unlabeled Dependency parsing
 - Various Text Cleaning and Pre-Processing methods like Stemming, Lemmatizing, Normalizing, Filtering, Cleaning pipelines and more

 </div><div class="h3-box" markdown="1">

## NLU 0.1 Features Google Collab Notebook Demos

- Named Entity Recognition (NER)
    - [NER pretrained on ONTO Notes](https://colab.research.google.com/drive/1_sgbJV3dYPZ_Q7acCgKWgqZkWcKAfg79?usp=sharing)
    - [NER pretrained on CONLL](https://colab.research.google.com/drive/1CYzHfQyFCdvIOVO2Z5aggVI9c0hDEOrw?usp=sharing)
</div><div class="h3-box" markdown="1">

- Part of speech (POS)
    - [POS pretrained on ANC dataset](https://colab.research.google.com/drive/1tW833T3HS8F5Lvn6LgeDd5LW5226syKN?usp=sharing)

</div><div class="h3-box" markdown="1">

# NLU 0.1 Features Google Collab Notebook Demos

- Named Entity Recognition (NER)
    -[NER pretrained on ONTO Notes](https://colab.research.google.com/drive/1_sgbJV3dYPZ_Q7acCgKWgqZkWcKAfg79?usp=sharing)
    -[NER pretrained on CONLL](https://colab.research.google.com/drive/1CYzHfQyFCdvIOVO2Z5aggVI9c0hDEOrw?usp=sharing)
- Part of speech (POS)
    - [POS pretrained on ANC dataset](https://colab.research.google.com/drive/1tW833T3HS8F5Lvn6LgeDd5LW5226syKN?usp=sharing)
- Classifiers
    - [Unsupervised Keyword Extraction with YAKE](https://colab.research.google.com/drive/1BdomIc1nhrGxLFOpK5r82Zc4eFgnIgaO?usp=sharing)
    - [Toxic Text Classifier](https://colab.research.google.com/drive/1QRG5ZtAvoJAMZ8ytFMfXj_W8ogdeRi9m?usp=sharing)
    - [Twitter Sentiment Classifier](https://colab.research.google.com/drive/1H1Gekn2qzXzOf5rrT8LmHmmuoOGsiu8m?usp=sharing)
    - [Movie Review Sentiment Classifier](https://colab.research.google.com/drive/1k5x1zxnG4bBkmYAc-bc63sMA4-oQ6-dP?usp=sharing)
    - [Sarcasm Classifier](https://colab.research.google.com/drive/1XffsjlRp9wxZgxyYvEF9bG2CiX-pjBEw?usp=sharing)
    - [50 Class Questions Classifier](https://colab.research.google.com/drive/1OwlmLzwkcJKhuz__RUH74O9HqFZutxzS?usp=sharing)
    - [20 Class Languages Classifier](https://colab.research.google.com/drive/1CzMfRFJZsj4j1fhormDQdHOIV5IybC57?usp=sharing)
    - [Fake News Classifier](https://colab.research.google.com/drive/1QuoeGLgmtkUnDQQ2oVS1tuZC2qHDj3p9?usp=sharing)
    - [E2E Classifier](https://colab.research.google.com/drive/1OSkiXGEpKlm9HWDoVb42uLNQQgb7nqNZ?usp=sharing)
    - [Cyberbullying Classifier](https://colab.research.google.com/drive/1OSkiXGEpKlm9HWDoVb42uLNQQgb7nqNZ?usp=sharing)
    - [Spam Classifier](https://colab.research.google.com/drive/1u-8Fs3Etz07bFNx0CDV_le3Xz73VbK0z?usp=sharing)

</div><div class="h3-box" markdown="1">

- Word and Sentence Embeddings 
    - [BERT Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1Rg1vdSeq6sURc48RV8lpS47ja0bYwQmt?usp=sharing)
    - [BERT Sentence Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1FmREx0O4BDeogldyN74_7Lur5NeiOVye?usp=sharing)
    - [ALBERT Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/18yd9pDoPkde79boTbAC8Xd03ROKisPsn?usp=sharing)
    - [ELMO Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1TtNYB9z0yH8d1ZjfxkH0TVxQ2O_iOYVV?usp=sharing)
    - [XLNET Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1C9T29QA00yjLuJ1yEMTbjUQMpUv35pHb?usp=sharing)
    - [ELECTRA Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1FueGEaOj2JkbqHzdmxwKrNMHzgVt4baE?usp=sharing)
    - [COVIDBERT Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1Yzc-GuNQyeWewJh5USTN7PbbcJvd-D7s?usp=sharing)
    - [BIOBERT Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1llANd-XGD8vkGNMcqTi_8Dr_Ys6cr83W?usp=sharing)
    - [GLOVE Word Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1IQxf4pJ_EnrIDyd0fAX-dv6u0YQWae2g?usp=sharing)
    - [USE Sentence Embeddings and T-SNE plotting](https://colab.research.google.com/drive/1gZzOMiCovmrp7z8FIidzDTLS0nt8kPJT?usp=sharing)

</div><div class="h3-box" markdown="1">

- Depenency Parsing 
    - [Untyped Dependency Parsing](https://colab.research.google.com/drive/1PC8ga_NFlOcTNeDVJY4x8Pl5oe0jVmue?usp=sharing)
    - [Typed Dependency Parsing](https://colab.research.google.com/drive/1KXUqcF8e-LU9cXnHE8ni8z758LuFPvY7?usp=sharing)

</div><div class="h3-box" markdown="1">

- Depenency Parsing 
    -[Untyped Dependency Parsing](https://colab.research.google.com/drive/1PC8ga_NFlOcTNeDVJY4x8Pl5oe0jVmue?usp=sharing)
    -[Typed Dependency Parsing](https://colab.research.google.com/drive/1KXUqcF8e-LU9cXnHE8ni8z758LuFPvY7?usp=sharing)

- Text Pre Processing and Cleaning
    - [Tokenization](https://colab.research.google.com/drive/13BC6k6gLj1w5RZ0SyHjKsT2EOwJwbYwb?usp=sharing)
    - [Stopwords removal](https://colab.research.google.com/drive/1nWob4u93t2EJYupcOIanuPBDfShtYjGT?usp=sharing)
    - [Stemming](https://colab.research.google.com/drive/1gKTJJmffR9wz13Ms3pDy64jhUI8ZHZYu?usp=sharing)
    - [Lemmatization](https://colab.research.google.com/drive/1cBtx9cVCjavt-Oq5TG1lO-9JfUfqznnK?usp=sharing)
    - [Normalizing](https://colab.research.google.com/drive/1kfnnwkiQPQa465Jic6va9QXTRssU4mlX?usp=sharing)
    - [Spellchecking](https://colab.research.google.com/drive/1bnRR8FygiiN3zJz3mRdbjPBUvFsx6IVB?usp=sharing)
    - [Sentence Detecting](https://colab.research.google.com/drive/1CAXEdRk_q3U5qbMXsxoVyZRwvonKthhF?usp=sharing)

</div><div class="h3-box" markdown="1">

- Chunkers
    - [N Gram](https://colab.research.google.com/drive/1pgqoRJ6yGWbTLWdLnRvwG5DLSU3rxuMq?usp=sharing)
    - [Entity Chunking](https://colab.research.google.com/drive/1svpqtC3cY6JnRGeJngIPl2raqxdowpyi?usp=sharing)

</div><div class="h3-box" markdown="1">

- Matchers
    - [Date Matcher](https://colab.research.google.com/drive/1JrlfuV2jNGTdOXvaWIoHTSf6BscDMkN7?usp=sharing)

</div><div class="h3-box" markdown="1">


</div></div>
- Chunkers
    -[N Gram](https://colab.research.google.com/drive/1pgqoRJ6yGWbTLWdLnRvwG5DLSU3rxuMq?usp=sharing)
    -[Entity Chunking](https://colab.research.google.com/drive/1svpqtC3cY6JnRGeJngIPl2raqxdowpyi?usp=sharing)
- Matchers
    -[Date Matcher](https://colab.research.google.com/drive/1JrlfuV2jNGTdOXvaWIoHTSf6BscDMkN7?usp=sharing)



