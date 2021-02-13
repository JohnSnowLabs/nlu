class NameSpace():

    # NLU model_base_names =
    # These reference tell NLU to which component resolved to route a request, they help NLU map a NLP reference to the correct class
    word_embeddings = ['embed','bert','electra','albert','elmo','glove','xlnet','biobert','covidbert','tfhub_use']
    sentence_embeddings = ['embed_sentence','use', 'bert', 'electra','tfhub_use']
    classifiers = ['classify', 'e2e', 'emotion', 'sentiment', 'ner',
                   'pos', 'trec6','trec50', 'questions',
                   'sarcasm','emotion', 'spam','fakenews', 'cyberbullying',
                   'wiki','wiki_7', 'wiki_20','yake','toxic'
                   ]
    seq2seq = ['t5','marian','translate_to']
    actions = ['tokenize', 'sentence', 'embed', 'embed_sentence', 'embed_chunk','classify', 'chunk', 'pos', 'ner',
               'dep', 'dep.untyped', 'lemma', 'match', 'norm', 'spell','stem', 'stopwords','clean','ngram',
                'translate_to',

               ]
    trainable_model_references = ['classifier_dl']



    '''
    NLU REFERENCE FORMATS : 
    <lang>.<action>.<
    '''
    trainable_models = {
# 1 clasifier wraps iuip multiple algos (multi class/class/sentiment)
# Columns should be X and y, ML style
        #
        # map NLU references to NLP approaches
    'train.deep_sentence_detector' : '' ,
    'train.sentence_detector' : '' , # deep sentence detector alias
    'train.symmetric_spell' : '' ,
    'train.context_spell' : '' ,
    'train.spell' : '' , ## context spell alias
    'train.norvig_spell' : '' ,
    'train.unlabeled_dependency_parser' : '' ,
    'train.labeled_dependency_parser' : '' ,
    'train.classifier_dl' : '' ,
    'train.classifier' : '' , #classifier DL alias
    'train.named_entity_recognizer_dl' : '' ,
    'train.ner' : '' , # ner DL alias
    'train.vivekn_sentiment' : '' ,
    'train.sentiment_dl' : '' ,
    'train.sentiment' : '' , #sent DL alias
    'train.pos' : '' ,
    'train.multi_classifier' : '' ,

    }

    #Reference to all datasets for which we have pretrained models
    datasets = []
    chunk_embeddings = ['embed_sentence']
    # The vocabulary of the nlu Namespace. Any of this references give you a model
    # keys inside a language dict are NLU references and value is the name in SparkNLP

    component_alias_references = {
        # references for SparkNLPAnnotators without pretrained models.
        #  These are names for NLU components that can be created withouth a language prefix

        # multi lang pipes
        'lang': ('detect_language_375','pipe'),  # multi lang alias
        'lang.7': ('detect_language_7','pipe'),  # multi lang detector alias
        'lang.20': ('detect_language_20','pipe'),  # multi lang detector alias
        'lang.21' :  ('detect_language_21', 'pipe') ,
        'lang.43' :  ('detect_language_43', 'pipe') ,
        'lang.95' :  ('detect_language_95', 'pipe') ,
        'lang.99' :  ('detect_language_99', 'pipe') ,
        'lang.220' : ('detect_language_220', 'pipe') ,
        'lang.231' : ('detect_language_231', 'pipe') ,


        # Aliases
        'classify.lang': ('detect_language_375','pipe'),  # multi lang detector default
        'classify.lang.7': ('detect_language_7','pipe'),  # multi lang detector alias
        'classify.lang.20': ('detect_language_20','pipe'),  # multi lang detector alias
        'classify.lang.21' :  ('detect_language_21', 'pipe') ,
        'classify.lang.43' :  ('detect_language_43', 'pipe') ,
        'classify.lang.95' :  ('detect_language_95', 'pipe') ,
        'classify.lang.99' :  ('detect_language_99', 'pipe') ,
        'classify.lang.220' : ('detect_language_220', 'pipe') ,
        'classify.lang.231' : ('detect_language_231', 'pipe') ,


        # eng pipes
        'classify': ('analyze_sentiment','pipe'), #default classifier
        'explain': ('explain_document_ml','pipe'),  # default explain
        'explain.ml': ('explain_document_ml','pipe'),
        'explain.dl': ('explain_document_dl','pipe'),
        'ner.conll': ('recognize_entities_dl','pipe'),  # default ner
        'ner.dl': ('recognize_entities_dl','pipe'),
        'ner.bert': ('recognize_entities_bert','pipe'),
        'ner': ('onto_recognize_entities_sm','pipe'),  # default  ner.onto
        'ner.onto': ('onto_recognize_entities_sm','pipe'),  # default  ner.onto
        'ner.onto.sm': ('onto_recognize_entities_sm','pipe'),
        'ner.onto.lg': ('onto_recognize_entities_lg','pipe'),
        'match.datetime': ('match_datetime','pipe'),
        'match.text': ('text_matcher','model'),
        'match.regex': ('regex_matcher','model'),

        'match.pattern': ('match_pattern','pipe'),
        'match.chunks': ('match_chunks','pipe'),
        'match.phrases': ('match_phrases','pipe'),
        'clean.stop': ('clean_stop','pipe'),
        'clean.pattern': ('clean_pattern','pipe'),
        'clean.slang': ('clean_slang','pipe'),
        # 'spell': ('check_spelling','pipe'),  # bad spell_checker,
        'spell': ('spellcheck_dl','model'),  # default spell
        'sentiment': ('analyze_sentiment','pipe'),
        'emotion': ('classifierdl_use_emotion','model'), # default emotion model

        'sentiment.imdb': ('analyze_sentimentdl_use_imdb','pipe'),
        'sentiment.imdb.use': ('analyze_sentimentdl_use_imdb','pipe'),
        'sentiment.twitter.use': ('analyze_sentimentdl_use_twitter','pipe'),
        'sentiment.twitter': ('analyze_sentimentdl_use_twitter','pipe'),
        'dependency': ('dependency_parse','pipe'),


        'tokenize': ('spark_nlp_tokenizer', 'model'),  # tokenizer rule based model
        'stem': ('stemmer', 'model'),  # stem rule based model
        'norm': ('normalizer', 'model'),  #  rule based model
        'norm_document': ('normalizer', 'model'),  #  rule based model

        'chunk': ('default_chunker', 'model'),  #  rule based model
        'embed_chunk': ('chunk_embeddings', 'model'),  # rule based model
        'ngram': ('ngram', 'model'),  #  rule based model
    

        'lemma': ('lemma_antbnc', 'model'),  # lemma default en
        'lemma.antbnc': ('lemma_antbnc', 'model'),
        'pos': ('pos_anc', 'model'),  # pos default en
        'pos.anc': ('pos_anc', 'model'),
        'pos.ud_ewt': ('pos_ud_ewt', 'model'),
        # 'ner.crf' :'ner_crf', # crf not supported in NLU
        'ner.dl.glove.6B_100d': ('ner_dl', 'model'),
        'ner.dl.bert': ('ner_dl_bert', 'model'),  # points ner bert
        'ner.onto.glove.6B_100d': ('onto_100', 'model'),
        'ner.onto.glove.6B_300d': ('onto_300', 'model'),  # this uses multi lang embeds!
        'sentence_detector': ('sentence_detector_dl', 'model'),
        'sentence_detector.deep': ('sentence_detector_dl', 'model'), #ALIAS


        'sentence_detector.pragmatic': ('pragmatic_sentence_detector', 'model'), # todo

        # 'spell.symmetric': ('spellcheck_sd', 'model'), # TODO erronous
        'spell.norivg': ('spellcheck_norvig', 'model'),
        'sentiment.vivekn': ('sentiment_vivekn', 'model'),
        'dep.untyped.conllu': ('dependency_conllu', 'model'),
        'dep.untyped': ('dependency_conllu.untyped', 'model'),  # default untyped dependency
        'dep': ('dependency_typed_conllu', 'model'),  # default typed dependency
        'dep.typed': ('dependency_typed_conllu', 'model'),  # default typed dependency dataset
        'dep.typed.conllu': ('dependency_typed_conllu', 'model'),
        'stopwords': ('stopwords_en', 'model'),

        # embeddings models
        'embed': ('glove_100d','model'),  # default overall embed
        'glove': ('glove_100d', 'model'),  # default glove
        'embed.glove': ('glove_100d', 'model'),  # default glove en
        'embed.glove.100d': ('glove_100d', 'model'),
        'bert': ('small_bert_L2_128', 'model'),  # default bert
        'covidbert': ('covidbert_large_uncased','model'),

        'en.toxic': ('multiclassifierdl_use_toxic','model'),
        'en.e2e': ('multiclassifierdl_use_e2e','model'),
        'embed.bert': ('bert_base_uncased', 'model'),  # default bert
        'embed.bert_base_uncased': ('bert_base_uncased', 'model'),
        'embed.bert_base_cased': ('bert_base_cased', 'model'),
        'embed.bert_large_uncased': ('bert_large_uncased', 'model'),
        'embed.bert_large_cased': ('bert_large_cased', 'model'),
        'biobert': ('biobert_pubmed_base_cased', 'model'),  # alias
        'embed.biobert': ('biobert_pubmed_base_cased', 'model'),  # default bio bert
        'embed.biobert_pubmed_base_cased': ('biobert_pubmed_base_cased', 'model'),
        'embed.biobert_pubmed_large_cased': ('biobert_pubmed_large_cased', 'model'),
        'embed.biobert_pmc_base_cased': ('biobert_pmc_base_cased', 'model'),
        'embed.biobert_pubmed_pmc_base_cased': ('biobert_pubmed_pmc_base_cased', 'model'),
        'embed.biobert_clinical_base_cased': ('biobert_clinical_base_cased', 'model'),
        'embed.biobert_discharge_base_cased': ('biobert_discharge_base_cased', 'model'),
        'elmo': ('elmo', 'model'),

        'embed.electra': ('electra_small_uncased','model'),
        'electra': ('electra_small_uncased','model'),
        'e2e': ('multiclassifierdl_use_e2e','model'),

        'embed.elmo': ('elmo', 'model'),
        'embed_sentence': ('tfhub_use', 'model'),  # default use
        'embed_sentence.small_bert_L2_128': ('sent_small_bert_L2_128','model'),
        'embed_sentence.bert': ('sent_small_bert_L2_128','model'),
        'embed_sentence.electra': ('sent_electra_small_uncased','model'),

        'embed_sentence.use': ('tfhub_use', 'model'),  # default use
        'use': ('tfhub_use', 'model'),  # alias
        'embed_sentence.tfhub_use': ('tfhub_use', 'model'),
        'embed_sentence.use_lg': ('tfhub_use_lg', 'model'),  # alias
        'embed_sentence.tfhub_use_lg': ('tfhub_use_lg', 'model'),
        'albert': ('albert_base_uncased', 'model'),  # albert alias en
        'embed.albert_base_uncased': ('albert_base_uncased', 'model'),
        'embed.albert_large_uncased': ('albert_large_uncased', 'model'),
        'embed.albert_xlarge_uncased': ('albert_xlarge_uncased', 'model'),
        'embed.albert_xxlarge_uncased': ('albert_xxlarge_uncased', 'model'),
        'embed.xlnet': ('xlnet_base_cased', 'model'),  # xlnet default en
        'xlnet': ('xlnet_base_cased', 'model'),  # xlnet alias
        'embed.xlnet_base_cased': ('xlnet_base_cased', 'model'),
        'embed.xlnet_large_cased': ('xlnet_large_cased', 'model'),




        # classifiers and sentiment models
        'classify.trec6.use': ('classifierdl_use_trec6','model'),
        'classify.trec50.use': ('classifierdl_use_trec50','model'),
        'classify.questions': ('classifierdl_use_trec50','model'),
        'questions': ('classifierdl_use_trec50','model'),

        'classify.spam.use': ('classifierdl_use_spam','model'),
        'classify.fakenews.use': ('classifierdl_use_fakenews','model'),
        'classify.emotion.use': ('classifierdl_use_emotion','model'),
        'classify.cyberbullying.use': ('classifierdl_use_cyberbullying','model'),
        'classify.sarcasm.use': ('classifierdl_use_sarcasm','model'),
        'sentiment.imdb.glove': ('sentimentdl_glove_imdb','model'),
        'classify.trec6': ('classifierdl_use_trec6','model'),  # Alias withouth embedding
        'classify.trec50': ('classifierdl_use_trec50','model'),  # Alias withouth embedding
        'classify.spam': ('classifierdl_use_spam','model'),  # Alias withouth embedding
        'spam': ('classifierdl_use_spam','model'),  # Alias withouth embedding
        'toxic': ('multiclassifierdl_use_toxic','model'),

        'classify.fakenews': ('classifierdl_use_fakenews','model'),  # Alias withouth embedding
        'classify.emotion': ('classifierdl_use_emotion','model'),  # Alias withouth embedding
        'classify.cyberbullying': ('classifierdl_use_cyberbullying','model'),  # Alias withouth embedding
        'cyberbullying': ('classifierdl_use_cyberbullying','model'),  # Alias withouth embedding
        'cyber': ('classifierdl_use_cyberbullying','model'),  # Alias withouth embedding

        'classify.sarcasm': ('classifierdl_use_sarcasm','model'),  # Alias withouth embedding
        'sarcasm': ('classifierdl_use_sarcasm','model'),  # Alias withouth embedding

        'embed.glove.840B_300': ('glove_840B_300','model'),
        'embed.glove.6B_300': ('glove_6B_300','model'),
        'embed.bert_multi_cased': ('bert_multi_cased','model'),
        'classify.wiki_7': ('ld_wiki_7','model'),
        'classify.wiki_20': ('ld_wiki_20','model'),
        'yake': ('yake','model'),


        # 2.7.0 new aliases
        't5': ('t5_base','model'),
        't5.summarize': ('t5_base','model'),
        't5.classify.grammar_correctness': ('t5_base','model'),
        't5.classify.sentiment': ('t5_base','model'),
        't5.answer_question': ('t5_base','model'),

        # 2.7.0 new aliases
        't5': ('t5_base','model'),
        'summarize': ('t5_base','model','summarize','t5'),
        'grammar_correctness': ('t5_base','model','grammar_correctness','t5'),
        'answer_question': ('t5_base','model','answer_question','t5'),
        # 'classify.sentiment': ('t5_base','model'),

    }

    # multi lang models
    pretrained_pipe_references = {

        'da': {
                'da.explain': 'explain_document_sm',
                'da.explain.sm': 'explain_document_sm',
                'da.explain.md': 'explain_document_md',
                'da.explain.lg': 'explain_document_lg',
                'da.ner': 'entity_recognizer_sm',
                'da.ner.sm': 'entity_recognizer_sm',
                'da.ner.md': 'entity_recognizer_md',
                'da.ner.lg': 'entity_recognizer_lg'},
         
        'nl': {
            'nl.explain': 'explain_document_sm',  # default
            'nl.explain.sm': 'explain_document_sm',
            'nl.explain.md': 'explain_document_md',
            'nl.explain.lg': 'explain_document_lg',
            'nl.ner': 'entity_recognizer_sm',
            # default,calling it nl.ner this makes creating actual NER object impossible!
            'nl.ner.sm': 'entity_recognizer_sm',
            'nl.ner.md': 'entity_recognizer_md',
            'nl.ner.lg': 'entity_recognizer_lg',
        },
        'en': {

            'en.classify': 'analyze_sentiment', #default classifier
            'en.explain': 'explain_document_ml',  # default explain
            'en.explain.ml': 'explain_document_ml',
            'en.explain.dl': 'explain_document_dl',
            'en.ner': 'recognize_entities_dl',  # default ner
            'en.ner.conll': 'recognize_entities_dl',  # default ner

            'en.ner.dl': 'recognize_entities_dl',
            'en.ner.bert': 'recognize_entities_bert',
            # 'en.ner.onto': 'onto_recognize_entities_sm',  # default  ner.onto
            'en.ner.onto.sm': 'onto_recognize_entities_sm',
            'en.ner.onto.lg': 'onto_recognize_entities_lg',
            'en.match.datetime': 'match_datetime',
            'en.match.pattern': 'match_pattern',
            'en.match.chunks': 'match_chunks',
            'en.match.phrases': 'match_phrases',
            'en.clean.stop': 'clean_stop',
            'en.clean.pattern': 'clean_pattern',
            'en.clean.slang': 'clean_slang',
            'en.spell': 'check_spelling_dl',  # dfault spell

            'en.spell.dl': 'check_spelling_dl',
            'en.spell.context': 'check_spelling_dl',
            'en.sentiment': 'analyze_sentiment',
            'en.classify.sentiment': 'analyze_sentiment',

            'en.sentiment.imdb': 'analyze_sentimentdl_use_imdb',
            'en.sentiment.imdb.use': 'analyze_sentimentdl_use_imdb',
            'en.sentiment.twitter.use': 'analyze_sentimentdl_use_twitter',
            'en.sentiment.twitter': 'analyze_sentimentdl_use_twitter',
            'en.dependency': 'dependency_parse',


            #2.7
            'en.ner.onto.bert.base'  :'onto_recognize_entities_bert_base',
            'en.ner.onto.bert.large'  :'onto_recognize_entities_bert_large',
            'en.ner.onto.bert.medium'  :'onto_recognize_entities_bert_medium',
            'en.ner.onto.bert.mini'  :'onto_recognize_entities_bert_mini',
            'en.ner.onto.bert.small'  :'onto_recognize_entities_bert_small',
            'en.ner.onto.bert.tiny'  :'onto_recognize_entities_bert_tiny',
            'en.ner.onto.electra.base'  :'onto_recognize_entities_electra_base',
            'en.ner.onto.electra.small'  :'onto_recognize_entities_electra_small',


            # 2.7.1 and 2.7.2
            "en.sentiment.glove":"analyze_sentimentdl_glove_imdb",
            "en.sentiment.glove.imdb":"analyze_sentimentdl_glove_imdb",
            "en.classify.sentiment.glove.imdb":"analyze_sentimentdl_glove_imdb",
            "en.classify.sentiment.glove":"analyze_sentimentdl_glove_imdb",
            "en.classify.trec50.pipe":"classifierdl_use_trec50_pipeline",
            "en.ner.onto.large":"onto_recognize_entities_electra_large",



        },


        'sv': {
                'sv.explain': 'explain_document_sm',
               'sv.explain.sm': 'explain_document_sm',
               'sv.explain.md': 'explain_document_md',
               'sv.explain.lg': 'explain_document_lg',
               'sv.ner': 'entity_recognizer_sm',
               'sv.ner.sm': 'entity_recognizer_sm',
               'sv.ner.md': 'entity_recognizer_md',
               'sv.ner.lg': 'entity_recognizer_lg'},
    
    
        'fi' : {
            'fi.explain': 'explain_document_sm',
              'fi.explain.sm': 'explain_document_sm',
               'fi.explain.md': 'explain_document_md',
               'fi.explain.lg': 'explain_document_lg',
                'fi.ner': 'entity_recognizer_sm',
               'fi.ner.sm': 'entity_recognizer_sm',
               'fi.ner.md': 'entity_recognizer_md',
               'fi.ner.lg': 'entity_recognizer_lg'},

        'fr': {
            'fr.explain': 'explain_document_lg',  # default fr explain
            'fr.explain.lg': 'explain_document_lg',
            'fr.explain.md': 'explain_document_md',
            'fr.ner': 'entity_recognizer_lg',  # default fr ner pipe
            'fr.ner.lg': 'entity_recognizer_lg',
            'fr.ner.md': 'entity_recognizer_md',
        },
        'de': {
            'de.explain.document': 'explain_document_md',  # default de explain
            'de.explain.document.md': 'explain_document_md',
            'de.explain.document.lg': 'explain_document_lg',
            'de.ner.recognizer': 'entity_recognizer_md',  # default de ner
            'de.ner.recognizer.md': 'entity_recognizer_md',
            'de.ner.recognizer.lg': 'entity_recognizer_lg',
        },
        'it': {
            'it.explain.document': 'explain_document_md',  # it default explain
            'it.explain.document.md': 'explain_document_md',
            'it.explain.document.lg': 'explain_document_lg',
            'it.ner': 'entity_recognizer_md',  # it default ner
            'it.ner.md': 'entity_recognizer_md',
            'it.ner.lg': 'entity_recognizer_lg',
        },
        'no': {
            'no.explain': 'explain_document_sm',  # default no explain
            'no.explain.sm': 'explain_document_sm',
            'no.explain.md': 'explain_document_md',
            'no.explain.lg': 'explain_document_lg',
            'no.ner': 'entity_recognizer_sm',  # default no ner
            'no.ner.sm': 'entity_recognizer_sm',
            'no.ner.md': 'entity_recognizer_md',
            'no.ner.lg': 'entity_recognizer_lg',
        },
        'pl': {
            'pl.explain': 'explain_document_sm',  # defaul pl explain
            'pl.explain.sm': 'explain_document_sm',
            'pl.explain.md': 'explain_document_md',
            'pl.explain.lg': 'explain_document_lg',
            'pl.ner': 'entity_recognizer_sm',  # default pl ner
            'pl.ner.sm': 'entity_recognizer_sm',
            'pl.ner.md': 'entity_recognizer_md',
            'pl.ner.lg': 'entity_recognizer_lg',
        },
        'pt': {
            'pt.explain': 'explain_document_sm',  # default explain pt
            'pt.explain.sm': 'explain_document_sm',
            'pt.explain.md': 'explain_document_md',
            'pt.explain.lg': 'explain_document_lg',
            'pt.ner': 'entity_recognizer_sm',  # default ner pt
            'pt.ner.sm': 'entity_recognizer_sm',
            'pt.ner.md': 'entity_recognizer_md',
            'pt.ner.lg': 'entity_recognizer_lg',


        },
        'ru': {
            'ru.explain': 'explain_document_sm',  # default ru explain
            'ru.explain.sm': 'explain_document_sm',
            'ru.explain.md': 'explain_document_md',
            'ru.explain.lg': 'explain_document_lg',
            'ru.ner': 'entity_recognizer_sm',  # default ru ner
            'ru.ner.sm': 'entity_recognizer_sm',
            'ru.ner.md': 'entity_recognizer_md',
            'ru.ner.lg': 'entity_recognizer_lg',
        },
        'es': {
            'es.explain': 'explain_document_sm',  # es expplain deflaut
            'es.explain.sm': 'explain_document_sm',
            'es.explain.md': 'explain_document_md',
            'es.explain.lg': 'explain_document_lg',
            'es.ner': 'entity_recognizer_sm',  # es ner default
            'es.ner.sm': 'entity_recognizer_sm',
            'es.ner.md': 'entity_recognizer_md',
            'es.ner.lg': 'entity_recognizer_lg',
        },
        'xx': {
            'lang.' : 'detect_language_375' , # default lang classifer
            'lang.7': 'detect_language_7',  # multi lang detector alias
            'lang.20': 'detect_language_20',  # multi lang detector alias
            'xx.classify.lang': 'detect_language_20',  # multi lang detector default
            'xx.classify.lang.20': 'detect_language_20',  # multi lang detector default
            'xx.classify.lang.7': 'detect_language_7',

            # 2.7 lang classifiers

            'xx.classify.lang.' : 'detect_language_375' , # default lang classifer
            'xx.classify.lang.21' : 'detect_language_21' ,
            'xx.classify.lang.43' : 'detect_language_43' ,
            'xx.classify.lang.95' : 'detect_language_95' ,
            'xx.classify.lang.99' : 'detect_language_99' ,
            'xx.classify.lang.220' : 'detect_language_220' ,
            'xx.classify.lang.231' : 'detect_language_231' ,
            'xx.classify.lang.bigru' : 'detect_language_bigru_21' ,
            'lang.21' : 'detect_language_21' ,
            'lang.43' : 'detect_language_43' ,
            'lang.95' : 'detect_language_95' ,
            'lang.99' : 'detect_language_99' ,
            'lang.220' : 'detect_language_220' ,
            'lang.231' : 'detect_language_231' ,
            'lang.bigru' : 'detect_language_bigru_21' ,


            # 2.7 marian translate pipe references

            "xx.zlw.translate_to.en":"translate_zlw_en",
            "xx.en.translate_to.ti":"translate_en_ti",
            "xx.bem.translate_to.en":"translate_bem_en",
            "xx.ny.translate_to.en":"translate_ny_en",
            "xx.en.translate_to.lu":"translate_en_lu",
            "xx.taw.translate_to.en":"translate_taw_en",
            "xx.en.translate_to.ga":"translate_en_ga",
            "xx.en.translate_to.sw":"translate_en_sw",
            "xx.war.translate_to.en":"translate_war_en",
            "xx.en.translate_to.hu":"translate_en_hu",
            "xx.pqe.translate_to.en":"translate_pqe_en",
            "xx.en.translate_to.bem":"translate_en_bem",
            "xx.en.translate_to.tiv":"translate_en_tiv",
            "xx.en.translate_to.tll":"translate_en_tll",
            "xx.en.translate_to.cpp":"translate_en_cpp",
            "xx.efi.translate_to.en":"translate_efi_en",
            "xx.en.translate_to.itc":"translate_en_itc",
            "xx.uk.translate_to.en":"translate_uk_en",
            "xx.ee.translate_to.en":"translate_ee_en",
            "xx.nso.translate_to.en":"translate_nso_en",
            "xx.urj.translate_to.en":"translate_urj_en",
            "xx.sv.translate_to.en":"translate_sv_en",
            "xx.en.translate_to.rn":"translate_en_rn",
            "xx.nic.translate_to.en":"translate_nic_en",
            "xx.en.translate_to.bcl":"translate_en_bcl",
            "xx.en.translate_to.lg":"translate_en_lg",
            "xx.kwy.translate_to.en":"translate_kwy_en",
            "xx.en.translate_to.gmq":"translate_en_gmq",
            "xx.en.translate_to.ts":"translate_en_ts",
            "xx.bnt.translate_to.en":"translate_bnt_en",
            "xx.en.translate_to.pis":"translate_en_pis",
            "xx.kwn.translate_to.en":"translate_kwn_en",
            "xx.fi.translate_to.en":"translate_fi_en",
            "xx.en.translate_to.gaa":"translate_en_gaa",
            "xx.afa.translate_to.en":"translate_afa_en",
            "xx.itc.translate_to.en":"translate_itc_en",
            "xx.mh.translate_to.en":"translate_mh_en",
            "xx.en.translate_to.ln":"translate_en_ln",
            "xx.en.translate_to.zls":"translate_en_zls",
            "xx.en.translate_to.cy":"translate_en_cy",
            "xx.et.translate_to.en":"translate_et_en",
            "xx.en.translate_to.dra":"translate_en_dra",
            "xx.en.translate_to.sn":"translate_en_sn",
            "xx.lua.translate_to.en":"translate_lua_en",
            "xx.ln.translate_to.en":"translate_ln_en",
            "xx.ja.translate_to.en":"translate_ja_en",
            "xx.loz.translate_to.en":"translate_loz_en",
            "xx.en.translate_to.bi":"translate_en_bi",
            "xx.mg.translate_to.en":"translate_mg_en",
            "xx.vi.translate_to.en":"translate_vi_en",
            "xx.en.translate_to.vi":"translate_en_vi",
            "xx.hy.translate_to.en":"translate_hy_en",
            "xx.en.translate_to.mt":"translate_en_mt",
            "xx.ng.translate_to.en":"translate_ng_en",
            "xx.mkh.translate_to.en":"translate_mkh_en",
            "xx.en.translate_to.cpf":"translate_en_cpf",
            "xx.wal.translate_to.en":"translate_wal_en",
            "xx.en.translate_to.crs":"translate_en_crs",
            "xx.en.translate_to.zle":"translate_en_zle",
            "xx.en.translate_to.phi":"translate_en_phi",
            "xx.ine.translate_to.en":"translate_ine_en",
            "xx.en.translate_to.pap":"translate_en_pap",
            "xx.en.translate_to.sit":"translate_en_sit",
            "xx.bg.translate_to.en":"translate_bg_en",
            "xx.en.translate_to.ml":"translate_en_ml",
            "xx.en.translate_to.ss":"translate_en_ss",
            "xx.en.translate_to.tw":"translate_en_tw",
            "xx.en.translate_to.gv":"translate_en_gv",
            "xx.ca.translate_to.en":"translate_ca_en",
            "xx.umb.translate_to.en":"translate_umb_en",
            "xx.alv.translate_to.en":"translate_alv_en",
            "xx.gem.translate_to.en":"translate_gem_en",
            "xx.chk.translate_to.en":"translate_chk_en",
            "xx.kqn.translate_to.en":"translate_kqn_en",
            "xx.en.translate_to.afa":"translate_en_afa",
            "xx.gl.translate_to.en":"translate_gl_en",
            "xx.en.translate_to.ber":"translate_en_ber",
            "xx.en.translate_to.ig":"translate_en_ig",
            "xx.ase.translate_to.en":"translate_ase_en",
            "xx.en.translate_to.cs":"translate_en_cs",
            "xx.en.translate_to.pag":"translate_en_pag",
            "xx.en.translate_to.nic":"translate_en_nic",
            "xx.en.translate_to.hil":"translate_en_hil",
            "xx.en.translate_to.cel":"translate_en_cel",
            "xx.nl.translate_to.en":"translate_nl_en",
            "xx.en.translate_to.ho":"translate_en_ho",
            "xx.en.translate_to.inc":"translate_en_inc",
            "xx.ts.translate_to.en":"translate_ts_en",
            "xx.en.translate_to.tl":"translate_en_tl",
            "xx.ve.translate_to.en":"translate_ve_en",
            "xx.ceb.translate_to.en":"translate_ceb_en",
            "xx.en.translate_to.iir":"translate_en_iir",
            "xx.en.translate_to.aav":"translate_en_aav",
            "xx.en.translate_to.bat":"translate_en_bat",
            "xx.en.translate_to.alv":"translate_en_alv",
            "xx.ar.translate_to.en":"translate_ar_en",
            "xx.fiu.translate_to.en":"translate_fiu_en",
            "xx.en.translate_to.eu":"translate_en_eu",
            "xx.is.translate_to.en":"translate_is_en",
            "xx.wa.translate_to.en":"translate_wa_en",
            "xx.en.translate_to.tn":"translate_en_tn",
            "xx.ig.translate_to.en":"translate_ig_en",
            "xx.luo.translate_to.en":"translate_luo_en",
            "xx.en.translate_to.kwn":"translate_en_kwn",
            "xx.niu.translate_to.en":"translate_niu_en",
            "xx.en.translate_to.gl":"translate_en_gl",
            "xx.en.translate_to.ilo":"translate_en_ilo",
            "xx.en.translate_to.ur":"translate_en_ur",
            "xx.cus.translate_to.en":"translate_cus_en",
            "xx.phi.translate_to.en":"translate_phi_en",
            "xx.en.translate_to.loz":"translate_en_loz",
            "xx.tiv.translate_to.en":"translate_tiv_en",
            "xx.en.translate_to.id":"translate_en_id",
            "xx.zle.translate_to.en":"translate_zle_en",
            "xx.en.translate_to.mfe":"translate_en_mfe",
            "xx.id.translate_to.en":"translate_id_en",
            "xx.lv.translate_to.en":"translate_lv_en",
            "xx.en.translate_to.pon":"translate_en_pon",
            "xx.en.translate_to.sq":"translate_en_sq",
            "xx.tum.translate_to.en":"translate_tum_en",
            "xx.pl.translate_to.en":"translate_pl_en",
            "xx.xh.translate_to.en":"translate_xh_en",
            "xx.kab.translate_to.en":"translate_kab_en",
            "xx.tvl.translate_to.en":"translate_tvl_en",
            "xx.pa.translate_to.en":"translate_pa_en",
            "xx.iso.translate_to.en":"translate_iso_en",
            "xx.ho.translate_to.en":"translate_ho_en",
            "xx.cel.translate_to.en":"translate_cel_en",
            "xx.en.translate_to.om":"translate_en_om",
            "xx.kg.translate_to.en":"translate_kg_en",
            "xx.en.translate_to.lus":"translate_en_lus",
            "xx.om.translate_to.en":"translate_om_en",
            "xx.lun.translate_to.en":"translate_lun_en",
            "xx.crs.translate_to.en":"translate_crs_en",
            "xx.cy.translate_to.en":"translate_cy_en",
            "xx.tll.translate_to.en":"translate_tll_en",
            "xx.gil.translate_to.en":"translate_gil_en",
            "xx.en.translate_to.mkh":"translate_en_mkh",
            "xx.en.translate_to.euq":"translate_en_euq",
            "xx.en.translate_to.sem":"translate_en_sem",
            "xx.cs.translate_to.en":"translate_cs_en",
            "xx.en.translate_to.sk":"translate_en_sk",
            "xx.en.translate_to.bzs":"translate_en_bzs",
            "xx.en.translate_to.trk":"translate_en_trk",
            "xx.cpf.translate_to.en":"translate_cpf_en",
            "xx.bi.translate_to.en":"translate_bi_en",
            "xx.en.translate_to.mul":"translate_en_mul",
            "xx.en.translate_to.gmw":"translate_en_gmw",
            "xx.en.translate_to.fi":"translate_en_fi",
            "xx.en.translate_to.zlw":"translate_en_zlw",
            "xx.lg.translate_to.en":"translate_lg_en",
            "xx.en.translate_to.pqe":"translate_en_pqe",
            "xx.en.translate_to.xh":"translate_en_xh",
            "xx.en.translate_to.hi":"translate_en_hi",
            "xx.en.translate_to.nyk":"translate_en_nyk",
            "xx.th.translate_to.en":"translate_th_en",
            "xx.en.translate_to.umb":"translate_en_umb",
            "xx.en.translate_to.af":"translate_en_af",
            "xx.tpi.translate_to.en":"translate_tpi_en",
            "xx.ti.translate_to.en":"translate_ti_en",
            "xx.en.translate_to.chk":"translate_en_chk",
            "xx.mos.translate_to.en":"translate_mos_en",
            "xx.en.translate_to.sm":"translate_en_sm",
            "xx.pon.translate_to.en":"translate_pon_en",
            "xx.en.translate_to.bg":"translate_en_bg",
            "xx.en.translate_to.ny":"translate_en_ny",
            "xx.kl.translate_to.en":"translate_kl_en",
            "xx.en.translate_to.hy":"translate_en_hy",
            "xx.nyk.translate_to.en":"translate_nyk_en",
            "xx.it.translate_to.en":"translate_it_en",
            "xx.mt.translate_to.en":"translate_mt_en",
            "xx.pap.translate_to.en":"translate_pap_en",
            "xx.srn.translate_to.en":"translate_srn_en",
            "xx.da.translate_to.en":"translate_da_en",
            "xx.en.translate_to.lue":"translate_en_lue",
            "xx.rn.translate_to.en":"translate_rn_en",
            "xx.en.translate_to.tut":"translate_en_tut",
            "xx.lu.translate_to.en":"translate_lu_en",
            "xx.ru.translate_to.en":"translate_ru_en",
            "xx.en.translate_to.toi":"translate_en_toi",
            "xx.ccs.translate_to.en":"translate_ccs_en",
            "xx.aav.translate_to.en":"translate_aav_en",
            "xx.en.translate_to.ha":"translate_en_ha",
            "xx.rnd.translate_to.en":"translate_rnd_en",
            "xx.de.translate_to.en":"translate_de_en",
            "xx.en.translate_to.luo":"translate_en_luo",
            "xx.fr.translate_to.en":"translate_fr_en",
            "xx.bcl.translate_to.en":"translate_bcl_en",
            "xx.ilo.translate_to.en":"translate_ilo_en",
            "xx.en.translate_to.jap":"translate_en_jap",
            "xx.en.translate_to.fj":"translate_en_fj",
            "xx.sk.translate_to.en":"translate_sk_en",
            "xx.bzs.translate_to.en":"translate_bzs_en",
            "xx.ka.translate_to.en":"translate_ka_en",
            "xx.ko.translate_to.en":"translate_ko_en",
            "xx.sq.translate_to.en":"translate_sq_en",
            "xx.mul.translate_to.en":"translate_mul_en",
            "xx.en.translate_to.run":"translate_en_run",
            "xx.sn.translate_to.en":"translate_sn_en",
            "xx.en.translate_to.pqw":"translate_en_pqw",
            "xx.ss.translate_to.en":"translate_ss_en",
            "xx.sm.translate_to.en":"translate_sm_en",
            "xx.en.translate_to.kwy":"translate_en_kwy",
            "xx.jap.translate_to.en":"translate_jap_en",
            "xx.en.translate_to.kqn":"translate_en_kqn",
            "xx.mk.translate_to.en":"translate_mk_en",
            "xx.hu.translate_to.en":"translate_hu_en",
            "xx.en.translate_to.map":"translate_en_map",
            "xx.yo.translate_to.en":"translate_yo_en",
            "xx.hi.translate_to.en":"translate_hi_en",
            "xx.iir.translate_to.en":"translate_iir_en",
            "xx.en.translate_to.guw":"translate_en_guw",
            "xx.en.translate_to.es":"translate_en_es",
            "xx.en.translate_to.gem":"translate_en_gem",
            "xx.en.translate_to.ht":"translate_en_ht",
            "xx.zls.translate_to.en":"translate_zls_en",
            "xx.sg.translate_to.en":"translate_sg_en",
            "xx.en.translate_to.ty":"translate_en_ty",
            "xx.en.translate_to.lun":"translate_en_lun",
            "xx.guw.translate_to.en":"translate_guw_en",
            "xx.trk.translate_to.en":"translate_trk_en",
            "xx.mfe.translate_to.en":"translate_mfe_en",
            "xx.en.translate_to.nl":"translate_en_nl",
            "xx.en.translate_to.sv":"translate_en_sv",
            "xx.ber.translate_to.en":"translate_ber_en",
            "xx.to.translate_to.en":"translate_to_en",
            "xx.en.translate_to.da":"translate_en_da",
            "xx.en.translate_to.urj":"translate_en_urj",
            "xx.inc.translate_to.en":"translate_inc_en",
            "xx.wls.translate_to.en":"translate_wls_en",
            "xx.pis.translate_to.en":"translate_pis_en",
            "xx.en.translate_to.mh":"translate_en_mh",
            "xx.en.translate_to.iso":"translate_en_iso",
            "xx.en.translate_to.ru":"translate_en_ru",
            "xx.swc.translate_to.en":"translate_swc_en",
            "xx.en.translate_to.rnd":"translate_en_rnd",
            "xx.en.translate_to.nso":"translate_en_nso",
            "xx.en.translate_to.swc":"translate_en_swc",
            "xx.ur.translate_to.en":"translate_ur_en",
            "xx.en.translate_to.ro":"translate_en_ro",
            "xx.ml.translate_to.en":"translate_ml_en",
            "xx.grk.translate_to.en":"translate_grk_en",
            "xx.rw.translate_to.en":"translate_rw_en",
            "xx.tr.translate_to.en":"translate_tr_en",
            "xx.gmq.translate_to.en":"translate_gmq_en",
            "xx.euq.translate_to.en":"translate_euq_en",
            "xx.en.translate_to.tdt":"translate_en_tdt",
            "xx.eo.translate_to.en":"translate_eo_en",
            "xx.cau.translate_to.en":"translate_cau_en",
            "xx.en.translate_to.mk":"translate_en_mk",
            "xx.en.translate_to.mr":"translate_en_mr",
            "xx.af.translate_to.en":"translate_af_en",
            "xx.run.translate_to.en":"translate_run_en",
            "xx.en.translate_to.ng":"translate_en_ng",
            "xx.en.translate_to.mg":"translate_en_mg",
            "xx.en.translate_to.bnt":"translate_en_bnt",
            "xx.en.translate_to.kj":"translate_en_kj",
            "xx.en.translate_to.he":"translate_en_he",
            "xx.en.translate_to.sla":"translate_en_sla",
            "xx.en.translate_to.el":"translate_en_el",
            "xx.ht.translate_to.en":"translate_ht_en",
            "xx.en.translate_to.et":"translate_en_et",
            "xx.en.translate_to.poz":"translate_en_poz",
            "xx.roa.translate_to.en":"translate_roa_en",
            "xx.en.translate_to.de":"translate_en_de",
            "xx.fj.translate_to.en":"translate_fj_en",
            "xx.en.translate_to.lua":"translate_en_lua",
            "xx.en.translate_to.kg":"translate_en_kg",
            "xx.en.translate_to.fiu":"translate_en_fiu",
            "xx.gv.translate_to.en":"translate_gv_en",
            "xx.cpp.translate_to.en":"translate_cpp_en",
            "xx.en.translate_to.tpi":"translate_en_tpi",
            "xx.en.translate_to.grk":"translate_en_grk",
            "xx.en.translate_to.sal":"translate_en_sal",
            "xx.en.translate_to.niu":"translate_en_niu",
            "xx.en.translate_to.ca":"translate_en_ca",
            "xx.en.translate_to.roa":"translate_en_roa",
            "xx.sal.translate_to.en":"translate_sal_en",
            "xx.ha.translate_to.en":"translate_ha_en",
            "xx.sem.translate_to.en":"translate_sem_en",
            "xx.tn.translate_to.en":"translate_tn_en",
            "xx.gaa.translate_to.en":"translate_gaa_en",
            "xx.en.translate_to.to":"translate_en_to",
            "xx.en.translate_to.ee":"translate_en_ee",
            "xx.toi.translate_to.en":"translate_toi_en",
            "xx.lue.translate_to.en":"translate_lue_en",
            "xx.en.translate_to.rw":"translate_en_rw",
            "xx.st.translate_to.en":"translate_st_en",
            "xx.dra.translate_to.en":"translate_dra_en",
            "xx.en.translate_to.mos":"translate_en_mos",
            "xx.eu.translate_to.en":"translate_eu_en",
            "xx.lus.translate_to.en":"translate_lus_en",
            "xx.sla.translate_to.en":"translate_sla_en",
            "xx.en.translate_to.ceb":"translate_en_ceb",
            "xx.art.translate_to.en":"translate_art_en",
            "xx.bat.translate_to.en":"translate_bat_en",
            "xx.az.translate_to.en":"translate_az_en",
            "xx.en.translate_to.ine":"translate_en_ine",
            "xx.pag.translate_to.en":"translate_pag_en",
            "xx.yap.translate_to.en":"translate_yap_en",
            "xx.en.translate_to.eo":"translate_en_eo",
            "xx.en.translate_to.tvl":"translate_en_tvl",
            "xx.kj.translate_to.en":"translate_kj_en",
            "xx.en.translate_to.st":"translate_en_st",
            "xx.gmw.translate_to.en":"translate_gmw_en",
            "xx.mr.translate_to.en":"translate_mr_en",
            "xx.es.translate_to.en":"translate_es_en",
            "xx.en.translate_to.sg":"translate_en_sg",
            "xx.en.translate_to.cus":"translate_en_cus",
            "xx.en.translate_to.it":"translate_en_it",
            "xx.ga.translate_to.en":"translate_ga_en",
            "xx.bn.translate_to.en":"translate_bn_en",
            "xx.en.translate_to.efi":"translate_en_efi",
            "xx.en.translate_to.az":"translate_en_az",
            "xx.en.translate_to.zh":"translate_en_zh",
            "xx.en.translate_to.is":"translate_en_is",
            "xx.zh.translate_to.en":"translate_zh_en",
            "xx.hil.translate_to.en":"translate_hil_en",
            "xx.en.translate_to.ar":"translate_en_ar",
            "xx.tl.translate_to.en":"translate_tl_en",
            "xx.en.translate_to.gil":"translate_en_gil",
            "xx.en.translate_to.uk":"translate_en_uk",
            "xx.en.translate_to.fr":"translate_en_fr",


        },

    }
    pretrained_models_references = {
        'nl': {
            'nl.lemma': 'lemma',  # default lemma, dataset unknown
            'nl.pos': 'pos_ud_alpino',  # default pos nl
            'nl.pos.ud_alpino': 'pos_ud_alpino',
            'nl.ner': 'wikiner_6B_100',  # default ner nl
            'nl.ner.wikiner': 'wikiner_6B_100',  # default ner nl with embeds
            'nl.ner.wikiner.glove.6B_100': 'wikiner_6B_100',
            'nl.ner.wikiner.glove.6B_300': 'wikiner_6B_300',
            'nl.ner.wikiner.glove.840B_300': 'wikiner_840B_300',

        },
        'en': {
            # models
            'en.stem': 'stemmer',  # stem  default en
            'en.tokenize': 'spark_nlp_tokenizer',  # token default en
            'en.norm': 'norm',  #  norm default en
            'en.chunk': 'default_chunker',  #  default chunker  en
            'en.ngram': 'ngram',  #  default chunker  en
            'en.embed_chunk': 'chunk_embeddings',  #  default chunker  en

            
            'en.lemma': 'lemma_antbnc',  # lemma default en
            'en.lemma.antbnc': 'lemma_antbnc',
            'en.pos': 'pos_anc',  # pos default en
            'en.pos.anc': 'pos_anc',
            'en.pos.ud_ewt': 'pos_ud_ewt',
            # 'en.ner.crf' :'ner_crf', # crf not supported in NLU
            'en.ner': 'ner_dl',  # ner default en
            'en.ner.dl': 'ner_dl',  # ner embeds  default  en
            'en.ner.dl.glove.6B_100d': 'ner_dl',
            'en.ner.dl.bert': 'ner_dl_bert',  # points ner bert
            'en.ner.onto': 'onto_100',  # ner  onto default embeds en
            'en.ner.onto.glove.6B_100d': 'onto_100',
            'en.ner.onto.glove.840B_300d': 'onto_300',  # this uses multi lang embeds!

            #2.7
            'en.ner.onto.bert.cased_base' :'onto_bert_base_cased',
            'en.ner.onto.bert.cased_large' :'onto_bert_large_cased',
            'en.ner.onto.electra.uncased_large'                 :'onto_electra_large_uncased',
            'en.ner.onto.bert.small_l2_128'    :'onto_small_bert_L2_128',
            'en.ner.onto.bert.small_l4_256'    :'onto_small_bert_L4_256',
            'en.ner.onto.bert.small_l4_512'    :'onto_small_bert_L4_512',
            'en.ner.onto.bert.small_l8_512'    :'onto_small_bert_L8_512',
            'en.ner.onto.electra.uncased_small'                 :'onto_electra_small_uncased',
            'en.ner.onto.electra.uncased_base'                 :'onto_electra_base_uncased',
            'en.ner.bert_base_cased' : 'ner_dl_bert_base_cased' ,

            'en.ner.ade' : 'ade_ner_100d' ,
            'en.ner.aspect_sentiment' : 'ner_aspect_based_sentiment' ,

            'en.ner.glove.100d': 'ner_dl_sentence',
            'en.spell.symmetric': 'spellcheck_sd',
            'en.spell.norvig': 'spellcheck_norvig',
            'en.sentiment.vivekn': 'sentiment_vivekn',
            'en.dep.untyped.conllu': 'dependency_conllu',
            'en.dep.untyped': 'dependency_conllu',  # default untyped dependency
            'en.dep': 'dependency_typed_conllu',  # default typed dependency
            'en.dep.typed': 'dependency_typed_conllu',  # default typed dependency dataset
            'en.dep.typed.conllu': 'dependency_typed_conllu',
            'en.stopwords': 'stopwords_en',

            # embeddings
            'en.glove': 'glove_100d',  # default embed
            'en.embed': 'glove_100d',  # default glove en
            'en.embed.glove': 'glove_100d',  # default glove en
            'en.embed.glove.100d': 'glove_100d',
            'en.bert': 'bert_base_uncased',  # default bert
            'en.embed.bert': 'bert_base_uncased',  # default bert
            'en.embed.bert.base_uncased': 'bert_base_uncased',
            'en.embed.bert.base_cased': 'bert_base_cased',
            'en.embed.bert.large_uncased': 'bert_large_uncased',
            'en.embed.bert.large_cased': 'bert_large_cased',
            'biobert': 'biobert_pubmed_base_cased',  # alias
            'en.embed.biobert': 'biobert_pubmed_base_cased',  # default bio bert
            'en.embed.biobert.pubmed_base_cased': 'biobert_pubmed_base_cased',
            'en.embed.biobert.pubmed_large_cased': 'biobert_pubmed_large_cased',
            'en.embed.biobert.pmc_base_cased': 'biobert_pmc_base_cased',
            'en.embed.biobert.pubmed_pmc_base_cased': 'biobert_pubmed_pmc_base_cased',
            'en.embed.biobert.clinical_base_cased': 'biobert_clinical_base_cased',
            'en.embed.biobert.discharge_base_cased': 'biobert_discharge_base_cased',
            'en.embed.elmo': 'elmo',
            'en.embed_sentence': 'tfhub_use',  # default sentence

            'en.embed_sentence.use': 'tfhub_use',  # default use
            'en.use': 'tfhub_use',  # alias
            'en.embed.use': 'tfhub_use',  # alias
            'en.embed_sentence.tfhub_use': 'tfhub_use',
            'en.embed_sentence.use.lg': 'tfhub_use_lg',  # alias
            'en.embed_sentence.tfhub_use.lg': 'tfhub_use_lg',


            'en.embed_sentence.albert': 'albert_base_uncased',  # albert default en


            'en.albert': 'albert_base_uncased',  # albert alias en
            'en.embed.albert': 'albert_base_uncased',  # albert alias en
            'en.embed.albert.base_uncased': 'albert_base_uncased',
            'en.embed.albert.large_uncased': 'albert_large_uncased',
            'en.embed.albert.xlarge_uncased': 'albert_xlarge_uncased',
            'en.embed.albert.xxlarge_uncased': 'albert_xxlarge_uncased',
            'en.embed.xlnet': 'xlnet_base_cased',  # xlnet default en
            'en.xlnet': 'xlnet_base_cased',  # xlnet alias
            'en.embed.xlnet_base_cased': 'xlnet_base_cased',
            'en.embed.xlnet_large_cased': 'xlnet_large_cased',

            # classifiers and sentiment

            'en.classify.trec6.use': 'classifierdl_use_trec6',
            'en.classify.trec50.use': 'classifierdl_use_trec50',
            'en.classify.question': 'classifierdl_use_trec50',
            'en.classify.questions': 'classifierdl_use_trec50',

            'en.classify.spam.use': 'classifierdl_use_spam',
            'en.classify.fakenews.use': 'classifierdl_use_fakenews',
            'en.classify.emotion.use': 'classifierdl_use_emotion',
            'en.classify.cyberbullying.use': 'classifierdl_use_cyberbullying',
            'en.classify.sarcasm.use': 'classifierdl_use_sarcasm',
            'en.sentiment.imdb.use': 'sentimentdl_use_imdb',
            'en.sentiment.twitter.use': 'sentimentdl_use_twitter',
            'en.sentiment.imdb.glove': 'sentimentdl_glove_imdb',
            'en.classify.trec6': 'classifierdl_use_trec6',  # Alias withouth embedding
            'en.classify.trec50': 'classifierdl_use_trec50',  # Alias withouth embedding
            'en.classify.spam': 'classifierdl_use_spam',  # Alias withouth embedding
            'en.classify.fakenews': 'classifierdl_use_fakenews',  # Alias withouth embedding
            'en.classify.emotion': 'classifierdl_use_emotion',  # Alias withouth embedding
            'en.classify.cyberbullying': 'classifierdl_use_cyberbullying',  # Alias withouth embedding
            'en.classify.sarcasm': 'classifierdl_use_sarcasm',  # Alias withouth embedding
            'en.sentiment.twitter': 'sentimentdl_use_twitter',  # Alias withouth embedding
            'en.sentiment.imdb': 'sentimentdl_glove_imdb',  # Default sentiment imdb with embeddigns glvoe
        
            #2.6 Release models
            'en.yake' :'yake',

            #2.6  embeds
            'en.embed.electra': 'electra_small_uncased',
            'en.embed.electra.small_uncased': 'electra_small_uncased',
            'en.embed.electra.base_uncased': 'electra_base_uncased',
            'en.embed.electra.large_uncased': 'electra_large_uncased',


            'en.embed.covidbert': 'covidbert_large_uncased',
            'en.embed.covidbert.large_uncased': 'covidbert_large_uncased',
            'en.embed.bert.small_L2_128': 'small_bert_L2_128',
            'en.embed.bert.small_L4_128': 'small_bert_L4_128',
            'en.embed.bert.small_L6_128': 'small_bert_L6_128',
            'en.embed.bert.small_L8_128': 'small_bert_L8_128',
            'en.embed.bert.small_L10_128': 'small_bert_L10_128',
            'en.embed.bert.small_L12_128': 'small_bert_L12_128',
            'en.embed.bert.small_L2_256': 'small_bert_L2_256',
            'en.embed.bert.small_L4_256': 'small_bert_L4_256',
            'en.embed.bert.small_L6_256': 'small_bert_L6_256',
            'en.embed.bert.small_L8_256': 'small_bert_L8_256',
            'en.embed.bert.small_L10_256': 'small_bert_L10_256',
            'en.embed.bert.small_L12_256': 'small_bert_L12_256',
            'en.embed.bert.small_L2_512': 'small_bert_L2_512',
            'en.embed.bert.small_L4_512': 'small_bert_L4_512',
            'en.embed.bert.small_L6_512': 'small_bert_L6_512',
            'en.embed.bert.small_L8_512': 'small_bert_L8_512',
            'en.embed.bert.small_L10_512': 'small_bert_L10_512',
            'en.embed.bert.small_L12_512': 'small_bert_L12_512',
            'en.embed.bert.small_L2_768': 'small_bert_L2_768',
            'en.embed.bert.small_L4_768': 'small_bert_L4_768',
            'en.embed.bert.small_L6_768': 'small_bert_L6_768',
            'en.embed.bert.small_L8_768': 'small_bert_L8_768',
            'en.embed.bert.small_L10_768': 'small_bert_L10_768',
            'en.embed.bert.small_L12_768': 'small_bert_L12_768',
            
            #2.6 sent embeddings
            'en.embed_sentence.electra': 'sent_electra_small_uncased',

            'en.embed_sentence.electra_small_uncased': 'sent_electra_small_uncased',
            'en.embed_sentence.electra_base_uncased': 'sent_electra_base_uncased',
            'en.embed_sentence.electra_large_uncased': 'sent_electra_large_uncased',
            'en.embed_sentence.bert': 'sent_bert_base_uncased',

            'en.embed_sentence.bert_base_uncased': 'sent_bert_base_uncased',
            'en.embed_sentence.bert_base_cased': 'sent_bert_base_cased',
            'en.embed_sentence.bert_large_uncased': 'sent_bert_large_uncased',
            'en.embed_sentence.bert_large_cased': 'sent_bert_large_cased',
            'en.embed_sentence.biobert.pubmed_base_cased': 'sent_biobert_pubmed_base_cased',
            'en.embed_sentence.biobert.pubmed_large_cased': 'sent_biobert_pubmed_large_cased',
            'en.embed_sentence.biobert.pmc_base_cased': 'sent_biobert_pmc_base_cased',
            'en.embed_sentence.biobert.pubmed_pmc_base_cased': 'sent_biobert_pubmed_pmc_base_cased',
            'en.embed_sentence.biobert.clinical_base_cased': 'sent_biobert_clinical_base_cased',
            'en.embed_sentence.biobert.discharge_base_cased': 'sent_biobert_discharge_base_cased',
            'en.embed_sentence.covidbert.large_uncased': 'sent_covidbert_large_uncased',
            'en.embed_sentence.small_bert_L2_128': 'sent_small_bert_L2_128',
            'en.embed_sentence.small_bert_L4_128': 'sent_small_bert_L4_128',
            'en.embed_sentence.small_bert_L6_128': 'sent_small_bert_L6_128',
            'en.embed_sentence.small_bert_L8_128': 'sent_small_bert_L8_128',
            'en.embed_sentence.small_bert_L10_128': 'sent_small_bert_L10_128',
            'en.embed_sentence.small_bert_L12_128': 'sent_small_bert_L12_128',
            'en.embed_sentence.small_bert_L2_256': 'sent_small_bert_L2_256',
            'en.embed_sentence.small_bert_L4_256': 'sent_small_bert_L4_256',
            'en.embed_sentence.small_bert_L6_256': 'sent_small_bert_L6_256',
            'en.embed_sentence.small_bert_L8_256': 'sent_small_bert_L8_256',
            'en.embed_sentence.small_bert_L10_256': 'sent_small_bert_L10_256',
            'en.embed_sentence.small_bert_L12_256': 'sent_small_bert_L12_256',
            'en.embed_sentence.small_bert_L2_512': 'sent_small_bert_L2_512',
            'en.embed_sentence.small_bert_L4_512': 'sent_small_bert_L4_512',
            'en.embed_sentence.small_bert_L6_512': 'sent_small_bert_L6_512',
            'en.embed_sentence.small_bert_L8_512': 'sent_small_bert_L8_512',
            'en.embed_sentence.small_bert_L10_512': 'sent_small_bert_L10_512',
            'en.embed_sentence.small_bert_L12_512': 'sent_small_bert_L12_512',
            'en.embed_sentence.small_bert_L2_768': 'sent_small_bert_L2_768',
            'en.embed_sentence.small_bert_L4_768': 'sent_small_bert_L4_768',
            'en.embed_sentence.small_bert_L6_768': 'sent_small_bert_L6_768',
            'en.embed_sentence.small_bert_L8_768': 'sent_small_bert_L8_768',
            'en.embed_sentence.small_bert_L10_768': 'sent_small_bert_L10_768',
            'en.embed_sentence.small_bert_L12_768': 'sent_small_bert_L12_768',
            
            # 2.6 classifiers
            'en.classify.toxic': 'multiclassifierdl_use_toxic',
            'en.toxic': 'multiclassifierdl_use_toxic',

            'en.e2e': 'multiclassifierdl_use_e2e',

            'en.classify.toxic.sm': 'multiclassifierdl_use_toxic_sm',
            'en.classify.e2e': 'multiclassifierdl_use_e2e',
            
            # 2.7 seq2seq
            'en.t5' : 'google_t5_small_ssm_nq',
            'en.t5.small' : 't5_small',
            'en.t5.base' : 't5_base',
            # 2.7.0 new aliases
            'en.t5.summarize': 't5_base',
            'en.t5.classify.grammar_correctness': 't5_base',
            'en.t5.classify.sentiment': 't5_base',
            'en.t5.answer_question': 't5_base',


            # 2.7,1 and 2.7.2 ATIS classifier and ALIASES
            "en.classify.questions.atis":"classifierdl_use_atis",
            "en.classify.questions.airline":"classifierdl_use_atis",
            "en.classify.intent.atis":"classifierdl_use_atis",
            "en.classify.intent.airline":"classifierdl_use_atis",

            # 2.7,1 and 2.7.2 ATIS NER and ALIASES
            "en.ner.atis":"nerdl_atis_840b_300d",
            "en.ner.airline":"nerdl_atis_840b_300d",
            "en.ner.aspect.airline":"nerdl_atis_840b_300d",
            "en.ner.aspect.atis":"nerdl_atis_840b_300d",



        },
        'fr': {
            'fr.lemma': 'lemma',
            'fr.pos': 'pos_ud_gsd',  # default pos fr
            'fr.pos.ud_gsd': 'pos_ud_gsd',
            'fr.ner': 'wikiner_840B_300',  # default ner fr
            'fr.ner.wikiner': 'wikiner_840B_300',  # default nr embeds fr
            'fr.ner.wikiner.glove.840B_300': 'wikiner_840B_300',
            'fr.stopwords': 'stopwords_fr',
            'fr.ner.wikiner.glove.6B_300': 'wikiner_6B_300',

        },
        'de': {
            'de.lemma': 'lemma',
            'de.pos.ud_hdt': 'pos_ud_hdt',
            'de.pos': 'pos_ud_hdt',  # default pos de
            'de.ner': 'wikiner_840B_300',  # default ner de
            'de.ner.wikiner': 'wikiner_840B_300',  # default ner embeds de
            'de.ner.wikiner.glove.840B_300': 'wikiner_840B_300',
            'de.stopwords': 'stopwords_de',
            'de.ner.wikiner.glove.6B_300': 'wikiner_6B_300',

        },
        'it': {
            'it.lemma': 'lemma_dxc',  # default lemma it
            'it.lemma.dxc': 'lemma_dxc',
            'it.sentiment.dxc': 'sentiment_dxc',
            'it.sentiment': 'sentiment_dxc',  # defauult sentiment it
            'it.pos': 'pos_ud_isdt',  # default pos it
            'it.pos.ud_isdt': 'pos_ud_isdt',
            'it.ner': 'wikiner_840B_300',  # default ner it
            'it.ner.wikiner.glove.6B_300': 'wikiner_6B_300',
            'it.stopwords': 'stopwords_it',


        },
        'nb': {
            'nb.lemma': 'lemma',
            'nb.pos.ud_bokmaal': 'pos_ud_bokmaal',

        },
        'no': {
            'no.ner': 'norne_6B_100',  # ner default no
            'no.ner.norne': 'norne_6B_100',  # ner default no embeds
            'no.ner.norne.glove.6B_100': 'norne_6B_100',
            'no.ner.norne.glove.6B_300': 'norne_6B_300',
            'no.ner.norne.glove.840B_300': 'norne_840B_300',

        },
        'nn': {
            'nn.pos': 'pos_ud_nynorsk',  # nn default pos
            'nn.pos.ud_nynorsk': 'pos_ud_nynorsk',

        },
        'pl': {
            'pl.lemma': 'lemma',
            'pl.pos': 'pos_ud_lfg',  # pls default pos
            'pl.pos.ud_lfg': 'pos_ud_lfg',
            'pl.ner': 'wikiner_6B_100',  # pl default ner
            'pl.ner.wikiner': 'wikiner_6B_100',  # pls default ner embeds
            'pl.ner.wikiner.glove.6B_100': 'wikiner_6B_100',
            'pl.ner.wikiner.glove.6B_300': 'wikiner_6B_300',
            'pl.ner.wikiner.glove.840B_300': 'wikiner_840B_300',
            'pl.stopwords': 'stopwords_pl'
        },
        'pt': {
            'pt.lemma': 'lemma',
            'pt.pos.ud_bosque': 'pos_ud_bosque',
            'pt.pos': 'pos_ud_bosque',  # pt default pos
            'pt.ner': 'wikiner_6B_100',  # pt default ner
            'pt.ner.wikiner.glove.6B_100': 'wikiner_6B_100',  # pt default embeds ner
            'pt.ner.wikiner.glove.6B_300': 'wikiner_6B_300',
            'pt.ner.wikiner.glove.840B_300': 'wikiner_840B_300',
            'pt.stopwords': 'stopwords_pt',
            'pt.bert': 'bert_portuguese_base_cased',
            'pt.bert.cased': 'bert_portuguese_base_cased',
            'pt.bert.cased.large':'bert_portuguese_large_cased',

        },
        'ru': {
            'ru.lemma': 'lemma',
            'ru.pos.ud_gsd': 'pos_ud_gsd',
            'ru.pos': 'pos_ud_gsd',  # pos default ru
            'ru.ner': 'wikiner_6B_100',  # ner default ru
            'ru.ner.wikiner': 'wikiner_6B_100',  # ner embeds default ru
            'ru.ner.wikiner.glove.6B_100': 'wikiner_6B_100',
            'ru.ner.wikiner.glove.6B_300': 'wikiner_6B_300',
            'ru.ner.wikiner.glove.840B_300': 'wikiner_840B_300',
            'ru.stopwords': 'stopwords_ru',

        },
        'es': {
            'es.lemma': 'lemma',
            'es.pos': 'pos_ud_gsd',  # pos default es
            'es.pos.ud_gsd': 'pos_ud_gsd',
            'es.ner': 'wikiner_6B_100',  # ner default es
            'es.ner.wikiner': 'wikiner_6B_100',  # ner default embeds es
            'es.ner.wikiner.glove.6B_100': 'wikiner_6B_100',
            'es.ner.wikiner.glove.6B_300': 'wikiner_6B_300',
            'es.ner.wikiner.glove.840B_300': 'wikiner_840B_300',
            'es.stopwords_es': 'stopwords_es',
        },
        'af': {
            'af.stopwords': 'stopwords_af'

        },
        'ar': {
            'ar.stopwords': 'stopwords_ar',
            'ar.lemma' : 'lemma',
            'ar.pos' : 'pos_ud_padt',
            'ar.embed' : 'arabic_w2v_cc_300d' ,
            'ar.embed.cbow' : 'arabic_w2v_cc_300d' ,
            'ar.embed.cbow.300d' : 'arabic_w2v_cc_300d' ,
            "ar.embed.aner":"arabic_w2v_cc_300d",
            "ar.embed.aner.300d":"arabic_w2v_cc_300d",
            "ar.embed.glove":"arabic_w2v_cc_300d",

            "ar.ner" :"aner_cc_300d",
            "ar.ner.aner" :"aner_cc_300d",

        },
        'hy': {
            'hy.stopwords': 'stopwords_hy',
            'hy.lemma': 'lemma',
            'hy.pos': 'pos_ud_armtdp',


        },
        'eu': {
            'eu.stopwords': 'stopwords_eu',
            'eu.lemma': 'lemma',
            'eu.pos': 'pos_ud_bdt',

        },
        'bn': {
            'bn.stopwords': 'stopwords_bn',
            "bn.lemma" :"lemma",
            "bn.pos": "pos_msri",
            'bn.ner':'ner_jifs_glove_840B_300d',
            'bn.ner.glove':'ner_jifs_glove_840B_300d',

    },
        'br': {
            'br.stopwords': 'stopwords_br',
            'br.lemma': 'lemma',
            'br.pos': 'pos_ud_keb',

        },
        'bg': {
            'bg.lemma': 'lemma',
            'bg.pos': 'pos_ud_btb',  # default bg pos
            'bg.pos.ud_btb': 'pos_ud_btb',
            'bg.stopwords': 'stopwords_bg',

        },
        'ca': {
            'ca.stopwords': 'stopwords_ca',
            'ca.lemma': 'lemma',
            'ca.pos': 'pos_ud_ancora',

        },
        'cs': {
            'cs.lemma': 'lemma',
            'cs.pos': 'pos_ud_pdt',  # default cs pos
            'cs.pos.ud_pdt': 'pos_ud_pdt',
            'cs.stopwords': 'stopwords_cs',
        },
        'eo': {
            'eo.stopwords': 'stopwords_eo'

        },
        'fi': {
            'fi.lemma': 'lemma',
            'fi.pos.ud_tdt': 'pos_ud_tdt',
            'fi.pos': 'pos_ud_tdt',  # default pos fi
            'fi.stopwords': 'stopwords_fi',
            'fi.ner': 'wikiner_6B_100',
            'fi.ner.6B_100': 'wikiner_6B_100',
            'fi.ner.6B_300': 'wikiner_6B_300',
            'fi.ner.840B_300': 'wikiner_840B_300',
            'fi.embed.bert.': 'bert_finnish_cased',
            'fi.embed.bert.cased.': 'bert_finnish_cased',
            'fi.embed.bert.uncased.': 'bert_finnish_uncased',
            'fi.embed_sentence': 'sent_bert_finnish_cased',
            'fi.embed_sentence.bert.cased': 'sent_bert_finnish_cased',
            'fi.embed_sentence.bert.uncased': 'sent_bert_finnish_uncased',
            'fi.ner.6B_100d' : 'finnish_ner_6B_100' ,
            'fi.ner.6B_300d' : 'finnish_ner_6B_300' ,
            'fi.ner.840B_300d' : 'finnish_ner_840B_300' ,


        },
        'gl': {
            'gl.stopwords': 'stopwords_gl',
            'gl.lemma': 'lemma',
            'gl.pos': 'pos_ud_treegal',


        },
        'el': {
            'el.lemma': 'lemma',
            'el.pos': 'pos_ud_gdt',  # default POS  el
            'el.pos.ud_gdt': 'pos_ud_gdt',
            'el.stopwords': 'stopwords_el',
        },
        'ha': {
            'ha.stopwords': 'stopwords_ha'

        },
        'he': {
            'he.stopwords': 'stopwords_he',
            'he.embed' : 'hebrew_cc_300d' ,  # default he embeddings
            'he.embed.glove' : 'hebrew_cc_300d' ,  # default he cbow embeddings
            'he.embed.cbow_300d' : 'hebrew_cc_300d' ,  # default he embeddings
            'he.ner' : 'hebrewner_cc_300d' , #default he ner
            'he.ner.cc_300d' : 'hebrewner_cc_300d' ,
            'he.pos' : 'pos_ud_htb' , #defauklt he pos
            'he.pos.ud_htb' : 'pos_ud_htb' , #
            'he.lemma' : 'lemma' ,


        },
        'hi': {
            'hi.stopwords': 'stopwords_hi',
            'hi.lemma': 'lemma',
            'hi.pos': 'pos_ud_hdtb',
            'hi.embed' : 'hindi_cc_300d'


        },
        'hu': {
            'hu.lemma': 'lemma',
            'hu.pos': 'pos_ud_szeged',  # hu default pos
            'hu.pos.ud_szeged': 'pos_ud_szeged',
            'hu.stopwords': 'stopwords_hu',

        },
        'id': {
            'id.stopwords': 'stopwords_id',
            'id.lemma': 'lemma',
            'id.pos': 'pos_ud_gsd',

        },
        'ga': {
            'ga.stopwords': 'stopwords_ga',
            'ga.lemma': 'lemma',
            'ga.pos': 'pos_ud_idt',


        },
        'da': {
            'da.lemma': 'lemma',
            'da.pos': 'pos_ud_ddt',
            'da.ner': 'dane_ner_6B_100',
            'da.ner.6B_100D': 'dane_ner_6B_100',
            'da.ner.6B_300D': 'dane_ner_6B_300',
            'da.ner.840B_300D': 'dane_ner_840B_300'
            


        },
        'ja': {
            'ja.stopwords': 'stopwords_ja',
            'ja.segment_words' : 'wordseg_gsd_ud' ,
             'ja.ner' : 'ner_ud_gsd_glove_840B_300d' , #default ja ner
            'ja.pos' : 'pos_ud_gsd' ,  # default pos ner
            'ja.ner.ud_gsd' : 'ner_ud_gsd_glove_840B_300d' , # default ner ud_gsd
            'ja.ner.ud_gsd.glove_840B_300D' : 'ner_ud_gsd_glove_840B_300d' ,
            'ja.pos.ud_gsd' : 'pos_ud_gsd' ,
            "ja.lemma": "lemma",

        },
        'la': {
            'la.stopwords': 'stopwords_la',
            'la.lemma': 'lemma',
            'la.pos': 'pos_ud_llct',


        },
        'lv': {
            'lv.stopwords': 'stopwords_lv',
            'lv.lemma': 'lemma',
            'lv.pos': 'pos_ud_lvtb',

        },
        'mr': {
            'mr.stopwords': 'stopwords_mr',
            'mr.lemma': 'lemma',
            'mr.pos': 'pos_ud_ufal',
        },
        'fa': {
            'fa.stopwords': 'stopwords_fa',
            'fa.lemma': 'lemma',
            'fa.pos' : 'pos_ud_perdt',
            'fa.ner' : 'personer_cc_300d' ,#default fa pos
            'fa.ner.person' : 'personer_cc_300d' , # default fa ner person
            'fa.ner.person.cc_300d' : 'personer_cc_300d' ,
            'fa.embed' : 'persian_w2v_cc_300d' , # default fa embeds
            'fa.embed.word2vec' : 'persian_w2v_cc_300d' , # default fa word2vec embeds
            'fa.embed.word2vec.300d' : 'persian_w2v_cc_300d' ,

        },
        'ro': {
            'ro.lemma': 'lemma',
            'ro.pos': 'pos_ud_rrt',
            'ro.pos.ud_rrt': 'pos_ud_rrt',
            'ro.stopwords': 'stopwords_ro',
        },
        'sk': {
            'sk.lemma': 'lemma',
            'sk.pos': 'pos_ud_snk',  # default sk pos
            'sk.pos.ud_snk': 'pos_ud_snk',
            'sk.stopwords': 'stopwords_sk',
        },
        'sl': {
            'sl.stopwords': 'stopwords_sl',
            'sl.lemma': 'lemma',
            'sl.pos': 'pos_ud_ssj',


        },
        'so': {
            'so.stopwords': 'stopwords_so'

        },
        'st': {
            'st.stopwords': 'stopwords_st'
        },
        'sw': {
            'sw.stopwords': 'stopwords_sw'
        },
        'sv': {
            'sv.lemma': 'lemma',
            'sv.pos': 'pos_ud_tal',  # default sv pos
            'sv.pos.ud_tal': 'pos_ud_tal',
            'sv.stopwords': 'stopwords_sv',
            'sv.ner': 'swedish_ner_6B_100',
            'sv.ner.6B_100': 'swedish_ner_6B_100',
            'sv.ner.6B_300': 'swedish_ner_6B_300',
            'sv.ner.840B_300': 'swedish_ner_840B_300'
        },
        'th': {
            'th.stopwords': 'stopwords_th',
            'th.ner.lst20.glove_840B_300D'             : 'ner_lst20_glove_840B_300d',
            "th.segment_words":"wordseg_best",
            "th.pos":"pos_lst20",
            "th.sentiment":"sentiment_jager_use",
            "th.classify.sentiment":"sentiment_jager_use",

        },
        'tr': {
            'tr.lemma': 'lemma',
            'tr.pos': 'pos_ud_imst',  # pos tr default
            'tr.pos.ud_imst': 'pos_ud_imst',
            'tr.stopwords': 'stopwords_tr',
            'tr.ner' : 'turkish_ner_840B_300',#ner tr default
           'tr.ner.bert' : 'turkish_ner_bert'#ner tr default


        },
        'uk': {
            'uk.lemma': 'lemma',  # default uk lemma
            'uk.pos': 'pos_ud_iu',  # default uk pos
            'uk.pos.ud_iu': 'pos_ud_iu',
        },
        'yo': {
            'yo.stopwords': 'stopwords_yo',
            'yo.lemma': 'lemma',
            'yo.pos': 'pos_ud_ytb'

        },
        'zu': {
            'zu.stopwords': 'stopwords_zu'
        },
        'zh' : {

            'zh.segment_words'   : 'wordseg_weibo', #defaul zh wordseg,
            'zh.tokenize'   : 'wordseg_weibo', #defaul tokenize alias

            'zh.segment_words.weibo'   : 'wordseg_weibo',
            'zh.segment_words.pku'   : 'wordseg_pku',
            'zh.segment_words.msra'   : 'wordseg_msra',
            'zh.segment_words.large'   : 'wordseg_large',
            'zh.segment_words.ctb9'   : 'wordseg_ctb9',
            "zh.segment_words.gsd" : "wordseg_gsd_ud_trad",

            'zh.pos'   : 'pos_ud_gsd', # default zh pos,
            'zh.pos.ud_gsd'   : 'pos_ud_gsd',
            'zh.pos.ctb9'   : 'pos_ctb9',
            "zh.pos.ud_gsd_trad": "pos_ud_gsd_trad",

            'zh.ner'   : 'ner_msra_bert_768d', # default zh ner,
            'zh.ner.bert'   : 'ner_msra_bert_768d', # default zh bert nert,
            'zh.ner.msra.bert_768D'   : 'ner_msra_bert_768d', # default zh bert nert,
            'zh.ner.weibo.bert_768d'   : 'ner_weibo_bert_768d',
            'zh.lemma' : 'lemma' ,
            'zh.embed' : 'bert_base_chinese' ,
            'zh.embed.bert' : 'bert_base_chinese' , # default zh embeds


        }
,

        'et': {
            'et.lemma' : 'lemma',
            'et.pos' : 'pos_ud_edt',
        },
        'ur': {
            'ur.lemma' : 'lemma',
            'ur.pos' : 'pos_ud_udtb', #default ur pos
            'ur.pos.ud_udtb' : 'pos_ud_udtb',

            'ur.sentiment' : 'sentimentdl_urduvec_imdb',
            'ur.embed' : 'urduvec_140M_300d' , # default ur embeds
            'ur.embed.urdu_vec_140M_300d' : 'urduvec_140M_300d' ,
            'ur.ner' : 'uner_mk_140M_300d' ,
            'ur.ner.mk_140M_300d' : 'uner_mk_140M_300d' ,

        },

        'ko':{

            'ko.segment_words' : 'wordseg_kaist_ud' ,
            'ko.pos' : 'pos_ud_kaist' ,  #default ko pos
            'ko.ner' : 'ner_kmou_glove_840B_300d' , # default ko ner
            'ko.pos.ud_kaist' : 'pos_ud_kaist' ,
            'ko.ner.kmou' : 'ner_kmou_glove_840B_300d' , #default ner kmou
            'ko.ner.kmou.glove_840B_300d' : 'ner_kmou_glove_840B_300d' ,
            "ko.lemma": "lemma",

        },

        'bh': {
            "bh.pos" : "pos_ud_bhtb",
            "bh.lemma": "lemma",
            },
        'am' : {
            "am.pos":"pos_ud_att",
            "am.lemma":"lemma",
               },



                                 'xx': {
            'xx.embed': 'glove_840B_300',

            'xx.embed.glove.840B_300': 'glove_840B_300',
            'xx.embed.glove.6B_300': 'glove_6B_300',
            'xx.embed.bert_multi_cased': 'bert_multi_cased',
            'xx.embed.bert': 'bert_multi_cased',

            'xx.classify.wiki_7': 'ld_wiki_7',
            'xx.classify.wiki_20': 'ld_wiki_20',
            'xx.classify.wiki_21' : 'ld_wiki_tatoeba_cnn_21' ,
            'xx.classify.wiki_21.bigru' : 'ld_tatoeba_bigru_21' ,
            'xx.classify.wiki_99' : 'ld_tatoeba_cnn_99' ,
            'xx.classify.wiki_231' : 'ld_wiki_cnn_231' ,
            'xx.classify.wiki_43' : 'ld_wiki_tatoeba_cnn_43' ,
            'xx.classify.wiki_95' : 'ld_wiki_tatoeba_cnn_95' ,
            'xx.classify.wiki_220' : 'ld_wiki_tatoeba_cnn_220' ,
            'xx.classify.wiki_375' : 'ld_wiki_tatoeba_cnn_375' ,



            'xx.embed_sentence': 'sent_bert_multi_cased',
            'xx.embed_sentence.bert': 'sent_bert_multi_cased',
            'xx.embed_sentence.bert.cased': 'sent_bert_multi_cased',
            'xx.embed_sentence.labse': 'labse',
            'xx.sentence_detector' : 'sentence_detector_dl',
            'xx.marian' : 'opus_mt_en_fr',

            # 2.7
            'xx.use.multi'   :'tfhub_use_multi',
            'xx.use.multi_lg'   :'tfhub_use_multi_lg',
            'xx.use.xling_en_de'   :'tfhub_use_xling_en_de',
            'xx.use.xling_en_es'   :'tfhub_use_xling_en_es',
            'xx.use.xling_en_fr'   :'tfhub_use_xling_en_fr',
            'xx.use.xling_many'   :'tfhub_use_xling_many',


            # 2.7.0 marian translate model references
            "xx.swc.marian.translate_to.en":"opus_mt_swc_en",
            "xx.en.marian.translate_to.umb":"opus_mt_en_umb",
            "xx.en.marian.translate_to.bem":"opus_mt_en_bem",
            "xx.kwy.marian.translate_to.en":"opus_mt_kwy_en",
            "xx.en.marian.translate_to.ti":"opus_mt_en_ti",
            "xx.efi.marian.translate_to.en":"opus_mt_efi_en",
            "xx.taw.marian.translate_to.en":"opus_mt_taw_en",
            "xx.bem.marian.translate_to.en":"opus_mt_bem_en",
            "xx.en.marian.translate_to.sw":"opus_mt_en_sw",
            "xx.en.marian.translate_to.sn":"opus_mt_en_sn",
            "xx.pqe.marian.translate_to.en":"opus_mt_pqe_en",
            "xx.ny.marian.translate_to.en":"opus_mt_ny_en",
            "xx.urj.marian.translate_to.en":"opus_mt_urj_en",
            "xx.war.marian.translate_to.en":"opus_mt_war_en",
            "xx.nso.marian.translate_to.en":"opus_mt_nso_en",
            "xx.en.marian.translate_to.hu":"opus_mt_en_hu",
            "xx.en.marian.translate_to.cpp":"opus_mt_en_cpp",
            "xx.ja.marian.translate_to.en":"opus_mt_ja_en",
            "xx.en.marian.translate_to.rn":"opus_mt_en_rn",
            "xx.en.marian.translate_to.bcl":"opus_mt_en_bcl",
            "xx.en.marian.translate_to.bi":"opus_mt_en_bi",
            "xx.vi.marian.translate_to.en":"opus_mt_vi_en",
            "xx.ln.marian.translate_to.en":"opus_mt_ln_en",
            "xx.en.marian.translate_to.itc":"opus_mt_en_itc",
            "xx.itc.marian.translate_to.en":"opus_mt_itc_en",
            "xx.uk.marian.translate_to.en":"opus_mt_uk_en",
            "xx.en.marian.translate_to.gmq":"opus_mt_en_gmq",
            "xx.loz.marian.translate_to.en":"opus_mt_loz_en",
            "xx.mg.marian.translate_to.en":"opus_mt_mg_en",
            "xx.bnt.marian.translate_to.en":"opus_mt_bnt_en",
            "xx.en.marian.translate_to.zls":"opus_mt_en_zls",
            "xx.mkh.marian.translate_to.en":"opus_mt_mkh_en",
            "xx.en.marian.translate_to.lg":"opus_mt_en_lg",
            "xx.et.marian.translate_to.en":"opus_mt_et_en",
            "xx.fi.marian.translate_to.en":"opus_mt_fi_en",
            "xx.en.marian.translate_to.vi":"opus_mt_en_vi",
            "xx.en.marian.translate_to.ga":"opus_mt_en_ga",
            "xx.nic.marian.translate_to.en":"opus_mt_nic_en",
            "xx.lua.marian.translate_to.en":"opus_mt_lua_en",
            "xx.afa.marian.translate_to.en":"opus_mt_afa_en",
            "xx.en.marian.translate_to.pis":"opus_mt_en_pis",
            "xx.en.marian.translate_to.ts":"opus_mt_en_ts",
            "xx.sv.marian.translate_to.en":"opus_mt_sv_en",
            "xx.en.marian.translate_to.dra":"opus_mt_en_dra",
            "xx.en.marian.translate_to.pag":"opus_mt_en_pag",
            "xx.en.marian.translate_to.ln":"opus_mt_en_ln",
            "xx.alv.marian.translate_to.en":"opus_mt_alv_en",
            "xx.ee.marian.translate_to.en":"opus_mt_ee_en",
            "xx.bg.marian.translate_to.en":"opus_mt_bg_en",
            "xx.en.marian.translate_to.tw":"opus_mt_en_tw",
            "xx.en.marian.translate_to.gaa":"opus_mt_en_gaa",
            "xx.en.marian.translate_to.tll":"opus_mt_en_tll",
            "xx.en.marian.translate_to.phi":"opus_mt_en_phi",
            "xx.en.marian.translate_to.tiv":"opus_mt_en_tiv",
            "xx.en.marian.translate_to.ss":"opus_mt_en_ss",
            "xx.en.marian.translate_to.cs":"opus_mt_en_cs",
            "xx.kwn.marian.translate_to.en":"opus_mt_kwn_en",
            "xx.mh.marian.translate_to.en":"opus_mt_mh_en",
            "xx.en.marian.translate_to.cy":"opus_mt_en_cy",
            "xx.en.marian.translate_to.mt":"opus_mt_en_mt",
            "xx.en.marian.translate_to.ber":"opus_mt_en_ber",
            "xx.ine.marian.translate_to.en":"opus_mt_ine_en",
            "xx.ng.marian.translate_to.en":"opus_mt_ng_en",
            "xx.ase.marian.translate_to.en":"opus_mt_ase_en",
            "xx.en.marian.translate_to.afa":"opus_mt_en_afa",
            "xx.en.marian.translate_to.ml":"opus_mt_en_ml",
            "xx.en.marian.translate_to.zle":"opus_mt_en_zle",
            "xx.wal.marian.translate_to.en":"opus_mt_wal_en",
            "xx.en.marian.translate_to.sit":"opus_mt_en_sit",
            "xx.hy.marian.translate_to.en":"opus_mt_hy_en",
            "xx.umb.marian.translate_to.en":"opus_mt_umb_en",
            "xx.en.marian.translate_to.cpf":"opus_mt_en_cpf",
            "xx.en.marian.translate_to.gv":"opus_mt_en_gv",
            "xx.en.marian.translate_to.pap":"opus_mt_en_pap",
            "xx.gl.marian.translate_to.en":"opus_mt_gl_en",
            "xx.kqn.marian.translate_to.en":"opus_mt_kqn_en",
            "xx.ca.marian.translate_to.en":"opus_mt_ca_en",
            "xx.gem.marian.translate_to.en":"opus_mt_gem_en",
            "xx.en.marian.translate_to.crs":"opus_mt_en_crs",
            "xx.en.marian.translate_to.rw":"opus_mt_en_rw",
            "xx.en.marian.translate_to.ig":"opus_mt_en_ig",
            "xx.en.marian.translate_to.aav":"opus_mt_en_aav",
            "xx.en.marian.translate_to.tl":"opus_mt_en_tl",
            "xx.sm.marian.translate_to.en":"opus_mt_sm_en",
            "xx.en.marian.translate_to.hi":"opus_mt_en_hi",
            "xx.chk.marian.translate_to.en":"opus_mt_chk_en",
            "xx.en.marian.translate_to.kwy":"opus_mt_en_kwy",
            "xx.en.marian.translate_to.ho":"opus_mt_en_ho",
            "xx.ve.marian.translate_to.en":"opus_mt_ve_en",
            "xx.ceb.marian.translate_to.en":"opus_mt_ceb_en",
            "xx.aav.marian.translate_to.en":"opus_mt_aav_en",
            "xx.st.marian.translate_to.en":"opus_mt_st_en",
            "xx.en.marian.translate_to.cus":"opus_mt_en_cus",
            "xx.en.marian.translate_to.bnt":"opus_mt_en_bnt",
            "xx.iso.marian.translate_to.en":"opus_mt_iso_en",
            "xx.tr.marian.translate_to.en":"opus_mt_tr_en",
            "xx.om.marian.translate_to.en":"opus_mt_om_en",
            "xx.fj.marian.translate_to.en":"opus_mt_fj_en",
            "xx.ml.marian.translate_to.en":"opus_mt_ml_en",
            "xx.en.marian.translate_to.ro":"opus_mt_en_ro",
            "xx.en.marian.translate_to.ny":"opus_mt_en_ny",
            "xx.en.marian.translate_to.ty":"opus_mt_en_ty",
            "xx.cel.marian.translate_to.en":"opus_mt_cel_en",
            "xx.ccs.marian.translate_to.en":"opus_mt_ccs_en",
            "xx.hu.marian.translate_to.en":"opus_mt_hu_en",
            "xx.art.marian.translate_to.en":"opus_mt_art_en",
            "xx.en.marian.translate_to.ar":"opus_mt_en_ar",
            "xx.en.marian.translate_to.mfe":"opus_mt_en_mfe",
            "xx.yap.marian.translate_to.en":"opus_mt_yap_en",
            "xx.ha.marian.translate_to.en":"opus_mt_ha_en",
            "xx.cpf.marian.translate_to.en":"opus_mt_cpf_en",
            "xx.en.marian.translate_to.hy":"opus_mt_en_hy",
            "xx.en.marian.translate_to.roa":"opus_mt_en_roa",
            "xx.crs.marian.translate_to.en":"opus_mt_crs_en",
            "xx.en.marian.translate_to.lun":"opus_mt_en_lun",
            "xx.zls.marian.translate_to.en":"opus_mt_zls_en",
            "xx.en.marian.translate_to.mk":"opus_mt_en_mk",
            "xx.en.marian.translate_to.fiu":"opus_mt_en_fiu",
            "xx.en.marian.translate_to.sv":"opus_mt_en_sv",
            "xx.gmw.marian.translate_to.en":"opus_mt_gmw_en",
            "xx.en.marian.translate_to.st":"opus_mt_en_st",
            "xx.en.marian.translate_to.om":"opus_mt_en_om",
            "xx.en.marian.translate_to.pqw":"opus_mt_en_pqw",
            "xx.en.marian.translate_to.hil":"opus_mt_en_hil",
            "xx.sk.marian.translate_to.en":"opus_mt_sk_en",
            "xx.en.marian.translate_to.mr":"opus_mt_en_mr",
            "xx.tll.marian.translate_to.en":"opus_mt_tll_en",
            "xx.en.marian.translate_to.jap":"opus_mt_en_jap",
            "xx.en.marian.translate_to.chk":"opus_mt_en_chk",
            "xx.tum.marian.translate_to.en":"opus_mt_tum_en",
            "xx.dra.marian.translate_to.en":"opus_mt_dra_en",
            "xx.en.marian.translate_to.kqn":"opus_mt_en_kqn",
            "xx.en.marian.translate_to.tdt":"opus_mt_en_tdt",
            "xx.en.marian.translate_to.tvl":"opus_mt_en_tvl",
            "xx.en.marian.translate_to.lue":"opus_mt_en_lue",
            "xx.en.marian.translate_to.ilo":"opus_mt_en_ilo",
            "xx.en.marian.translate_to.poz":"opus_mt_en_poz",
            "xx.pap.marian.translate_to.en":"opus_mt_pap_en",
            "xx.af.marian.translate_to.en":"opus_mt_af_en",
            "xx.en.marian.translate_to.bzs":"opus_mt_en_bzs",
            "xx.toi.marian.translate_to.en":"opus_mt_toi_en",
            "xx.en.marian.translate_to.mg":"opus_mt_en_mg",
            "xx.ber.marian.translate_to.en":"opus_mt_ber_en",
            "xx.kj.marian.translate_to.en":"opus_mt_kj_en",
            "xx.ko.marian.translate_to.en":"opus_mt_ko_en",
            "xx.inc.marian.translate_to.en":"opus_mt_inc_en",
            "xx.en.marian.translate_to.is":"opus_mt_en_is",
            "xx.mr.marian.translate_to.en":"opus_mt_mr_en",
            "xx.ka.marian.translate_to.en":"opus_mt_ka_en",
            "xx.en.marian.translate_to.nic":"opus_mt_en_nic",
            "xx.en.marian.translate_to.eu":"opus_mt_en_eu",
            "xx.pl.marian.translate_to.en":"opus_mt_pl_en",
            "xx.hil.marian.translate_to.en":"opus_mt_hil_en",
            "xx.en.marian.translate_to.luo":"opus_mt_en_luo",
            "xx.en.marian.translate_to.zh":"opus_mt_en_zh",
            "xx.en.marian.translate_to.fi":"opus_mt_en_fi",
            "xx.en.marian.translate_to.kg":"opus_mt_en_kg",
            "xx.bi.marian.translate_to.en":"opus_mt_bi_en",
            "xx.sg.marian.translate_to.en":"opus_mt_sg_en",
            "xx.en.marian.translate_to.ng":"opus_mt_en_ng",
            "xx.en.marian.translate_to.inc":"opus_mt_en_inc",
            "xx.en.marian.translate_to.uk":"opus_mt_en_uk",
            "xx.en.marian.translate_to.es":"opus_mt_en_es",
            "xx.en.marian.translate_to.swc":"opus_mt_en_swc",
            "xx.sq.marian.translate_to.en":"opus_mt_sq_en",
            "xx.en.marian.translate_to.niu":"opus_mt_en_niu",
            "xx.trk.marian.translate_to.en":"opus_mt_trk_en",
            "xx.en.marian.translate_to.mh":"opus_mt_en_mh",
            "xx.luo.marian.translate_to.en":"opus_mt_luo_en",
            "xx.mt.marian.translate_to.en":"opus_mt_mt_en",
            "xx.to.marian.translate_to.en":"opus_mt_to_en",
            "xx.tiv.marian.translate_to.en":"opus_mt_tiv_en",
            "xx.en.marian.translate_to.ca":"opus_mt_en_ca",
            "xx.en.marian.translate_to.iso":"opus_mt_en_iso",
            "xx.gil.marian.translate_to.en":"opus_mt_gil_en",
            "xx.en.marian.translate_to.trk":"opus_mt_en_trk",
            "xx.ga.marian.translate_to.en":"opus_mt_ga_en",
            "xx.en.marian.translate_to.sem":"opus_mt_en_sem",
            "xx.da.marian.translate_to.en":"opus_mt_da_en",
            "xx.is.marian.translate_to.en":"opus_mt_is_en",
            "xx.en.marian.translate_to.mul":"opus_mt_en_mul",
            "xx.en.marian.translate_to.nyk":"opus_mt_en_nyk",
            "xx.de.marian.translate_to.en":"opus_mt_de_en",
            "xx.rn.marian.translate_to.en":"opus_mt_rn_en",
            "xx.yo.marian.translate_to.en":"opus_mt_yo_en",
            "xx.sn.marian.translate_to.en":"opus_mt_sn_en",
            "xx.lun.marian.translate_to.en":"opus_mt_lun_en",
            "xx.en.marian.translate_to.iir":"opus_mt_en_iir",
            "xx.en.marian.translate_to.toi":"opus_mt_en_toi",
            "xx.pag.marian.translate_to.en":"opus_mt_pag_en",
            "xx.en.marian.translate_to.ee":"opus_mt_en_ee",
            "xx.grk.marian.translate_to.en":"opus_mt_grk_en",
            "xx.zle.marian.translate_to.en":"opus_mt_zle_en",
            "xx.zh.marian.translate_to.en":"opus_mt_zh_en",
            "xx.en.marian.translate_to.tn":"opus_mt_en_tn",
            "xx.en.marian.translate_to.az":"opus_mt_en_az",
            "xx.en.marian.translate_to.bat":"opus_mt_en_bat",
            "xx.en.marian.translate_to.tut":"opus_mt_en_tut",
            "xx.eu.marian.translate_to.en":"opus_mt_eu_en",
            "xx.kab.marian.translate_to.en":"opus_mt_kab_en",
            "xx.en.marian.translate_to.fr":"opus_mt_en_fr",
            "xx.rnd.marian.translate_to.en":"opus_mt_rnd_en",
            "xx.ur.marian.translate_to.en":"opus_mt_ur_en",
            "xx.ss.marian.translate_to.en":"opus_mt_ss_en",
            "xx.cau.marian.translate_to.en":"opus_mt_cau_en",
            "xx.cs.marian.translate_to.en":"opus_mt_cs_en",
            "xx.en.marian.translate_to.de":"opus_mt_en_de",
            "xx.en.marian.translate_to.sq":"opus_mt_en_sq",
            "xx.roa.marian.translate_to.en":"opus_mt_roa_en",
            "xx.eo.marian.translate_to.en":"opus_mt_eo_en",
            "xx.wa.marian.translate_to.en":"opus_mt_wa_en",
            "xx.zlw.marian.translate_to.en":"opus_mt_zlw_en",
            "xx.lg.marian.translate_to.en":"opus_mt_lg_en",
            "xx.en.marian.translate_to.gmw":"opus_mt_en_gmw",
            "xx.fr.marian.translate_to.en":"opus_mt_fr_en",
            "xx.en.marian.translate_to.it":"opus_mt_en_it",
            "xx.tpi.marian.translate_to.en":"opus_mt_tpi_en",
            "xx.bcl.marian.translate_to.en":"opus_mt_bcl_en",
            "xx.id.marian.translate_to.en":"opus_mt_id_en",
            "xx.ht.marian.translate_to.en":"opus_mt_ht_en",
            "xx.en.marian.translate_to.run":"opus_mt_en_run",
            "xx.en.marian.translate_to.bg":"opus_mt_en_bg",
            "xx.mul.marian.translate_to.en":"opus_mt_mul_en",
            "xx.mk.marian.translate_to.en":"opus_mt_mk_en",
            "xx.en.marian.translate_to.gil":"opus_mt_en_gil",
            "xx.en.marian.translate_to.nso":"opus_mt_en_nso",
            "xx.gv.marian.translate_to.en":"opus_mt_gv_en",
            "xx.es.marian.translate_to.en":"opus_mt_es_en",
            "xx.ilo.marian.translate_to.en":"opus_mt_ilo_en",
            "xx.srn.marian.translate_to.en":"opus_mt_srn_en",
            "xx.guw.marian.translate_to.en":"opus_mt_guw_en",
            "xx.fiu.marian.translate_to.en":"opus_mt_fiu_en",
            "xx.wls.marian.translate_to.en":"opus_mt_wls_en",
            "xx.en.marian.translate_to.nl":"opus_mt_en_nl",
            "xx.rw.marian.translate_to.en":"opus_mt_rw_en",
            "xx.en.marian.translate_to.ceb":"opus_mt_en_ceb",
            "xx.run.marian.translate_to.en":"opus_mt_run_en",
            "xx.nyk.marian.translate_to.en":"opus_mt_nyk_en",
            "xx.ho.marian.translate_to.en":"opus_mt_ho_en",
            "xx.en.marian.translate_to.urj":"opus_mt_en_urj",
            "xx.en.marian.translate_to.pon":"opus_mt_en_pon",
            "xx.phi.marian.translate_to.en":"opus_mt_phi_en",
            "xx.kl.marian.translate_to.en":"opus_mt_kl_en",
            "xx.bn.marian.translate_to.en":"opus_mt_bn_en",
            "xx.en.marian.translate_to.efi":"opus_mt_en_efi",
            "xx.en.marian.translate_to.pqe":"opus_mt_en_pqe",
            "xx.en.marian.translate_to.rnd":"opus_mt_en_rnd",
            "xx.en.marian.translate_to.id":"opus_mt_en_id",
            "xx.nl.marian.translate_to.en":"opus_mt_nl_en",
            "xx.ru.marian.translate_to.en":"opus_mt_ru_en",
            "xx.en.marian.translate_to.sla":"opus_mt_en_sla",
            "xx.en.marian.translate_to.guw":"opus_mt_en_guw",
            "xx.euq.marian.translate_to.en":"opus_mt_euq_en",
            "xx.lu.marian.translate_to.en":"opus_mt_lu_en",
            "xx.en.marian.translate_to.fj":"opus_mt_en_fj",
            "xx.en.marian.translate_to.ru":"opus_mt_en_ru",
            "xx.en.marian.translate_to.gem":"opus_mt_en_gem",
            "xx.en.marian.translate_to.map":"opus_mt_en_map",
            "xx.en.marian.translate_to.euq":"opus_mt_en_euq",
            "xx.en.marian.translate_to.kwn":"opus_mt_en_kwn",
            "xx.en.marian.translate_to.gl":"opus_mt_en_gl",
            "xx.en.marian.translate_to.ha":"opus_mt_en_ha",
            "xx.cpp.marian.translate_to.en":"opus_mt_cpp_en",
            "xx.en.marian.translate_to.mkh":"opus_mt_en_mkh",
            "xx.en.marian.translate_to.kj":"opus_mt_en_kj",
            "xx.bat.marian.translate_to.en":"opus_mt_bat_en",
            "xx.en.marian.translate_to.sal":"opus_mt_en_sal",
            "xx.pa.marian.translate_to.en":"opus_mt_pa_en",
            "xx.jap.marian.translate_to.en":"opus_mt_jap_en",
            "xx.sla.marian.translate_to.en":"opus_mt_sla_en",
            "xx.ar.marian.translate_to.en":"opus_mt_ar_en",
            "xx.xh.marian.translate_to.en":"opus_mt_xh_en",
            "xx.mfe.marian.translate_to.en":"opus_mt_mfe_en",
            "xx.en.marian.translate_to.zlw":"opus_mt_en_zlw",
            "xx.en.marian.translate_to.lua":"opus_mt_en_lua",
            "xx.en.marian.translate_to.ine":"opus_mt_en_ine",
            "xx.en.marian.translate_to.loz":"opus_mt_en_loz",
            "xx.lus.marian.translate_to.en":"opus_mt_lus_en",
            "xx.lv.marian.translate_to.en":"opus_mt_lv_en",
            "xx.en.marian.translate_to.af":"opus_mt_en_af",
            "xx.sal.marian.translate_to.en":"opus_mt_sal_en",
            "xx.hi.marian.translate_to.en":"opus_mt_hi_en",
            "xx.gaa.marian.translate_to.en":"opus_mt_gaa_en",
            "xx.pon.marian.translate_to.en":"opus_mt_pon_en",
            "xx.ig.marian.translate_to.en":"opus_mt_ig_en",
            "xx.en.marian.translate_to.lus":"opus_mt_en_lus",
            "xx.th.marian.translate_to.en":"opus_mt_th_en",
            "xx.kg.marian.translate_to.en":"opus_mt_kg_en",
            "xx.cus.marian.translate_to.en":"opus_mt_cus_en",
            "xx.cy.marian.translate_to.en":"opus_mt_cy_en",
            "xx.en.marian.translate_to.alv":"opus_mt_en_alv",
            "xx.sem.marian.translate_to.en":"opus_mt_sem_en",
            "xx.en.marian.translate_to.tpi":"opus_mt_en_tpi",
            "xx.en.marian.translate_to.grk":"opus_mt_en_grk",
            "xx.gmq.marian.translate_to.en":"opus_mt_gmq_en",
            "xx.tn.marian.translate_to.en":"opus_mt_tn_en",
            "xx.en.marian.translate_to.sg":"opus_mt_en_sg",
            "xx.pis.marian.translate_to.en":"opus_mt_pis_en",
            "xx.en.marian.translate_to.sm":"opus_mt_en_sm",
            "xx.en.marian.translate_to.ht":"opus_mt_en_ht",
            "xx.mos.marian.translate_to.en":"opus_mt_mos_en",
            "xx.iir.marian.translate_to.en":"opus_mt_iir_en",
            "xx.en.marian.translate_to.mos":"opus_mt_en_mos",
            "xx.ts.marian.translate_to.en":"opus_mt_ts_en",
            "xx.en.marian.translate_to.he":"opus_mt_en_he",
            "xx.en.marian.translate_to.da":"opus_mt_en_da",
            "xx.it.marian.translate_to.en":"opus_mt_it_en",
            "xx.en.marian.translate_to.cel":"opus_mt_en_cel",
            "xx.en.marian.translate_to.el":"opus_mt_en_el",
            "xx.en.marian.translate_to.xh":"opus_mt_en_xh",
            "xx.en.marian.translate_to.eo":"opus_mt_en_eo",
            "xx.az.marian.translate_to.en":"opus_mt_az_en",
            "xx.tvl.marian.translate_to.en":"opus_mt_tvl_en",
            "xx.tl.marian.translate_to.en":"opus_mt_tl_en",
            "xx.bzs.marian.translate_to.en":"opus_mt_bzs_en",
            "xx.en.marian.translate_to.et":"opus_mt_en_et",
            "xx.en.marian.translate_to.sk":"opus_mt_en_sk",
            "xx.en.marian.translate_to.to":"opus_mt_en_to",
            "xx.en.marian.translate_to.ur":"opus_mt_en_ur",
            "xx.lue.marian.translate_to.en":"opus_mt_lue_en",
            "xx.niu.marian.translate_to.en":"opus_mt_niu_en",
            "xx.ti.marian.translate_to.en":"opus_mt_ti_en",
            "xx.en.marian.translate_to.lu":"opus_mt_en_lu"

        },

    }
