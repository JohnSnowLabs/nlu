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
    actions = ['tokenize', 'sentence', 'embed', 'embed_sentence', 'embed_chunk','classify', 'chunk', 'pos', 'ner',
               'dep', 'dep.untyped', 'lemma', 'match', 'norm', 'spell','stem', 'stopwords','clean','ngram',
               ]
    trainable_model_references = ['classifier_dl']



    '''
    NLU REFERENCE FORMATS : 
    <lang>.<action>.<
    '''
    trainable_models = {
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
        'lang': ('detect_language_20','pipe'),  # multi lang alias
        'lang.7': ('detect_language_7','pipe'),  # multi lang detector alias
        'lang.20': ('detect_language_20','pipe'),  # multi lang detector alias
        'classify.lang': ('detect_language_20','pipe'),  # multi lang detector default
        'classify.lang.20': ('detect_language_20','pipe'),  # multi lang detector default
        'classify.lang.7': ('detect_language_7','pipe'),


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
        'spell': ('check_spelling_dl','pipe'),  # default spell 
        'sentiment': ('analyze_sentiment','pipe'),
        'emotion': ('classifierdl_use_emotion','model'), # default emotion model

        'sentiment.imdb': ('analyze_sentimentdl_use_imdb','pipe'),
        'sentiment.imdb.use': ('analyze_sentimentdl_use_imdb','pipe'),
        'sentiment.twitter.use': ('analyze_sentimentdl_use_twitter','pipe'),
        'sentiment.twitter': ('analyze_sentimentdl_use_twitter','pipe'),
        'dependency': ('dependency_parse','pipe'),

        # models
        'tokenize': ('spark_nlp_tokenizer', 'model'),  # tokenizer rule based model
        'stem': ('stemmer', 'model'),  # stem rule based model
        'norm': ('normalizer', 'model'),  #  rule based model
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
        'sentence_detector': ('ner_dl_sentence', 'model'),
        'sentence_detector.deep': ('ner_dl_sentence', 'model'), #ALIAS
        # 'sentence_detector.pragmatic': ('ner_dl_sentence', 'model'), # todo

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
            'lang': 'detect_language_20',  # multi lang alias
            'lang.7': 'detect_language_7',  # multi lang detector alias
            'lang.20': 'detect_language_20',  # multi lang detector alias
            'xx.classify.lang': 'detect_language_20',  # multi lang detector default
            'xx.classify.lang.20': 'detect_language_20',  # multi lang detector default
            'xx.classify.lang.7': 'detect_language_7',
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
            'en.ner.onto.glove.6B_300d': 'onto_300',  # this uses multi lang embeds!
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
            'pt.ner.large': 'bert_portuguese_large_cased',
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
            'ar.stopwords_ar': 'stopwords_ar'

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
            'bn.stopwords': 'stopwords_bn'

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
            'fi.embed_sentence.bert.uncased': 'sent_bert_finnish_uncased'

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
            'he.stopwords': 'stopwords_he'

        },
        'hi': {
            'hi.stopwords': 'stopwords_hi',
            'hi.lemma': 'lemma',
            'hi.pos': 'pos_ud_hdtb',

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
            'da.ner.6B100D': 'dane_ner_6B_100',
            'da.ner.6B300D': 'dane_ner_6B_300',
            'da.ner840B100D.': 'dane_ner_840B_100'
            


        },
        'ja': {
            'ja.stopwords': 'stopwords_ja'

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
            'fa.stopwords': 'stopwords_fa'

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
            'th.stopwords': 'stopwords_th'
        },
        'tr': {
            'tr.lemma': 'lemma',
            'tr.pos': 'pos_ud_imst',  # pos tr default
            'tr.pos.ud_imst': 'pos_ud_imst',
            'tr.stopwords': 'stopwords_tr',
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
        'xx': {
            'xx.embed': 'glove_840B_300',

            'xx.embed.glove.840B_300': 'glove_840B_300',
            'xx.embed.glove.6B_300': 'glove_6B_300',
            'xx.embed.bert_multi_cased': 'bert_multi_cased',
            'xx.classify.wiki_7': 'ld_wiki_7',
            'xx.classify.wiki_20': 'ld_wiki_20',

            'xx.embed_sentence': 'sent_bert_multi_cased',
            'xx.embed_sentence.bert': 'sent_bert_multi_cased',
            'xx.embed_sentence.bert.cased': 'sent_bert_multi_cased',
            'xx.embed_sentence.labse': 'labse'

        },

    }
