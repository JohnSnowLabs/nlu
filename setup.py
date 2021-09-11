"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

REQUIRED_PKGS = [
    # 'pyspark>=2.4.0,<2.5',
    'spark-nlp>=3.2.0,<3.3.0',
    'numpy',
    'pyarrow>=0.16.0',
    'pandas',
    'dataclasses'
]
# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine how
    # users can install this project, e.g.:
    #
    # $ pip install sampleproject
    #
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    # There are some restrictions on what makes a valid project name
    # specification here:    # https://packaging.python.org/specifications/core-metadata/#name
    name='nlu',  #  Required #nlu

    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='3.2.0',  # Required

    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#summary
    description='John Snow Labs NLU provides state of the art algorithms for NLP&NLU with 1000+ of pretrained models in 200+ languages. It enables swift and simple development and research with its powerful Pythonic and Keras inspired API. It is powerd by John Snow Labs powerful Spark NLP library.',

    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    long_description=long_description,  # Optional
    install_requires=REQUIRED_PKGS,

    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructedText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',  # Optional (see note above)

    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url='http://nlu.johnsnowlabs.com',  # Optional

    # This should be your name or the name of the organization which owns the
    # project.
    author='John Snow Labs',  # Optional

    # This should be a valid email address corresponding to the author listed
    # above.
    author_email='christian@johnsnowlabs.com',

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[  # Optional
        # 'Development Status :: 3 - Alpha',
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # This field adds keywords for your project which will appear on the
    # project page. What does your project relate to?
    #
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='NLP spark development NLU ',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],

    packages=find_packages(exclude=['test*','tmp*']) , # exclude=['test']
    data_files=[
        #classifiers
        ('', ['nlu/components/classifiers/classifier_dl/component_infos.json']),
        ('', ['nlu/components/classifiers/language_detector/component_infos.json']),
        ('', ['nlu/components/classifiers/named_entity_recognizer_crf/component_infos.json']),
        ('', ['nlu/components/classifiers/ner/component_infos.json']),
        ('', ['nlu/components/classifiers/pos/component_infos.json']),
        ('', ['nlu/components/classifiers/sentiment_detector/component_infos.json']),
        ('', ['nlu/components/classifiers/sentiment_dl/component_infos.json']),
        ('', ['nlu/components/classifiers/vivekn_sentiment/component_infos.json']),
        ('', ['nlu/components/classifiers/yake/component_infos.json']),
        ('', ['nlu/components/classifiers/multi_classifier/component_infos.json']),

        #dependency
        ('', ['nlu/components/dependency_typeds/labeled_dependency_parser/component_infos.json']),
        ('', ['nlu/components/dependency_untypeds/unlabeled_dependency_parser/component_infos.json']),

        #embeds
        ('', ['nlu/components/embeddings/elmo/component_infos.json']),
        ('', ['nlu/components/embeddings/albert/component_infos.json']),
        ('', ['nlu/components/embeddings/bert/component_infos.json']),
        ('', ['nlu/components/embeddings/glove/component_infos.json']),
        ('', ['nlu/components/embeddings/xlnet/component_infos.json']),
        ('', ['nlu/components/embeddings/use/component_infos.json']),
        ('', ['nlu/components/embeddings/sentence_bert/component_infos.json']),

        ('', ['nlu/components/embeddings/distil_bert/component_infos.json']),
        ('', ['nlu/components/embeddings/roberta/component_infos.json']),
        ('', ['nlu/components/embeddings/xlm/component_infos.json']),


        ('', ['nlu/components/embeddings/distil_bert/component_infos.json']),
        ('', ['nlu/components/embeddings/longformer/component_infos.json']),
        ('', ['nlu/components/embeddings/token_bert/component_infos.json']),
        ('', ['nlu/components/embeddings/token_distilbert/component_infos.json']),


        #Seq2Seq

        ('', ['nlu/components/seq2seqs/marian/component_infos.json']),
        ('', ['nlu/components/seq2seqs/t5/component_infos.json']),



        #lemma
        ('', ['nlu/components/lemmatizers/lemmatizer/component_infos.json']),


        #matcher
        ('', ['nlu/components/matchers/date_matcher/component_infos.json']),
        ('', ['nlu/components/matchers/regex_matcher/component_infos.json']),
        ('', ['nlu/components/matchers/text_matcher/component_infos.json']),
        ('', ['nlu/components/matchers/context_parser/component_infos.json']),


        #normalizer
        ('', ['nlu/components/normalizers/normalizer/component_infos.json']),
        ('', ['nlu/components/normalizers/document_normalizer/component_infos.json']),
        ('', ['nlu/components/normalizers/drug_normalizer/component_infos.json']),

        #sentence detector
        ('', ['nlu/components/sentence_detectors/deep_sentence_detector/component_infos.json']),
        ('', ['nlu/components/sentence_detectors/pragmatic_sentence_detector/component_infos.json']),

        #spell checker
        ('', ['nlu/components/spell_checkers/context_spell/component_infos.json']),
        ('', ['nlu/components/spell_checkers/norvig_spell/component_infos.json']),
        ('', ['nlu/components/spell_checkers/symmetric_spell/component_infos.json']),

        #stemmer
        ('', ['nlu/components/stemmers/stemmer/component_infos.json']),
        #tokenizer
        ('', ['nlu/components/tokenizers/default_tokenizer/component_infos.json']),
        ('', ['nlu/components/tokenizers/regex_tokenizer/component_infos.json']),
        ('', ['nlu/components/tokenizers/word_segmenter/component_infos.json']),

        #stopwords
        ('', ['nlu/components/stopwordscleaners/stopwordcleaner/component_infos.json']),

        #chunker
        ('', ['nlu/components/chunkers/default_chunker/component_infos.json']),
        ('', ['nlu/components/chunkers/ngram/component_infos.json']),
        ('', ['nlu/components/chunkers/contextual_parser/component_infos.json']),

        ('', ['nlu/components/embeddings_chunks/chunk_embedder/component_infos.json']),


        #utils
        ('', ['nlu/components/utils/chunk_2_doc/component_infos.json']),
        ('', ['nlu/components/utils/doc2chunk/component_infos.json']),
        ('', ['nlu/components/utils/deep_sentence_detector/component_infos.json']),
        ('', ['nlu/components/utils/document_assembler/component_infos.json']),
        ('', ['nlu/components/utils/sentence_detector/component_infos.json']),
        ('', ['nlu/components/utils/sentence_embeddings/component_infos.json']),
        ('', ['nlu/components/utils/ner_to_chunk_converter/component_infos.json']),
        ('', ['nlu/components/utils/chunk_merger/component_infos.json']),

        ('', ['nlu/components/utils/token_assembler/component_infos.json']),
        ('', ['nlu/components/deidentifiers/deidentifier/component_infos.json']),
        ('', ['nlu/components/relation_extractors/relation_extractor/component_infos.json']),
        ('', ['nlu/components/relation_extractors/relation_extractor_dl/component_infos.json']),
        ('', ['nlu/components/resolutions/chunk_entity_resolver/component_infos.json']),
        ('', ['nlu/components/resolutions/sentence_entity_resolver/component_infos.json']),
        ('', ['nlu/components/assertions/assertion_dl/component_infos.json']),
        ('', ['nlu/components/assertions/assertion_log_reg/component_infos.json']),
        ('', ['nlu/components/classifiers/generic_classifier/component_infos.json']),
        ('', ['nlu/components/utils/feature_assembler/component_infos.json']),
        ('', ['nlu/components/classifiers/ner_healthcare/component_infos.json']),
        ('', ['nlu/components/utils/ner_to_chunk_converter_licensed/component_infos.json']),

    ],

    include_package_data=True  # Needed to install jar file

)

