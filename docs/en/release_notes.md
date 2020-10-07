---
layout: article
title: Spark NLP release notes
permalink: /docs/en/release_notes
key: docs-release-notes
modify_date: "2020-06-12"
---

NLU release notes


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

- Chunkers
    -[N Gram](https://colab.research.google.com/drive/1pgqoRJ6yGWbTLWdLnRvwG5DLSU3rxuMq?usp=sharing)
    -[Entity Chunking](https://colab.research.google.com/drive/1svpqtC3cY6JnRGeJngIPl2raqxdowpyi?usp=sharing)
- Matchers
    -[Date Matcher](https://colab.research.google.com/drive/1JrlfuV2jNGTdOXvaWIoHTSf6BscDMkN7?usp=sharing)



NLU on Medium :
Introduction to NLU
One line BERT Word Embeddings and t-SNE plotting with NLU
BERT, ALBERT, ELECTRA, ELMO, XLNET, GLOVE Word Embeddings in one line and plotting with t-SNE
NLU Documentation
NLU website
NLU Github
NLU Documentation
Overview of all NLU example notebooks
Having questions or want to share an idea? Join the new NLU slack channel! 


