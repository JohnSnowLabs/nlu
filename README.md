
# NLU: The Power of Spark NLP, the Simplicity of Python
John Snow Labs' NLU is a Python library for applying state-of-the-art text mining, directly on any dataframe, with a single line of code.
As a facade of the award-winning Spark NLP library, it comes with hundreds of pretrained models in tens of languages - all production-grade, scalable, and trainable.

## Project's Website
Take a look at our official Spark NLU page: [https://nlu.johnsnowlabs.com/](https://nlu.johnsnowlabs.com/)  for user documentation and examples



## NLU in action 
<img src="http://ckl-it.de/wp-content/uploads/2020/08/My-Video6.gif" width="1800" height="500"/>



## Getting Started with NLU 
To get your hands on the power of NLU, you just need to install it via pip and ensure Java 8 is installed and properly configured. Checkout [Quickstart for more infos](https://nlu.johnsnowlabs.com/docs/en/install)
```bash 
pip install nlu
``` 

## Loading and predict with any model in 1 line python 
```python
import nlu 
nlu.load('sentiment').predict('I love NLU! <3') 
``` 

## Loading and predict with multiple models in 1 line 

Get 6 different embeddings in 1 line and use them for downstream data science tasks! 

```python 
nlu.load('bert elmo albert xlnet glove use').predict('I love NLU! <3') 
``` 

## NLU notebooks and examples

Checkout the following notebooks for examples how how to work with NLU.


### NLU Demos on Datasets
- [Kaggle Twitter Airline Sentiment Analysis NLU demo](https://www.kaggle.com/kasimchristianloan/nlu-sentiment-airline-demo)
- [Kaggle Twitter Airline Emotion Analysis NLU demo](https://www.kaggle.com/kasimchristianloan/nlu-emotion-airline-demo)
- [Kaggle Twitter COVID Sentiment Analysis NLU demo](https://www.kaggle.com/kasimchristianloan/nlu-covid-sentiment-showcase)
- [Kaggle Twitter COVID Emotion Analysis nlu demo](https://www.kaggle.com/kasimchristianloan/nlu-covid-emotion-showcase)


### NLU examples by component types

- Named Entity Recognition (NER)
    - [NER pretrained on ONTO Notes](https://colab.research.google.com/drive/1_sgbJV3dYPZ_Q7acCgKWgqZkWcKAfg79?usp=sharing)
    - [NER pretrained on CONLL](https://colab.research.google.com/drive/1CYzHfQyFCdvIOVO2Z5aggVI9c0hDEOrw?usp=sharing)
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
- Dependency Parsing 
    - [Untyped Dependency Parsing](https://colab.research.google.com/drive/1PC8ga_NFlOcTNeDVJY4x8Pl5oe0jVmue?usp=sharing)
    - [Typed Dependency Parsing](https://colab.research.google.com/drive/1KXUqcF8e-LU9cXnHE8ni8z758LuFPvY7?usp=sharing)
- Text Pre Processing and Cleaning
    - [Tokenization](https://colab.research.google.com/drive/13BC6k6gLj1w5RZ0SyHjKsT2EOwJwbYwb?usp=sharing)
    - [Stopwords removal](https://colab.research.google.com/drive/1nWob4u93t2EJYupcOIanuPBDfShtYjGT?usp=sharing)
    - [Stemming](https://colab.research.google.com/drive/1gKTJJmffR9wz13Ms3pDy64jhUI8ZHZYu?usp=sharing)
    - [Lemmatization](https://colab.research.google.com/drive/1cBtx9cVCjavt-Oq5TG1lO-9JfUfqznnK?usp=sharing)
    - [Normalizing](https://colab.research.google.com/drive/1kfnnwkiQPQa465Jic6va9QXTRssU4mlX?usp=sharing)
    - [Spellchecking](https://colab.research.google.com/drive/1bnRR8FygiiN3zJz3mRdbjPBUvFsx6IVB?usp=sharing)
    - [Sentence Detecting](https://colab.research.google.com/drive/1CAXEdRk_q3U5qbMXsxoVyZRwvonKthhF?usp=sharing)
- Chunkers
    - [N Gram](https://colab.research.google.com/drive/1pgqoRJ6yGWbTLWdLnRvwG5DLSU3rxuMq?usp=sharing)
    - [Entity Chunking](https://colab.research.google.com/drive/1svpqtC3cY6JnRGeJngIPl2raqxdowpyi?usp=sharing)
- Matchers
    - [Date Matcher](https://colab.research.google.com/drive/1JrlfuV2jNGTdOXvaWIoHTSf6BscDMkN7?usp=sharing)

## What kind of models does NLU provide? 
NLU provides everything a data scientist might want to wish for in one line of code!  
- The 14+ of the latest NLP embeddings ( BERT, ELMO, ALBERT, XLNET, GLOVE, BIOBERT, USE) and different variations of them 
- Generation of Sentence, Chunk and Document from these embeddings 
- Language Classification of 20 languages 
- 36 pretrained NER models 
- 34 Part of Speech (POS) models
- 34 Lemmatizer models    
- Emotion models for 5 categories 
- Labeled and Unlabeled Dependency parsing 
- Spell Checking 
- Stopword removers for 41  languages 
- Classifiers for 12 different problems 
- **244 unique**  NLU components 
- **176 unique** NLP models and algorithms 
- **68 unique** NLP pipelines consisting of composed NLP models 



## Classifiers trained on many different different datasets 
Choose the right tool for the right task! Whether you analyze movies or twitter, NLU has the right model for you! 

- trec6 classifier 
- trec10 classifier 
- spam classifier 
- fake news classifier 
- emotion classifier 
- cyberbullying classifier 
- sarcasm classifier 
- sentiment classifier for movies 
- IMDB Movie Sentiment classifier 
- Twitter sentiment classifier 
- NER pretrained on ONTO notes 
- NER trainer on CONLL 
- Language classifier for 20 languages on the wiki 20 lang dataset. 

## Utilities for the Data Science NLU applications 
Working with text data can sometimes be quite a dirty Job. NLU helps you keep your hands clean by providing lots of components that take away data engineering intensive tasks. 

- Datetime Matcher
- Pattern Matcher
- Chunk Matcher
- Phrases Matcher
- Stopword Cleaners
- Pattern Cleaners
- Slang Cleaner 

## Where can I see NLUs entire offer? 
Checkout the [NLU Namespace](https://nlu.johnsnowlabs.com/docs/en/namespace) for everything that NLU has to offer! 



## Supported Data Types
- Pandas DataFrame and Series
- Spark DataFrames
- Modin with Ray backend
- Modin with Dask backend
- Numpy arrays
- Strings and lists of strings 


# Need help? 
- [Ping us on Slack](https://spark-nlp.slack.com/archives/C0196BQCDPY) 
- [Post an issue on Github](https://github.com/JohnSnowLabs/nlu/issues)

