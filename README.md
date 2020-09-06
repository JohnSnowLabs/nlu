
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

- [Collab demo of all NLU features](https://colab.research.google.com/drive/1hJ6BiYXxfeDfDjsZu0ZI2TnOa9nrIxfI?usp=sharing)
- [Kaggle Twitter Airline Sentiment Analysis NLU demo](https://www.kaggle.com/kasimchristianloan/nlu-sentiment-airline-demo)
- [Kaggle Twitter Airline Emotion Analysis NLU demo](https://www.kaggle.com/kasimchristianloan/nlu-emotion-airline-demo)
- [Kaggle Twitter COVID Sentiment Analysis NLU demo](https://www.kaggle.com/kasimchristianloan/nlu-covid-sentiment-showcase)
- [Kaggle Twitter COVID Emotion Analysis nlu demo](https://www.kaggle.com/kasimchristianloan/nlu-covid-emotion-showcase)



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

