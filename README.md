
# NLU : State of the Art Natural Language understanding  
John Snow Labs NLU provides state of the art algorithms for NLP&NLU with hundreds of pretrained models in 59 languages.  It enables swift and simple development and research with its powerful Pythonic and Keras inspired API.  NLU's focus lies in providing the latest and greatest results of NLP&NLU research with **1 Line of code at most**  
It is powerd by John Snow Labs powerful Spark NLP library.  
  
## NLU in action  
![NLU in action](http://ckl-it.de/wp-content/uploads/2020/08/NLU_IN_ACTION_high_qual.gif)
  
  
  
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
  
## Loading and predict with  <u>multiple models</u> in 1 line  
  
Get 6 different embeddings in 1 line and use them for downstream datas cience tasks!  
  
```python  
nlu.load('bert elmo albert xlnet glove use').predict('I love NLU! <3')  
```  
  
## What kind of models does NLU provides?  
NLU provides everything a data scientist might want to wish for in one line of code!   
- The 14+ of the latest NLP embeddings ( BERT, ELMO, ALBERT, XLNET, GLOVE, BIOBERT, USE) and different varaitions of them  
- Generation of Sentence, Chunk and Document from these embeddings  
- Language Classification of 20 languages  
- 36 pretrained NER models  
- Part of Speech (POS) models for 34  languages  
- Lemmatizers models for 34  languages  
- Sentiment models for 5 categories  
- Labeld and Unlabled Dependency parsing  
- Spell Checking  
- Stopword removers for 41  languages  
- Classifiesr for 12 different problems  
- **244 unique**  NLU components  
- **176 uniqe** NLP models and algorithns  
- **68 unique** NLP pipelines consisting of composed NLP models  
  
  
  
## Classifiers trained on many different different datasets  
Choose the right tool for the right task! Wether you analyise movies or twitter, NLU has the right model for you!  
  
- trec6 classifier  
- trec10 classifier  
- spam classifier  
- fakenews classifier  
- emotion classifier  
- cyberbullying classifier  
- sarcasm classifier  
- sentiment classifier for movies  
- IMDB Movie Sentiment classifier  
- Twitter sentiment classifier  
- NER pretrained on ONTO notes  
- NER trainer on CONLL  
- Language classifier for 20 langauges on the wiki 20 lang dataset.  
  
## Utilities for the Data Science NLU applications  
Working with text data can be sometimes quite a dirty Job. NLU helps you keep your hands clean by providing lots of components that take away data engineering intensive tasks.  
  
  
  
- Matchers  
 - Datetime Matcher - Pattern Matcher - Chunk Matcher - Phrases Matcher - Text cleaners - Stopword Cleaners - Pattern Cleaners - Slang Cleaner  
## Where can I see NLUs entire offer?  
Checkout the [NLU Namespace](https://nlu.johnsnowlabs.com/docs/en/namespace) for everything that NLU has to offer!  
  
  
  
### Support Pandas and Pandas Dataframes  
  
# Want to contribute?  
Use John Snow Labs model hub! Soon to be released  
  
# Need help?  
- Stack overflow  
- Github issues