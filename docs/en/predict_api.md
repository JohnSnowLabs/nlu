---
layout: article
title: The NLU predict function
key: predict-api
permalink: /docs/en/predict_api
modify_date: "2019-05-16"
---

# Predict method Parameters 


## Output metadata
The NLU predict method has a boolean metdata parameter.     
When it is set to True, NLU will output the confidence and additional metadata for each prediction.
Its default value is False.

```python
nlu.load('lang').predict('What a wonderful day!')
```




## Output Level parameter
NLU defines 4 output levels for the generated predictions.     
The output levels define how granular the predictions and outputs of NLU will be.     
Depending on what downstream tasks NLU will be used for the output level should be adjusted.
 

1. Token level: Outputs one row for every token in the input. **One to many mapping.** 
2. Chunk level: Outputs onw row for every chunk in the input. **One to many mapping.** 
3. Sentence level: Outputs one row for every sentence the input. **One to many mapping.**
4. Document level output: Outputs one row for every document in the input. **One to one mapping.**

NLU will try to infer the most useful output level automatically if a output level is not specified.     
The inferred output level will usually defined the last element of the pipeline.      


#### Document output level example
Every row in the input dataframe will be mapped to **one row** in the output dataframe.

```python
# NLU outputs 1 row for 1 input document
nlu.load('sentiment').predict(['I love data science! It is so much fun! It can also be quite helpful to people.', 'I love the city New-York'], output_level='document')
```

{:.steelBlueCols}
|document |	id | 	checked | 	sentiment_confidence | 	sentiment | 
|---------|-----|-----------|------------------------|------------|
|I love data science! It is so much fun! It can...	 | 0 | 	[I, love, data, science, !, It, is, so, much, ... ] |	[0.7540000081062317, 0.6121000051498413, 0.489... ] |	[positive, positive, positive]
|I love the city New-York	| 1 | 	[I, love, the, city, New-York] |	[0.7342000007629395]	 | [positive]


#### Sentence output level example

Every sentence in each row becomes a new row in the output dataframe.

```python
import nlu
# NLU will detect the 2 sentences and output 2 rows, one for each of the sentences.
nlu.load('sentiment').predict(['I love data science! It is so much fun! It can also be quite helpful to people.', 'I love the city New-York'], output_level='sentence')
```

{:.steelBlueCols}
|sentence |	sentiment_confidence | 	sentiment | 	id | 	checked | 
|---------|-----|-----------|------------------------|------------|
|I love data science!	                    | [0.7540]	| positive	|0	| [I, love, data, science, !, It, is, so, much, ...]|
|It is so much fun!	                         |[0.6121]	| positive	|0	| [I, love, data, science, !, It, is, so, much, ...]|
|It can also be quite helpful to people.	| [0.4895]	| positive	|0	| [I, love, data, science, !, It, is, so, much, ...] |
|I love the city New-York	                |[0.7342]	| positive	| 1	| [I, love, the, city, New-York] |


#### Token output level example

Every token in each input row becomes a new row in the output dataframe.

```python
# Every token in our sentence will become a row 
nlu.load('sentiment').predict(['I love data science! It is so much fun! It can also be quite helpful to people.', 'I love the city New-York'], output_level='token')
```

{:.steelBlueCols}
|	token	| checked	| id	| sentiment_confidence	|sentiment|
|---------|------|------|-------------------|----------------|
|I |	I | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489...] |	[positive, positive, positive] |
|love |	love | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489...] |	[positive, positive, positive] |
|data |	data | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489...] |	[positive, positive, positive] |
|science |	science | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ] | 	[positive, positive, positive] |
|! |	! | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|It |	It | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|is |	is | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|so |	so | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|much |	much | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|fun |	fun | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|! |	! | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|It |	It | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|can |	can | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|also |	also | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|be |	be | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]|	[positive, positive, positive] |
|quite |	quite | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ] |	[positive, positive, positive] |
|helpful |	helpful | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489...] |	[positive, positive, positive] |
|to |	to | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]| 	            [positive, positive, positive]| 
|people |	people | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489... ]| 	    [positive, positive, positive] | 
|. |	. | 	0 | 	[0.7540000081062317, 0.6121000051498413, 0.489...]|	    [positive, positive, positive] |
|I |	I | 	1 | 	[0.7342000007629395]|	[positive] |
|love |	love | 	1 | 	[0.7342000007629395]|	[positive]|
|the |	the | 	1 | 	[0.7342000007629395]|	[positive]|
|city |	city | 	1 | 	[0.7342000007629395] |	[positive]|
|New-York	| New-York	| 1	| [0.7342000007629395] | 	[positive]| 




#### Chunk output level example

Every chunk in each input row becomes a new row in the output datafarme

```python
# 'New York' is a Chunk. A chunk is an object that consists of multiple tokens but its not a sentence.
nlu.load('sentiment').predict(['I love data science! It is so much fun! It can also be quite helpful to people.', 'I love the city New-York'], output_level='chunk')
```








### Output positions parameter
By setting *output_positions=True*, the Dataframe generated by NLU will contain additional columns which describe the beginning and end of each feature inside of the original document.
These additional *_begining* and *_end* columns let you infer the piece of the original input string that has been used to generate the output.

- If the NLU output level is set to a **different output level** than some features outputlevel, the resulting features will be inside of lists
- If the NLU output level is set to the **same output level** as some feature, the generated positional features will be single integers
- positional :
 - For token based components the positional features referer to the beginning and the end of the token inside of the original document the text originates from.
 - For sentence based components like sentence embeddings and different sentence classifiers the output of positional will describe the beginning and the end of the sentence that was used to generate the output.


```python
nlu.load('sentiment').predict('I love data science!', output_level='token', output_positions=True)
```

{:.steelBlueCols}
|checked | 	checked_begin | 	checked_end | 	token | 	id | 	document_begin | 	document_end | 	sentence_begin | 	sentence_end |	sentiment_confidence | 	sentiment_begin| 	sentiment_end | 	sentiment | 
|-------|-----------------|-----------------|---------|---------|------------------|-----------------|-----------------|-----------------|------------------------|-----------------|
|I|	        0|	0|	I |	        0|         	[0] |	[78] | 	[0, 21, 40]	|[19, 38, 78] | 	[0.7540000081062317, 0.6121000051498413, 0.489...] | 	[0, 21, 40] | 	[19, 38, 78] | 	[positive, positive, positive]  | 
|love|	    2|	5|	love |	    0|         	[0] |	[78] | 	[0, 21, 40]	|[19, 38, 78] | 	[0.7540000081062317, 0.6121000051498413, 0.489...] | 	[0, 21, 40] | 	[19, 38, 78] | 	[positive, positive, positive]  | 
|data|	    7|	10|	data |	    0|         	[0] |	[78] | 	[0, 21, 40]	|[19, 38, 78] | 	[0.7540000081062317, 0.6121000051498413, 0.489...] | 	[0, 21, 40] | 	[19, 38, 78] | 	[positive, positive, positive]  | 
|science|	12|	18|	science |	0|         	[0] |	[78] | 	[0, 21, 40]	|[19, 38, 78] | 	[0.7540000081062317, 0.6121000051498413, 0.489...] | 	[0, 21, 40] | 	[19, 38, 78] | 	[positive, positive, positive]  | 
|!|	        19|	19|	! |	        0|         	[0] |	[78] | 	[0, 21, 40]	|[19, 38, 78] | 	[0.7540000081062317, 0.6121000051498413, 0.489...] | 	[0, 21, 40] | 	[19, 38, 78] | 	[positive, positive, positive]  | 



## Row origin inference for one to many mappings
NLU will give every input row an ID.      
The ID is useful if one row is mapped to many rows during prediction.      
The new rows which are generated from the input row will all have the same ID as the original source row.     
I.e. if one sentence row gets split into many token rows,  each token row will have the same id as the sentence row.
<!---  In the future NLU will support user provided ID's or use pandas indexes.     -->

 


## NaN Handling
- NLU will convert every NaN value to a Python None variable which is reflected in the final dataframe
- If a column containls **only** NaN or None, NLU will drop these columns for the output df.


## Memory optimization reccomendations
Instead of passing your entire Pandas Dataframe to NLU you can pass only the columns which you need for later tasks.        
This saves memory and computation time and can be achieved like in the following example, which assumes lattitude and longtitude are irrelevant for later tasks.     

```python
import nlu
import pandas as pd
data = {
    'text': ['@CKL-IT NLU ROCKS!', '@MaziyarPanahi NLU is pretty cool', '@JohnSnowLabs Try out NLU!'],
    'tweet_location': ['Berlin', 'Paris', 'United States'],
    'tweet_lattitude' : ['52.55035', '48.858093', '40.689247'],
    'tweet_longtitude' : ['13.39139', '2.294694','-74.044502']
    }

    
text_df = pd.DataFrame(data)
nlu.load('sentiment').predict(text_df[['text','tweet_location']])
```





## Supported data types
NLU supports all of the common Python data types and formats 

### Single strings
```python
import nlu
nlu.load('sentiment').predict('This is just one string')
```

### Lists of strings
```python
import nlu
nlu.load('sentiment').predict(['This is an arrray', ' Of strings!'])
```


### Pandas Dataframe 

One column must be named text and of object/string type
**note** : Passing the entire dataframe with additional features to the predict() method is very memory intensive.            
It  is recommended to only pass the columns required for further downstream tasks to the predict() method.      


```python
import nlu
import pandas as pd
data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami']}
text_df = pd.DataFrame(data)
nlu.load('sentiment').predict(text_df)
```

### Pandas Series

One column must be named text and of object/string type      
**note** : This way is the most memory efficient way 

```python
import nlu
import pandas as pd
data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami']}
text_df = pd.DataFrame(data)
nlu.load('sentiment').predict(text_df['text'])
```

###  Spark Dataframe 

One column must be named text and of string type

```python
import nlu
import pandas as pd
data = {"text": ['This day sucks', 'I love this day', 'I dont like Sami']}
text_pdf = pd.DataFrame(data)
text_sdf = nlu.spark.createDataFrame(text_pdf)
nlu.load('sentiment').predict(text_sdf)
```

