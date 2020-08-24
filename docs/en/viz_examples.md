---
layout: article
title: NLU Visualization Examples
key: viz-examples
permalink: /docs/en/viz_examples
modify_date: "2019-05-16"
---


NLU can do quite a lot of things in just one line.    
But imagine the things you could do in multiple lines, like visualizations!     
In this section we will demonstrate a few common NLU idoms for the data science lifecycle, especialy for the data exploration phase.          

The most common two liner you will use in NLU is loading a classifier like *emotion* or *sentiment* 
and then plotting the occurence of each predicted label .

An few exmples for this are the following : 


```python
emotion_df = nlu.load('sentiment').predict(df)
emotion_df['sentiment'].value_counts().plot.bar()
```

![Sentiment Counts](/assets/images/nlu/VizExamples/sentiment_counts.png)

```python
emotion_df = nlu.load('emotion').predict(df)
emotion_df['category'].value_counts().plot.bar()
```
![Category counts](/assets/images/nlu/VizExamples/category_counts.png)



Another simple idiom is to group by an arbitrary feature fromt he original dataset and then plot the counts four each group.

```python
emotion_df = nlu.load('sentiment').predict(df)
sentiment_df.groupby('source')['sentiment'].value_counts().plot.bar(figsize=(20,8))
```

![Sentiment Groupy ](/assets/images/nlu/VizExamples/sentiment_groupy.png)


```python
nlu_emotion_df = nlu.load('emotion').predict(df)
nlu_emotion_df.groupby('airline')['category'].value_counts().plot.bar(figsize=(20,8))
```

![Sentiment Groupy ](/assets/images/nlu/VizExamples/emotion_groupy.png)
