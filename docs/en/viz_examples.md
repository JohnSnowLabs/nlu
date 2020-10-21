---
layout: docs
header: true
title: NLU Visualization Examples
key: viz-examples
permalink: /docs/en/viz_examples
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1">

<div class="h3-box" markdown="1">

NLU can do quite a lot of things in just one line.   
But imagine the things you could do in multiple lines, like visualizations!    
In this section we will demonstrate a few common NLU idioms for the data science lifecycle, especially for the data exploration phase.         

The most common two liner you will use in NLU is loading a classifier like *emotion* or *sentiment*
and then plotting the occurence of each predicted label .

An few examples for this are the following :


```python
emotion_df = nlu.load('sentiment').predict(df)
emotion_df['sentiment'].value_counts().plot.bar()
```

![Sentiment Counts](/assets/images/nlu/VizExamples/sentiment_counts.png)

```python
emotion_df = nlu.load('emotion').predict(df)
emotion_df['emotion'].value_counts().plot.bar()
```
![Category counts](/assets/images/nlu/VizExamples/category_counts.png)

Another simple idiom is to group by an arbitrary feature from the original dataset and then plot the counts four each group.

```python
emotion_df = nlu.load('sentiment').predict(df)
sentiment_df.groupby('source')['sentiment'].value_counts().plot.bar(figsize=(20,8))
```

![Sentiment Groupy ](/assets/images/nlu/VizExamples/sentiment_groupy.png)


```python
nlu_emotion_df = nlu.load('emotion').predict(df)
nlu_emotion_df.groupby('airline')['emotion'].value_counts().plot.bar(figsize=(20,8))
```

![Sentiment Groupy ](/assets/images/nlu/VizExamples/emotion_groupy.png)

</div>
