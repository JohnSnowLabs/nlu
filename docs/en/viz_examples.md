---
layout: docs
header: true
seotitle: NLU | John Snow Labs
title: NLU Visualization Examples
key: viz-examples
permalink: /docs/en/viz_examples
modify_date: "2019-05-16"
---

<div class="main-docs" markdown="1"><div class="h3-box" markdown="1">

NLU can do quite a lot of things in just one line.   
But imagine the things you could do in multiple lines, like visualizations!    
In this section we will demonstrate a few common NLU idioms for the data science lifecycle, especially for the data exploration phase.         

</div><div class="h3-box" markdown="1">

{:.h2-select}
## Visualizations using nlu.load().viz()
You can use the build in visualization module on any pipeline or model returned by `nlu.load()`.
Simply call `viz()` and NLU will try to deduct a applicable visualization.    
Alternatively, you can also manually specify, which visualization you want to invoke.   
These visualizations are provided via [Spark-NLP-Display package](https://nlp.johnsnowlabs.com/docs/en/display) which NLU will try to automatically install when calling the .viz() method.

![NER visualization](/assets/images/nlu/VizExamples/viz_module/cheat_sheet.png)

- Named Entity Recognizers 
- Medical Named Entity Recognizers
- Dependency parser relationships which labels and part of speech tags
- Entity resolution for sentences and chunks
- Assertion of entity statuses

See the [visualization tutorial](https://github.com/JohnSnowLabs/nlu/blob/master/examples/colab/visualization/NLU_visualizations_tutorial.ipynb) notebook for more info.

</div><div class="h3-box" markdown="1">

## NER visualization
Applicable to any of the [100+ NER models! See here for an overview](https://nlp.johnsnowlabs.com/models?task=Named+Entity+Recognition)
```python
nlu.load('ner').viz("Donald Trump from America and Angela Merkel from Germany don't share many oppinions.")
```
![NER visualization](/assets/images/nlu/VizExamples/viz_module/NER.png)

</div><div class="h3-box" markdown="1">

## Dependency tree visualization
Visualizes the structure of the labeled dependency tree and part of speech tags
```python
nlu.load('dep.typed').viz("Billy went to the mall")
```

![Dependency Tree visualization](/assets/images/nlu/VizExamples/viz_module/DEP.png)

```python
#Bigger Example
nlu.load('dep.typed').viz("Donald Trump from America and Angela Merkel from Germany don't share many oppinions but they both love John Snow Labs software")
```
![Dependency Tree visualization](/assets/images/nlu/VizExamples/viz_module/DEP_big.png)

</div><div class="h3-box" markdown="1">

## Assertion status visualization
Visualizes asserted statuses and entities.        
Applicable to any of the [10 + Assertion models! See here for an overview](https://nlp.johnsnowlabs.com/models?task=Assertion+Status)
```python
nlu.load('med_ner.clinical assert').viz("The MRI scan showed no signs of cancer in the left lung")
```


![Assert visualization](/assets/images/nlu/VizExamples/viz_module/assertion.png)

```python
#bigger example
data ='This is the case of a very pleasant 46-year-old Caucasian female, seen in clinic on 12/11/07 during which time MRI of the left shoulder showed no evidence of rotator cuff tear. She did have a previous MRI of the cervical spine that did show an osteophyte on the left C6-C7 level. Based on this, negative MRI of the shoulder, the patient was recommended to have anterior cervical discectomy with anterior interbody fusion at C6-C7 level. Operation, expected outcome, risks, and benefits were discussed with her. Risks include, but not exclusive of bleeding and infection, bleeding could be soft tissue bleeding, which may compromise airway and may result in return to the operating room emergently for evacuation of said hematoma. There is also the possibility of bleeding into the epidural space, which can compress the spinal cord and result in weakness and numbness of all four extremities as well as impairment of bowel and bladder function. However, the patient may develop deeper-seated infection, which may require return to the operating room. Should the infection be in the area of the spinal instrumentation, this will cause a dilemma since there might be a need to remove the spinal instrumentation and/or allograft. There is also the possibility of potential injury to the esophageus, the trachea, and the carotid artery. There is also the risks of stroke on the right cerebral circulation should an undiagnosed plaque be propelled from the right carotid. She understood all of these risks and agreed to have the procedure performed.'
nlu.load('med_ner.clinical assert').viz(data)
```
![Assert visualization](/assets/images/nlu/VizExamples/viz_module/assertion_big.png)

</div><div class="h3-box" markdown="1">

## Relationship between entities visualization
Visualizes the extracted entities between relationship.    
Applicable to any of the [20 + Relation Extractor models See here for an overview](https://nlp.johnsnowlabs.com/models?task=Relation+Extraction)
```python
nlu.load('med_ner.jsl.wip.clinical relation.temporal_events').viz('The patient developed cancer after a mercury poisoning in 1999 ')
```
![Entity Relation visualization](/assets/images/nlu/VizExamples/viz_module/relation.png)

```python
# bigger example
data = 'This is the case of a very pleasant 46-year-old Caucasian female, seen in clinic on 12/11/07 during which time MRI of the left shoulder showed no evidence of rotator cuff tear. She did have a previous MRI of the cervical spine that did show an osteophyte on the left C6-C7 level. Based on this, negative MRI of the shoulder, the patient was recommended to have anterior cervical discectomy with anterior interbody fusion at C6-C7 level. Operation, expected outcome, risks, and benefits were discussed with her. Risks include, but not exclusive of bleeding and infection, bleeding could be soft tissue bleeding, which may compromise airway and may result in return to the operating room emergently for evacuation of said hematoma. There is also the possibility of bleeding into the epidural space, which can compress the spinal cord and result in weakness and numbness of all four extremities as well as impairment of bowel and bladder function. However, the patient may develop deeper-seated infection, which may require return to the operating room. Should the infection be in the area of the spinal instrumentation, this will cause a dilemma since there might be a need to remove the spinal instrumentation and/or allograft. There is also the possibility of potential injury to the esophageus, the trachea, and the carotid artery. There is also the risks of stroke on the right cerebral circulation should an undiagnosed plaque be propelled from the right carotid. She understood all of these risks and agreed to have the procedure performed'
pipe = nlu.load('med_ner.jsl.wip.clinical relation.clinical').viz(data)
```
![Entity Relation visualization](/assets/images/nlu/VizExamples/viz_module/relation_big.png)

</div><div class="h3-box" markdown="1">

## Entity Resolution visualization for chunks
Visualizes resolutions of entities
Applicable to any of the [100+ Resolver models See here for an overview](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution)
```python
nlu.load('med_ner.jsl.wip.clinical resolve_chunk.rxnorm.in').viz("He took Prevacid 30 mg  daily")
```
![Chunk Resolution visualization](/assets/images/nlu/VizExamples/viz_module/resolve_chunk.png)

```python
# bigger example
data = "This is an 82 - year-old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU ."
nlu.load('med_ner.jsl.wip.clinical resolve_chunk.rxnorm.in').viz(data)
```

![Chunk Resolution visualization](/assets/images/nlu/VizExamples/viz_module/resolve_chunk_big.png)

</div><div class="h3-box" markdown="1">

## Entity Resolution visualization for sentences
Visualizes resolutions of entities in sentences
Applicable to any of the [100+ Resolver models See here for an overview](https://nlp.johnsnowlabs.com/models?task=Entity+Resolution)
```python
nlu.load('med_ner.jsl.wip.clinical resolve.icd10cm').viz('She was diagnosed with a respiratory congestion')
```
![Sentence Resolution visualization](/assets/images/nlu/VizExamples/viz_module/resolve_sentence.png)

```python
# bigger example
data = 'The patient is a 5-month-old infant who presented initially on Monday with a cold, cough, and runny nose for 2 days. Mom states she had no fever. Her appetite was good but she was spitting up a lot. She had no difficulty breathing and her cough was described as dry and hacky. At that time, physical exam showed a right TM, which was red. Left TM was okay. She was fairly congested but looked happy and playful. She was started on Amoxil and Aldex and we told to recheck in 2 weeks to recheck her ear. Mom returned to clinic again today because she got much worse overnight. She was having difficulty breathing. She was much more congested and her appetite had decreased significantly today. She also spiked a temperature yesterday of 102.6 and always having trouble sleeping secondary to congestion'
nlu.load('med_ner.jsl.wip.clinical resolve.icd10cm').viz(data)
```
![Sentence Resolution visualization](/assets/images/nlu/VizExamples/viz_module/resolve_sentence_big.png)

</div><div class="h3-box" markdown="1">

## Configure visualizations
### Define custom colors for labels
Some entity and relation labels will be highlighted with a pre-defined color, which you [can find here](https://github.com/JohnSnowLabs/spark-nlp-display/tree/main/sparknlp_display/label_colors).    
For labels that have no color defined, a random color will be generated.     
You can define colors for labels manually, by specifying via the `viz_colors` parameter
and defining `hex color codes` in a dictionary that maps `labels` to `colors` .
```python
data = 'Dr. John Snow suggested that Fritz takes 5mg penicilin for his cough'
# Define custom colors for labels
viz_colors={'STRENGTH':'#800080', 'DRUG_BRANDNAME':'#77b5fe', 'GENDER':'#77ffe'}
nlu.load('med_ner.jsl.wip.clinical').viz(data,viz_colors =viz_colors)
```
![define colors labels](/assets/images/nlu/VizExamples/viz_module/define_colors.png)

</div><div class="h3-box" markdown="1">

### Filter entities that get highlighted
By default every entity class will be visualized.    
The `labels_to_viz` can be used to define a set of labels to highlight.       
Applicable for ner, resolution and assert.
```python
data = 'Dr. John Snow suggested that Fritz takes 5mg penicilin for his cough'
# Filter wich NER label to viz
labels_to_viz=['SYMPTOM']
nlu.load('med_ner.jsl.wip.clinical').viz(data,labels_to_viz=labels_to_viz)
```
![filter labels](/assets/images/nlu/VizExamples/viz_module/filter_labels.png)

</div><div class="h3-box" markdown="1">

{:.h2-select}
## Visualizations using Pandas
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


You can visualize a Keyword distribution generated by YAKE like this 
```python
keyword_predictions.explode('keywords').keywords.value_counts()[0:100].plot.bar(title='Top 100 Keywords in Stack Overflow Questions', figsize=(20,8))
```
![Category counts](/assets/images/nlu/VizExamples/keyword_distribution.png)

</div></div>