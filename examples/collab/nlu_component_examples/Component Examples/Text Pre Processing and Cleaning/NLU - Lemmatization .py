#!/usr/bin/env python
# coding: utf-8

# # Lemmatization with NLU 
# 
# Lemmatizing returns the base form, the so called lemma of every token in the input data.    
# 
# I. e. 'He was hungry' becomes 'He be hungry'
# 
# The Lemmatizer works by operating on a dictionary and taking context into account. This lets the Lemmatizer dervie a different base word for for a word in two different contexts which depends on the Part of Speech tags. 
# 
# 
# 
# This is the main difference  to Stemming, which solves the same problem by applying a heuristic process that removes the end of words.
# 
# 
# # 1. Install Java and NLU

# In[1]:



import os
get_ipython().system(' apt-get update -qq > /dev/null   ')
# Install java
get_ipython().system(' apt-get install -y openjdk-8-jdk-headless -qq > /dev/null')
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
get_ipython().system(' pip install nlu  > /dev/null    ')


# ## 2. Load Model and lemmatize sample string

# In[2]:


import nlu
pipe = nlu.load('en.lemma')
pipe.predict('He was suprised by the diversity of NLU')


# # 3. Get one row per lemmatized token by setting outputlevel to token.    
# This lets us compare what the original token was and what it was lemmatized to to. 

# In[3]:


pipe.predict('He was suprised by the diversity of NLU', output_level='token')


# # 4. Checkout the Lemma models NLU has to offer for other languages than English!

# In[4]:


nlu.print_all_model_kinds_for_action('lemma')


# ## 4.1 Let's try German lematization!

# In[5]:


nlu.load('de.lemma').predict("Er war von der Vielfältigkeit des NLU Packets begeistert",output_level='token')

