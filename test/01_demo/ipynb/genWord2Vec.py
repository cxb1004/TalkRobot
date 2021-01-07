#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import gensim
from gensim.models import word2vec


# In[2]:


wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)


# In[3]:


sentences = word2vec.LineSentence("/data4T/share/jiangxinyang848/textClassifier/data/preProcess/wordEmbdiing.txt")
a = list(sentences)
len(a)


# In[4]:


model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)  
model.wv.save_word2vec_format("./word2Vec" + ".bin", binary=True) 

