#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from bs4 import BeautifulSoup


# In[2]:


with open("/data4T/share/jiangxinyang848/textClassifier/data/unlabeledTrainData.tsv", "r") as f:
    unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]
    
with open("/data4T/share/jiangxinyang848/textClassifier/data/labeledTrainData.tsv", "r") as f:
    labeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 3]


# In[3]:


unlabel = pd.DataFrame(unlabeledTrain[1: ], columns=unlabeledTrain[0])
label = pd.DataFrame(labeledTrain[1: ], columns=labeledTrain[0])


# In[4]:


unlabel.head(5)


# In[5]:


label.head(5)


# In[6]:


def getRate(subject):
    splitList = subject[1:-1].split("_")
    return int(splitList[1])

label["rate"] = label["id"].apply(getRate)


# In[7]:


label.head(5)


# In[8]:


def cleanReview(subject):
    beau = BeautifulSoup(subject)
    newSubject = beau.get_text()
#     newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',', '').replace('.', '').replace('?', '').replace('(', '').replace(')', '')
    newSubject = newSubject.strip().split(" ")
    newSubject = [word.lower() for word in newSubject]
    newSubject = " ".join(newSubject)
    
    return newSubject
    
unlabel["review"] = unlabel["review"].apply(cleanReview)
label["review"] = label["review"].apply(cleanReview)


# In[9]:


label.head(5)


# In[10]:


newDf = pd.concat([unlabel["review"], label["review"]], axis=0) 


# In[11]:


newDf.to_csv("/data4T/share/jiangxinyang848/textClassifier/data/preProcess/wordEmbdiing.txt", index=False)


# In[11]:


newLabel = label[["review", "sentiment", "rate"]]
newLabel.to_csv("/data4T/share/jiangxinyang848/textClassifier/data/preProcess/labeledCharTrain.csv", index=False)

