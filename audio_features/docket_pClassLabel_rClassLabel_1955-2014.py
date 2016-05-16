
# coding: utf-8

# In[3]:

import pandas as pd
from model import *
partAudioData = pd.read_csv("restData/modifiedInput/docket_p_r_audioSplit.csv")
restAudioData = pd.read_csv("restData/modifiedInput/docket_p_r_audioCombined.csv")

for addedFeature in added_features:
    restAudioData[addedFeature][restAudioData[addedFeature] <= 0]=-1
    restAudioData[addedFeature][restAudioData[addedFeature] > 0]=1


partDockets = partAudioData['docket']


# In[4]:

partDocketList = partDockets.values


# In[5]:

disjointRestAudioData = restAudioData.loc[~restAudioData['docket'].isin(partDocketList)]


# In[6]:

completeAudioData = pd.concat((partAudioData,disjointRestAudioData))


# In[7]:

partAudioData.shape


# In[8]:

disjointRestAudioData.shape


# In[9]:

completeAudioData.shape


# In[10]:

completeAudioData.to_csv("data/modifiedInput/completeAudioDataClssification.csv")


# In[ ]:



