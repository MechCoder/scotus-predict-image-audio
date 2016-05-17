
# coding: utf-8

# In[16]:

import pandas as pd
from model import *
audio_data = pd.read_csv("data/input/docket_p_r_audio_ratings_1955-2014.csv")
raw_data = pd.read_csv("data/input/remove_features_unknown.csv")
f_data = pd.read_csv("data/input/remove_data_unknown.csv")


# In[17]:

print(raw_data.shape)
print(f_data.shape)
audio_data['docket'] = audio_data['docket'].astype('str')
raw_data['docket'] = raw_data['docket'].astype('str')
f_data['docket'] = f_data['docket'].astype('str')


# In[18]:

raw_data = raw_data.merge(audio_data, on="docket", how="inner")
raw_data = raw_data.fillna(0.0)

f_data = f_data.merge(audio_data, on="docket", how="inner")
f_data = f_data.fillna(0.0)

print(raw_data.shape)
print(f_data.shape)


# In[19]:

for addedFeature in added_features:
    f_data[addedFeature][f_data[addedFeature] < 0]=-1
    f_data[addedFeature][f_data[addedFeature] >= 0]=1

for addedFeature in added_features:
    raw_data[addedFeature][raw_data[addedFeature] < 0] = -1
    raw_data[addedFeature][raw_data[addedFeature] >= 0] = 1
    
raw_data.to_csv("data/input/audio_binarize_with_raw_data.csv")
f_data.to_csv("data/input/audio_binarize_with_features.csv")


# In[ ]:



