
# coding: utf-8

# In[9]:

import pandas as pd
from model import *
audio_files = pd.read_csv("data/pilot6b_long_only_f_n_outcomes.csv")


# In[10]:

import pandas as pd

audio_files = pd.read_csv("data/pilot6b_long_only_f_n_outcomes.csv")
cols = ['zAggressive', 'zAttractive', 'zConfident','zIntel', 'zMasculine', 'zQuality', 'zTrust','zWin']
for col in cols:
    audio_files[col][audio_files[col] < 0]=-1
    audio_files[col][audio_files[col] >= 0]=1

index_P = audio_files["Petitioner"] == "p"
audio_files[cols] = audio_files[cols].fillna(0.0)
audio_P = audio_files[index_P]
audio_R = audio_files[~index_P]

audio_R_merge = audio_R.groupby("CaseNum")[cols].mean()
audio_R_merge.columns = ['RAggressive',
 'RAttractive',
 'RConfident',
 'RIntel',
 'RMasculine',
 'RQuality',
 'RTrust',
 'RWin']
audio_P_merge = audio_P.groupby("CaseNum")[cols].mean()
audio_P_merge.columns = ['PAggressive',
 'PAttractive',
 'PConfident',
 'PIntel',
 'PMasculine',
 'PQuality',
 'PTrust',
 'PWin']
audio_ratings = pd.concat([audio_P_merge,audio_R_merge,], axis=1)
audio_ratings.reset_index(inplace=True)
audio_ratings = audio_ratings.rename(columns={'index': 'docket'})

print(audio_ratings.columns)

def check(x):
    x_list = x.split('--')
    tmp = x_list[0]
    if len(tmp) == 1:
        x_list[0] = "0" + tmp
    return "-".join(x_list)

audio_ratings["docket"] = audio_ratings["docket"].apply(lambda x: check(x))

print (audio_ratings.columns)


# In[11]:

print (audio_ratings.columns)


# In[12]:

audio_ratings.to_csv("restData/modifiedInput/docket_p_r_audioSplit.csv")


# In[ ]:



