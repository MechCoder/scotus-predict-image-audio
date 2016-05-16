
# coding: utf-8

# In[55]:

import pandas as pd
from model import *
df = pd.DataFrame({'a': [0, -1, 2], 'b': [-3, 2, 1], 'c': [3, -1, 2]})


# In[62]:

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from model import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier

columns = ['zAggressive', 'zAttractive', 'zConfident','zIntel', 'zMasculine', 'zQuality', 'zTrust','zWin']
mfcc_audio = pd.read_csv("data/output/mfcc_with_ratings.csv")
mfcc_audio.drop(["Unnamed: 0"], inplace=True, axis=1)

mfcc_audio[columns] = (mfcc_audio[columns] >= 0.0).astype(np.int32)

audios = np.unique(mfcc_audio["Audio"])
train_audio, test_audio = train_test_split(
    audios, train_size=0.7, test_size=0.3, random_state=0)

X_train = mfcc_audio[mfcc_audio["Audio"].isin(train_audio)]
X_test = mfcc_audio[mfcc_audio["Audio"].isin(test_audio)]
y_train = X_train[columns]
y_test = X_test[columns]

X_train.drop(columns + ["Audio"], inplace=True, axis=1)
X_test.drop(columns + ["Audio"], inplace=True, axis=1)

mor = MultiOutputClassifier(
    RandomForestClassifier(random_state=0, n_estimators=1000), n_jobs=-1)
mor.fit(X_train, y_train)
mor_pred = mor.predict(X_test)

dummy = DummyClassifier()
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

estimators = mor.estimators_

for i, col in enumerate(columns):

    true = y_test[col]
    pred = mor_pred[:, i]
    d_p = dummy_pred[:, i]

    print(col)
    print("accuracy score")
    score = accuracy_score(true, pred)
    print(score)


# In[63]:

rawAudioPR1955To1998 = pd.read_csv("restData/modifiedInput/mfcc_1955_p_r.csv")


# In[64]:

added_features


# In[65]:

raw_cols = (rawAudioPR1955To1998.columns.values).tolist()


# In[66]:

mfcc_p_cols = raw_cols[2:132]


# In[67]:

mfcc_r_cols = raw_cols[133:263]


# In[68]:

mfcc_p_df =  rawAudioPR1955To1998[mfcc_p_cols]


# In[69]:

mfcc_r_df = rawAudioPR1955To1998[mfcc_r_cols]


# In[70]:

print (mfcc_p_df.shape)
print (mfcc_r_df.shape)


# In[71]:

mor_Pred_m_cols = mor.predict(mfcc_p_df)


# In[72]:

mor_Pred_m_cols.shape


# In[73]:

mor_Pred_r_cols = mor.predict(mfcc_r_df)


# In[74]:

print (mor_Pred_r_cols.shape)


# In[75]:

docketcol = rawAudioPR1955To1998['docket']


# In[76]:

p_r_combined = np.concatenate((mor_Pred_m_cols,mor_Pred_r_cols),axis=1)
#p_r_docket_combined = pd.concat((docketcol,p_r_combined),axis=1)


# In[79]:

p_r_combined_df = pd.DataFrame(p_r_combined, columns=added_features)


# In[80]:

p_r_docket_combined = pd.concat((docketcol,p_r_combined_df),axis=1)


# In[81]:

p_r_docket_combined.shape


# In[82]:

p_r_docket_combined.columns


# In[84]:

p_r_docket_combined.to_csv("restData/modifiedInput/docket_p_r_audioCombined.csv")


# In[ ]:

audio_files = pd.read_csv("data/pilot6b_long_only_f_n_outcomes.csv")

