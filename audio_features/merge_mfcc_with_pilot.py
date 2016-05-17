import pandas as pd
import numpy as np

added_columns = ['zAggressive', 'zAttractive', 'zConfident',
                 'zIntel', 'zMasculine', 'zQuality', 'zTrust', 'zWin']
columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']


mfcc = pd.read_csv("data/input/mfccFileFeatures_1955_1998.csv")
pilot = pd.read_csv("data/input/pilot6b_long_only_f_n_outcomes.csv")
pilot = pilot.fillna(0.0)
pilot_columns = pilot.groupby('Audio')[added_columns].mean().reset_index()

audio_files = np.unique(mfcc['Audio'])

audio_with_mfcc = []
for audio in audio_files:
    mfcc_features = list(np.asarray(mfcc[mfcc['Audio'] == audio][columns]).ravel())
    audio_with_mfcc.append([audio] + mfcc_features)

audio_with_mfcc = pd.DataFrame(audio_with_mfcc)
audio_with_mfcc.columns = ["Audio"] + [str(i) for i in range(130)]
audio_with_mfcc = pd.DataFrame(audio_with_mfcc)
audio_with_mfcc = audio_with_mfcc.merge(pilot_columns, how="inner", on="Audio")
audio_with_mfcc.to_csv("data/input/mfcc_with_ratings1998-2014.csv")
