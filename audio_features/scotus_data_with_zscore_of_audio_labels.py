import pandas as pd

audio_files = pd.read_csv("data/pilot6b_long_only_f_n_outcomes.csv")
raw_data = pd.read_csv("data/output/remove_features_unknown.csv")
f_data = pd.read_csv("data/output/remove_data_unknown.csv")

index_P = audio_files["Petitioner"] == "p"
cols = ['zAggressive', 'zAttractive', 'zConfident', 'zMasculine', 'zWin', 'zQuality', 'zIntel', 'zTrust']
audio_files[cols] = audio_files[cols].fillna(0.0)
audio_P = audio_files[index_P]
audio_R = audio_files[~index_P]

audio_R_merge = audio_R.groupby("CaseNum")[cols].mean()
audio_R_merge.columns = ['RAggressive', 'RAttractive', 'RConfident', 'RMasculine', 'RWin', 'RQuality', 'RIntel', 'RTrust']
audio_P_merge = audio_P.groupby("CaseNum")[cols].mean()
audio_P_merge.columns = ['PAggressive', 'PAttractive', 'PConfident', 'PMasculine', 'PWin', 'PQuality', 'PIntel', 'PTrust']

audio_ratings = pd.concat([audio_R_merge, audio_P_merge], axis=1)
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
raw_data = raw_data.merge(audio_ratings, on="docket", how="inner")
raw_data = raw_data.fillna(0.0)
raw_data.to_csv("data/output/merge_audio_features.csv")

f_data = f_data.merge(audio_ratings, on="docket", how="inner")
f_data.to_csv("data/output/merge_audio_data.csv")
