import pandas as pd
import os.path

cols = ['zAggressive', 'zAttractive', 'zConfident', 'zMasculine', 'zWin', 'zQuality', 'zIntel', 'zTrust']
audio_file_path = "data/input/pilot6b_long_only_f_n_outcomes.csv"
if not os.path.isfile(audio_file_path):
    raise ValueError(
        "We are bounded by a NDA agreement to not share this file. If you have a "
        "csv file that contains ratings of every case that has the following columns, "
        "CaseNum, %s, this should work just as well. We are sorry that we cannot make "
        "the script completely reproducible but we at least have a decent sense of humour."
        % cols)
audio_files = pd.read_csv(audio_file_path)

raw_data_path = "data/input/remove_features_unknown.csv"
if not os.path.isfile(raw_data_path):
    import zipfile
    out_path = "data/input"
    data_zip = open("data/input/data_remove.zip", "rb")
    z = zipfile.ZipFile(data_zip)
    for name in z.namelist():
        z.extract(name, out_path)
    z.close()

raw_data = pd.read_csv("data/input/remove_features_unknown.csv")
f_data = pd.read_csv("data/input/remove_data_unknown.csv")

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
raw_data.to_csv("data/input/merge_scdb_with_audio_features.csv")

f_data = f_data.merge(audio_ratings, on="docket", how="inner")
f_data.to_csv("data/input/merge_scdb_with_audio_data.csv")
