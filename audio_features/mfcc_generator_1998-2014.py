from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import glob
import pandas as pd
import numpy as np
import wave
import contextlib
#Task 1
##audio features ----- 
##create rows : KEY AS file name; VALUE AS mfcc features.....
##create rows : KEY AS file name; VALUE AS Docket.....
##
##concatenate

fileFeatureDict = {};
fileDocketDict = {};
fileFeatureValues = np.array([]);
fileDocketValues = np.array([]);
mfccFrameCount=10
audioFiles = glob.glob("male/*.wav")
for audioFile in audioFiles:
    duration = 100
    with contextlib.closing(wave.open(audioFile,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        #print(duration)
    constant = duration/mfccFrameCount
    (rate,sig) = wav.read(audioFile)
    mfcc_feat = mfcc(sig,rate,winstep=constant,winlen=constant)
    if mfcc_feat.shape[0] > mfccFrameCount:
        mfcc_feat = mfcc_feat [0:mfccFrameCount,:]
    for fbank_feat_row in mfcc_feat:
        fileFeatureValue = np.concatenate((np.array([audioFile]),fbank_feat_row))
        if(fileFeatureValues.shape[0] == 0):
            fileFeatureValues=np.hstack((fileFeatureValues,fileFeatureValue))
        else:
            fileFeatureValues=np.vstack((fileFeatureValues,fileFeatureValue))
            
audioFiles = glob.glob("female/*.wav")
for audioFile in audioFiles:
    duration = 100
    with contextlib.closing(wave.open(audioFile,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        #print(duration)
    constant = duration/mfccFrameCount
    (rate,sig) = wav.read(audioFile)
    mfcc_feat = mfcc(sig,rate,winstep=constant,winlen=constant)
    if mfcc_feat.shape[0] > mfccFrameCount:
        mfcc_feat = mfcc_feat [0:mfccFrameCount,:]
    #print(mfcc_feat.shape)
    for fbank_feat_row in mfcc_feat:
        fileFeatureValue = np.concatenate((np.array([audioFile]),fbank_feat_row))
        if(fileFeatureValues.shape[0] == 0):
            fileFeatureValues=np.hstack((fileFeatureValues,fileFeatureValue))
        else:
            fileFeatureValues=np.vstack((fileFeatureValues,fileFeatureValue))

mfccColumns = np.array(list(range(1,14)))
fileFeatureColumns = np.concatenate ((np.array(["Audio"]) , mfccColumns))
print(fileFeatureValues.shape)

##convert fileFeatureValues and fileDocketValues to data frames
fileFeatureDf=pd.DataFrame(fileFeatureValues, columns=fileFeatureColumns)
fileFeatureDf.to_csv("data/input/mfccFileFeatures_1998_2014.csv")
