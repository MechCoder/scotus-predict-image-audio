from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import glob
import pandas as pd
import numpy as np
import wave
import contextlib
import os
import string

exclude = set(string.punctuation)
mfccFrameCount=10
fileFeatureValues = np.array([]);
fileFeatureLongValues = np.array([]);
for path, subdirs, files in os.walk("unzippedAudioData1955To1988"):
    for name in files:
        if name.endswith(".wav"):
            #get docket out of file name
            subName = os.path.join(name[6:])
            subNameList = subName.split('_',1)
            docket = subNameList[0]
            
            #get lawyerName out of file name
            restName = subNameList[1]
            restNameList=restName.split('-')
            lawyerName = restNameList[-2]
            matchLawyerName = (''.join(ch for ch in lawyerName if ch not in exclude))
            matchLawyerName = matchLawyerName.lower()
            
            # get audio duration
            duration = 100
            audioFile = os.path.join(path, name)
            with contextlib.closing(wave.open(audioFile,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                #print(duration)

            #get constant number of mfcc features
            constant = duration/mfccFrameCount
            (rate,sig) = wav.read(audioFile)
            mfcc_feat = mfcc(sig,rate,winstep=constant,winlen=constant)
            #fbank_feat = logfbank(sig,rate)
            #print(fbank_feat.shape)
            if mfcc_feat.shape[0] > mfccFrameCount:
                mfcc_feat = mfcc_feat [0:mfccFrameCount,:]
            #print(mfcc_feat.shape)

            #combine all of the above
            for fbank_feat_row in mfcc_feat:
                fileFeatureValue = np.concatenate((np.array([docket,matchLawyerName]),fbank_feat_row))
                if(fileFeatureValues.shape[0] == 0):
                    fileFeatureValues=np.hstack((fileFeatureValues,fileFeatureValue))
                else:
                    fileFeatureValues=np.vstack((fileFeatureValues,fileFeatureValue))
            
            #combine all of the above in a long way
            mfcc_feat_row = mfcc_feat.flatten()
            fileFeatureValue = np.concatenate((np.array([docket,matchLawyerName]),mfcc_feat_row))
            if(fileFeatureLongValues.shape[0] == 0):
                fileFeatureLongValues=np.hstack((fileFeatureLongValues,fileFeatureValue))
            else:
                fileFeatureLongValues=np.vstack((fileFeatureLongValues,fileFeatureValue))

#get to file - 130 features in a row - Plan B
mfccColumns = np.array(list(range(1,131)))
fileFeatureColumns = np.concatenate ((np.array(["docket","lawyer_name"]) , mfccColumns))
fileFeatureDf=pd.DataFrame(fileFeatureLongValues, columns=fileFeatureColumns)
fileFeatureDf.to_csv("data/input/mfccFileFeatures_1955_1998.csv")

