### Steps to generate Audio Features

Dependency on https://github.com/jameslyons/python_speech_features

1. Joining the continuous ranged voice trait features (z-scores) to the original model.
  * Because there can be multiple lawyers on the petitioner or the respondent side, we average ratings for lawyers on each side.
  * Also, Because there can be multiple mTurk workers rating the same audio clip we average the z-scores

  Run scotus_data_with_zscore_of_audio_labels.py
  
  Input : 
   * data/input/pilot6b_long_only_f_n_outcomes.csv
   * data/input/remove_data_unknown.csv
   * data/input/remove_features_unknown.csv
  
  Output : 
   * data/input/audio_cont_with_raw_data.csv
   * data/input/audio_cont_with_features.csv 

2. Every audio clip of lawyer’s opening statements from 1955 - 1998 was processed into fixed number of frames i.e ‘n’ and we obtained the 13 Mel-frequency Cepstral Coefficients (MFCC) of these frames.
  * We vectorized the matrix of every audio clip, thus obtaining “n*13” length vectors for every audio clip from 1955 - 2014.
  * In our case n=10
  * We even do string processing on file names of audio clips in 1955-1998 period to extract the lawyer name and the docket; so that we could later identify if the lawyer is a petitioner or a respondent corresponding to (step 5)
   
  Run mfcc_generator_1955-1998.py

  Input
   * Unzipped Audio Files from 1955 to 1998 
  
  Output
   * data/input/mfccFileFeatures_1955_1998.csv

3. Every audio clip of lawyer’s opening statements from 1998 - 2014 was processed into fixed number of frames i.e ‘n’ and we obtained the 13 Mel-frequency Cepstral Coefficients (MFCC) of each these frames for each of these files.
  * Thus we have “n” rows for every audio file corresponding to “n” data frames of every audio file.
  * In our case n=10
  
  RUN mfcc_generator_1998-2014.py
  
  Input
   * Unzipped Audio Files between 1998 and 2014

  Output
   * data/input/mfccFileFeatures_1998_2014.csv

4. Join MFCC of every audio clip with mTurk worker rating (z-score)
  a. We  vectorize the MFCC matrix of every audio clip, thus obtaining “n*13” length vectors for every audio clip from 1998 - 2014. (completing step 3)
  b. Because there could be multiple mTurk raters for the same audio clip; we average the ratings (z-score of that audio clip by that mTurk rater)
  Now corresponding to every audio clip we have MFCC and mTurk worker ratings(averaged z-scores)

  Run merge_mfcc_with_pilot.py
  
  Input
   * data/input/mfccFileFeatures_1998_2014.csv
   * data/input/pilot6b_long_only_f_n_outcomes.csv
  
  Output
   * data/input/mfcc_with_ratings1998-2014.csv
  
5. We do string processing lawyer names in the file advocates.csv for easy pattern matching corresponding to (step 2)

  Run modify_advocates.csv
  
  Input
   * data/input/advocates.csv

  Output
   * data/input/modifiedAdvocates.csv 
  
6. Joining modifiedAdvocates.csv and mfccFileFeatures_1955_1998.csv based on lawyer’s name. (modifiedAdvocates.csv contains information about the docket and also information regarding if the lawyer was a petitioner or a respondent)
  
  Input
   * data/input/modifiedAdvocates.csv
   * data/input/mfccFileFeatures_1955_1998.csv
  
  Output
   * data/input/advocates_mfcc_merged_p_r_1955-1998.csv
  
7. We remove duplicates from advocatesMfccMerged_1955-1998.csv and get into format (docket, pMFCC, rMFCC) as a row where pMFCC and rMFCC are the 130 MFCC features

  Input
   * data/input/advocates_mfcc_merged_p_r_1955-1998.csv

  Output
   * data/input/mfcc_1955-1998_p_r.csv
  
8. 
  a. We binarized the z-scores of the data obtained from (step 4.) by setting a threshold: if a z-score was positive, we replaced it with 1, if it was negative we replaced it with -1. 
  b. We trained and validated a random forest classifier model from the data in the period 1998 - 2014 based on MFCC and human perception binarized score 
  c. We used this trained binary trait prediction model to predict traits of lawyer’s voices for the period 1955 - 1998. 
  
  Run docket_predict_pClass_label_rClassLabel_1955-1998.py
  
  Input
   * data/input/mfcc_with_ratings1998-2014.csv ………to train and validate the model
   * data/input/mfcc_1955-1998_p_r.csv ………to predict unknown binarized voice trait ratings

  Output
   * data/input/docket_p_r_audio_ratings_1955_1998.csv
  
9. 
  a. We binarize the z-scores by setting a threshold: if a z-score was positive, we replaced it with 1, if it was negative we replaced it with -1.
 
  b. Because there are multiple mTurk worker who have rated the same audio clip, we average their z-scores and then binarize them.
  
  Run docket_pClassLabel_rClassLabel_1998-2014.py
  Input
   * data/input/pilot6b_long_only_f_n_outcomes.csv
  
  Output
   * data/input/docket_p_r_audio_ratings_1998_2014.csv

10. Basically output of 8 and 9 are in same format but rows belong to different period in time….so we simply stack them vertically

  Run docket_pClassLabel_rClassLabel_1955-2014.py

  Input
   * data/input/docket_p_r_audio_ratings_1998_2014.csv
   * data/input/docket_p_r_audio_ratings_1955_1998.csv
  
  Output
   * data/input/docket_p_r_audio_ratings_1955-2014.csv
  
11. Joining the binarized voice trait features (from 10) to the original model based on “docket” 

 Run ScotusData_docket_pClassLabel_rClassLabel.py

 Input
  * data/input/docket_p_r_audio_ratings_1955-2014.csv
  * data/input/remove_data_unknown.csv remove_features_unknown.csv

 Output
  * data/input/audio_binarize_with_raw_data.csv
  * data/input/audio_binarize_with_features.csv
  

###Go to new_model
