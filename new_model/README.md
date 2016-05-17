## New supreme court model with audio and image features.

Every script has to be run from the root directory in the following way.
python3 new_model/build_scdb_model_x.py

Scripts here compare the old supreme court model to the new supreme court model.
Each script has a counterpart that has a ``_removed`` suffix which runs the same
model except that the added features are stripped. This makes sure that the
training data with and without adding the features are the same.

The following apply to all scripts that have the term **audio** in them.
For every lawyer, we have 8 new features which include "Aggressive", "Attractive",
"Confident", "Intelligent", "Masculine", "Quality", "Trust" and "Win" which are
are obtained by rating the audio transcript of the lawyer. For every case we have
16 new features, 8 of which include the averaged ratings of all the lawyers across
the petitioner side and the other 8 include the averaged ratings of all the lawyers
across the respondant side.

The following apply to all scripts that have the term **image** in them.
For every lawyer, we have 8 new features which include happy, friendly, caring, unhappy
sociable, cold, Kind and Unemotional which are obtained by rating the image of the lawyer.
For every case we have 16 new features, 8 of which include the averaged ratings of
all the lawyers across the petitioner side and the other 8 include the averaged ratings of
all the lawyers across the respondant side.

For details on how these features were generated, you need to look at the **audio_features**
and **image_features** subdirectories.

Here is a short description of all the scripts.

* *build_scdb_model_audio_binarize.py*:
  Audio ratings are binarized such that when > greater than mean, a positive label is given
     and when < than mean, a negative label is given. These ratings are then added to
     the features of the original data.

* *build_scdb_model_audio_binarize_image.py*:
  Audio ratings are binarized such that when > greater than mean, a positive label is given
  and when < than mean, a negative label is given. These are addhesed with the audio ratings
  which are then added to the faetures of the original data.

* *build_scdb_model_audio_cont.py*:
  Append the audio ratings as they are to the original data.

* *build_scdb_model_audio_cont_image.py*:
  Append the audio ratings after merging with the continuous image ratings to the features of
  the original data.

* *build_scdb_model_audio_cont_image.py*:
  Append the continuous image ratings to the original data.
