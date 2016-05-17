# New supreme court model with audio and image features.

Every script has to be run from the root directory in the following way.
python3 new_model/build_scdb_model_x.py

Scripts here compare the old supreme court model to the new supreme court model.
Each script has a counterpart that has a ``_removed`` suffix which runs the same
model except that the added features are stripped. This makes sure that the
training data with and without adding the features are the same. This is a short
description of each script.

The following apply to all scripts that have the term **audio** in them.
Every audio transcript of a lawyer is rated by a mTurk worker. These ratings
include "Aggressive", "Attractive", "Confident", "Intelligent", "Masculine",
"Quality", "Trust" and "Win". For every case, 16 features are added, 
8 of which include averaging these ratings across the P side and the other 8
include averaging these ratings across the R side.

The following apply to all scripts that have the term **image** in them.
The image of the lawyer is rated on by predicting a trained ridge model. Look at
the sub directory image_featured for further details. These ratings include
Happy, Friendly, Caring, Unhappy, Sociable, Cold, Kind and Unemotional. Similar to
the audio features 16 features are added, 8 of which include averaging these ratings
across the P side and the other 8 include averaging these ratings across the R side.

Here is a short description of all the scripts.

* *build_scdb_model_audio_binarize.py*:
  ** Audio ratings: