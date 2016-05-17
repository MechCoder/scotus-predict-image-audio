## Image feature generation pipeline:

1. Train trait prediction models using **image_HOG_trait_regression.ipynb**: this script trains facial feature prediction models on Bainbridge's face data applying HOG processing. It stores the models in the "../data/image_data/tmp" folder. 

2. Extract lawyer faces from lawyer images
 * **get_img_and_pdfs.ipynb**: this script takes a batch of results of the mturk lawyer image gathering task (batches are stored in "../data/image_data/mturk") and it downloads and saves all the images to the "lawyer_images/" directory.
 * **face_extractor.ipynb**: this script extracts a face from an image. It will prompt you to provide a path to the relevant image files ("../data/image_data/img"). By default this script stores the output faces in the directory "lawyer_faces/".

3. Predict Lawyer face trait ratings with **predict_lawyer_traits.ipynb**: this script predicts trait ratings for a set of face images (by default it looks for the face images in the "lawyer_faces/" folder), and it saves the ratings for each lawyer as "lawyer_trait_predictions.csv"

4. Merge all available advocate-case data (1946 - 2014): **advocate_extractor.ipynb** gathers together all available advocate data and joins it into one data set. This data contains lawyer names and petitioner vs. respondent flag for lawyers in a given case.

5. Aggregate lawyer ratings by side (petitioner/respondent): **cases_join_traits.ipynb** takes the predicted lawyer ratings from section 4, and it averages the ratings across the petitioner or respondent side for each case. It produces an output file which contains petitioner and respondent side ratings for each case where at least one is available. This final output is incorportated as features in the SCOTUS model.
