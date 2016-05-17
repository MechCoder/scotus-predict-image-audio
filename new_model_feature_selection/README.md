This scripts in this directory are very similar to those in **new_model** except that
it performs a random forest based feature selection. The algorithm can be described as follows

We needed to provide init_features_num, increment_features_num, and limit_features.
For our analysis we set these values to 30, 10 and 200.For our analysis we set these values to 30, 10 and 200.

1. Fit the random forest classifier on the data.
2. Extract the feature importances from the “feature_importances_” attribute.
3. Set n_features=init_features_num.
4. Select the top n_features as described by 2.
5. Do 1 and calculate the mean accuracy score.
6. Increment n_features by increment_features_num and go to 4 again till n_features > limit_features.
