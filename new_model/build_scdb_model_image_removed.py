import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas
import numpy as np
import numpy
from model import *

raw_data = pd.read_csv("data/input/images_with_raw_data.csv")
raw_data.drop(['Unnamed: 0'], axis=1, inplace=True)

feature_df = pd.read_csv("data/input/images_with_features.csv")
for col in feature_df.columns:
    if col.startswith('Unnamed'):
        feature_df.drop(col, axis=1, inplace=True)
feature_df.drop(image_Features, axis=1, inplace=True)
feature_df.drop("docket", axis=1, inplace=True)
print(feature_df.columns)

# Output basic quantities for sample
print(pandas.DataFrame(raw_data["justice_outcome_disposition"].value_counts()))
print(pandas.DataFrame(raw_data["justice_outcome_disposition"].value_counts(normalize=True)))
imps = np.zeros(len(feature_df.columns))

# In[5]:

# Setup training time period
min_training_years = 5
term_range = np.unique(raw_data["term"])[5:]

# Setting growing random forest parameters
# Number of trees to grow per term
trees_per_term = 100

# Number of trees to begin with
initial_trees = min_training_years * trees_per_term

# Number of years between "forest fires"
reset_interval = 9999

# Setup model
m = None
term_count = 0

for term in term_range:
    # Diagnostic output
    print("Term: {0}".format(term))
    term_count += 1
    
    # Setup train and test periods
    train_index = (raw_data.loc[:, "term"] < term).values
    test_index = (raw_data.loc[:, "term"] == term).values
    
    # Setup train data
    feature_data_train = feature_df.loc[train_index, :]
    target_data_train = raw_data.loc[train_index, "justice_outcome_disposition"].astype(int).values

    # Setup test data
    feature_data_test = feature_df.loc[test_index, :]
    target_data_test = raw_data.loc[test_index, "justice_outcome_disposition"].astype(int).values
    
    # Check if we should rebuild the model based on changing natural court
    if term_count % reset_interval == 0:
        # "Forest fire;" grow a new forest from scratch
        print("Reset interval hit; rebuilding with {0} trees".format(initial_trees + (term_count * trees_per_term)))
        m = None
    else:
        # Check if the justice set has changed
        if set(raw_data.loc[raw_data.loc[:, "term"] == (term-1), "justice"].unique()) != set(raw_data.loc[raw_data.loc[:, "term"] == (term), "justice"].unique()):
            # natural Court change; trigger forest fire
            print("Natural court change; rebuilding with {0} trees".format(initial_trees + (term_count * trees_per_term)))
        
            m = None
                                              
    # Build or grow a model depending on initial/reset condition
    if not m:
        # Grow an initial forest
        m = RandomForestClassifier(n_estimators=initial_trees + (term_count * trees_per_term), 
                                   class_weight="balanced_subsample",
                                   warm_start=True, n_jobs=-1, random_state=0)
    else:
        # Grow the forest by increasing the number of trees (requires warm_start=True)
        m.set_params(n_estimators=initial_trees + (term_count * trees_per_term))

    # Fit the forest model
    m.fit(feature_data_train,
          target_data_train)
    imps += m.feature_importances_

    # Fit the "dummy" model
    d = DummyClassifier(strategy="most_frequent")
    d.fit(feature_data_train, target_data_train)
    
    # Perform forest predictions
    raw_data.loc[test_index, "rf_predicted"] = m.predict(feature_data_test)
    
    # Store scores per class
    scores = m.predict_proba(feature_data_test)
    raw_data.loc[test_index, "rf_predicted_score_affirm"] = scores[:, 0]
    raw_data.loc[test_index, "rf_predicted_score_reverse"] = scores[:, 1]
    
    # Store dummy predictions
    raw_data.loc[test_index, "dummy_predicted"] = d.predict(feature_data_test)


arg_indices = np.argsort(-imps)
print("top 30")
f_30 = feature_df.columns[arg_indices][:50]
i_30 = imps[arg_indices][:50]

for feature, imp in zip(f_30, i_30):
    print("|" + str(feature) + " " + "|" + str(imp) + "" + "|")

# In[6]:

# Evaluation range
evaluation_index = raw_data.loc[:, "term"].isin(term_range)
target_actual = raw_data.loc[evaluation_index, "justice_outcome_disposition"]
target_predicted = raw_data.loc[evaluation_index, "rf_predicted"]
target_dummy = raw_data.loc[evaluation_index, "dummy_predicted"]
raw_data.loc[:, "rf_correct"] = numpy.nan
raw_data.loc[:, "dummy_correct"] = numpy.nan
raw_data.loc[evaluation_index, "rf_correct"] = (target_actual == target_predicted).astype(float)
raw_data.loc[evaluation_index, "dummy_correct"] = (target_actual == target_dummy).astype(float)

# Compare model
print("RF model")
print("="*32)
print(classification_report(target_actual, target_predicted))
print(confusion_matrix(target_actual, target_predicted))
print(accuracy_score(target_actual, target_predicted))
print("="*32)
print("")

# Dummy model
print("Dummy model")
print("="*32)
print(classification_report(target_actual, target_dummy))
print(confusion_matrix(target_actual, target_dummy))
print(accuracy_score(target_actual, target_dummy))
print("="*32)
print("")


# In[9]:

# Feature importance
feature_importance_df = pandas.DataFrame(list(zip(feature_df.columns, m.feature_importances_)),
                                         columns=["feature", "importance"])
feature_importance_df.sort_values(["importance"], ascending=False).head(10)

 
rf_predicted_case = pandas.DataFrame(raw_data.loc[evaluation_index, :]    .groupby(["docketId"])["rf_predicted"]    .agg(lambda x: x.value_counts().index[0]))
rf_predicted_case.columns = ["rf_predicted_case"]

dummy_predicted_case = pandas.DataFrame(raw_data.loc[evaluation_index, :]    .groupby(["docketId"])["dummy_predicted"]    .agg(lambda x: x.value_counts().index[0]))
dummy_predicted_case.columns = ["dummy_predicted_case"]

# Set DFs
rf_predicted_case = raw_data[["docketId", "case_outcome_disposition", "rf_predicted"]].join(rf_predicted_case, on="docketId")
dumy_predicted_case = raw_data[["docketId", "dummy_predicted"]].join(dummy_predicted_case, on="docketId")

raw_data.loc[:, "rf_predicted_case"] = rf_predicted_case
raw_data.loc[:, "dummy_predicted_case"] = dumy_predicted_case


# In[80]:

# Output case distribution
case_outcomes = raw_data.groupby(["docketId"])["case_outcome_disposition"].agg(lambda x: x.mode())
case_outcomes = case_outcomes.apply(lambda x: int(x) if type(x) in [numpy.float64] else None)
print(case_outcomes.value_counts())
print(case_outcomes.value_counts(normalize=True))


# In[81]:

# Output comparison
# Evaluation range
evaluation_index = raw_data.loc[:, "term"].isin(term_range) & -raw_data.loc[:, "case_outcome_disposition"].isnull()
target_actual = raw_data.loc[evaluation_index, "case_outcome_disposition"]
target_predicted = raw_data.loc[evaluation_index, "rf_predicted_case"]
target_dummy = raw_data.loc[evaluation_index, "dummy_predicted_case"]

raw_data.loc[:, "rf_correct_case"] = numpy.nan
raw_data.loc[:, "dummy_correct_case"] = numpy.nan
raw_data.loc[evaluation_index, "rf_correct_case"] = (target_actual == target_predicted).astype(float)
raw_data.loc[evaluation_index, "dummy_correct_case"] = (target_actual == target_dummy).astype(float)
case_classification_data = raw_data.loc[evaluation_index, ["docketId", "term", 
                                                           "case_outcome_disposition", "rf_predicted_case", "dummy_predicted_case",
                                                           "rf_correct_case", "dummy_correct_case"]
                                                          ].copy().drop_duplicates()

# Compare model
print("RF model")
print("="*32)
print(classification_report(case_classification_data["case_outcome_disposition"],
                                            case_classification_data["rf_predicted_case"]))
print(confusion_matrix(case_classification_data["case_outcome_disposition"],
                                            case_classification_data["rf_predicted_case"]))
print(accuracy_score(case_classification_data["case_outcome_disposition"],
                                            case_classification_data["rf_predicted_case"]))
print("="*32)
print("")

# Dummy model
print("Dummy model")
print("="*32)
print(classification_report(case_classification_data["case_outcome_disposition"],
                                            case_classification_data["dummy_predicted_case"]))
print(confusion_matrix(case_classification_data["case_outcome_disposition"],
                                            case_classification_data["dummy_predicted_case"]))
print(accuracy_score(case_classification_data["case_outcome_disposition"],
                                            case_classification_data["dummy_predicted_case"]))
print("="*32)
print("")
