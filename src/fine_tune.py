##############################################################################################################
## This is a script to fine-tune main hyperparameters of the random forest model for classification of
## samples based on mutations in any given gene of interest.
##############################################################################################################

# import required libraries
import pandas as pd
import timeit
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# timing the run-time
start_time = timeit.default_timer()

##########################################################
########## Read feature matrix and label vector ##########
##########################################################
print('Reading feature matrix and label vector ...')
print('')

X = pd.read_csv(snakemake.input.feature_matrix, delimiter = '\t', header=0, index_col=0)
y = pd.read_csv(snakemake.input.label_vector, delimiter = '\t', header=0, index_col=0)

#####################################
##### Find best hyperparameters #####
#####################################

# convert y to Series instead of DF by extracting label column
y = pd.Series(y.y)

# put 10% of data aside for final check
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1)

# set parameters
clf = RandomForestClassifier()
n_jobs = 40
#cv = 5
n_splits = 5
param_space = {
    'n_estimators': [25, 50, 100, 200],
    'max_samples': [0.2, 0.4, 0.6, 0.8, 0.99],
    'max_features': ['sqrt', 0.01, 0.03, 0.05],
    'max_depth': [10, 50, 100, 200],
    'min_samples_split': [2, 10, 25, 50],
    'min_samples_leaf': [1, 2, 10, 50]
}

# function to find the best hyperparameters using f1-score
def findHyperparam(X, y):
    grid = GridSearchCV(estimator=clf, param_grid=param_space, scoring='f1_macro', n_jobs=n_jobs, 
                        cv=StratifiedKFold(n_splits=n_splits, shuffle=True))

    grid_result = grid.fit(X, y)
    return grid_result

# run the above function to find best hyperparameters
grid_result = findHyperparam(X_train, y_train)

# extract best hyperparameter values and write them in a file
f = open(snakemake.output.best_hyper_param, 'w')
print('best score:', file=f)
print(grid_result.best_score_, file=f)
print('', file=f)
print('best params:', file=f)
print(grid_result.best_params_, file=f)
print('', file=f)

best_n_estimators = grid_result.best_params_.get('n_estimators')
best_max_depth = grid_result.best_params_.get('max_depth')
best_max_features = grid_result.best_params_.get('max_features')
best_max_samples = grid_result.best_params_.get('max_samples')
best_min_samples_leaf = grid_result.best_params_.get('min_samples_leaf')
best_min_samples_split = grid_result.best_params_.get('min_samples_split')

########################################################
##### Test the model with the best hyperparameters #####
########################################################

# train the model on 90% of samples with the best hyperparameters
clf = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth,
                             max_features=best_max_features, max_samples=best_max_samples, 
                             min_samples_leaf=best_min_samples_leaf, min_samples_split=best_min_samples_split, n_jobs=40)
clf.fit(X_train, y_train)

# record the training score
print('Test validation set:', file=f)
print(clf.score(X_train, y_train), file=f)

# test the performance on 10% of samples that were put aside at the beginning
sample_ids = X_test.index.values
rf_predictions = clf.predict(X_test)
rf_pred_df = pd.DataFrame({'p_id':sample_ids, 'status':y_test, 'predict':rf_predictions})

print(classification_report(rf_pred_df.status, rf_pred_df.predict), file=f)

# train the model on 90% of samples with the RF model default hyperparameters
clf2 = RandomForestClassifier(n_jobs=40)
clf2.fit(X_train, y_train)
print('Validation set results with clf default params:', file=f)
print(clf2.score(X_train, y_train), file=f)

# test the performance of the default model on 10% of samples that were put aside at the beginning
rf_predictions2 = clf2.predict(X_test)
rf_pred_df2 = pd.DataFrame({'p_id':sample_ids, 'status':y_test, 'predict':rf_predictions2})

print(classification_report(rf_pred_df2.status, rf_pred_df2.predict), file=f)
f.close()

print('Best hyperparameters are found and tested . . .')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
