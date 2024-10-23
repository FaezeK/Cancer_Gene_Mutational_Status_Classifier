##############################################################################################################
## This is a script to test the model's performance on different combination of data types (SNVs/INDELs, 
## CNVs and SVs) to investiaget if CNV and SV data would improve the results. The model is ran in 30
## permutations for each setting.  
##############################################################################################################

# import required libraries
import pandas as pd
import numpy as np
import timeit
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# timing the run-time
start_time = timeit.default_timer()

######################################
########## Read input files ##########
######################################
print('Reading input files ...')
print('')

# read feature matrix and label vector
X = pd.read_csv(snakemake.input.feature_matrix, delimiter = '\t', header=0, index_col=0)
y = pd.read_csv(snakemake.input.label_vector, delimiter = '\t', header=0, index_col=0)

X_cnv = pd.read_csv(snakemake.input.feature_matrix_cnv, delimiter = '\t', header=0, index_col=0)
y_cnv = pd.read_csv(snakemake.input.label_vector_cnv, delimiter = '\t', header=0, index_col=0)

X_sv = pd.read_csv(snakemake.input.feature_matrix_sv, delimiter = '\t', header=0, index_col=0)
y_sv = pd.read_csv(snakemake.input.label_vector_sv, delimiter = '\t', header=0, index_col=0)

X_all = pd.read_csv(snakemake.input.feature_matrix_all, delimiter = '\t', header=0, index_col=0)
y_all = pd.read_csv(snakemake.input.label_vector_all, delimiter = '\t', header=0, index_col=0)

# read the file with best hyperparameters
best_hp = pd.read_csv(snakemake.input.best_hyper_param, delimiter = '\t', header=0)

####################################################################################
######### Testing RF Performance using Different Combination of Data Types #########
####################################################################################

skf = StratifiedKFold(n_splits=5, shuffle=True)

# extracting best hyperparameters
hps = best_hp.iloc[2,][0].split(',')

for i in range(len(hps)):
    if 'max_depth' in hps[i]:
        best_max_depth = int(hps[i].split(':')[1])
    elif 'max_features' in hps[i]:
        best_max_features = float(hps[i].split(':')[1])
    elif 'max_samples' in hps[i]:
        best_max_samples = float(hps[i].split(':')[1])
    elif 'min_samples_leaf' in hps[i]:
        best_min_samples_leaf = int(hps[i].split(':')[1])
    elif 'min_samples_split' in hps[i]:
        best_min_samples_split = int(hps[i].split(':')[1])

# convert label dataframe to vector of just labels
y = y.y
y_cnv = y_cnv.y
y_sv = y_sv.y
y_all = y_all.y

# function to run each analysis setting in 30 permutations and get performance metrics
def get_avg_metrics(feat_matrix, label_vec):
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracies = []

    clf = RandomForestClassifier(n_estimators=3000, max_depth=int(best_max_depth), max_features=float(best_max_features), 
                                max_samples=float(best_max_samples), min_samples_split=int(best_min_samples_split), 
                                min_samples_leaf=int(best_min_samples_leaf), n_jobs=40)

    # run 5-fold CV analysis in 30 permutations
    for i in range(30):
        print('Round '+str(i+1)+' of CV ...')
        all_pred_df = pd.DataFrame({'p_id':['a'], 'status':['mut_wt'], 'predict':['mut_wt']})
        
        for train_index, test_index in skf.split(feat_matrix, label_vec):

            y_train, y_test = label_vec.iloc[train_index], label_vec.iloc[test_index]
            X_train, X_test = feat_matrix.iloc[train_index], feat_matrix.iloc[test_index]

            # train the model on 80% of data
            clf.fit(X_train, y_train)

            # test on 20% of data
            sample_ids = X_test.index.values

            rf_predictions = clf.predict(X_test)
            rf_pred_df = pd.DataFrame({'p_id':sample_ids, 'status':y_test, 'predict':rf_predictions})
            all_pred_df = pd.concat([all_pred_df, rf_pred_df], axis=0)
        
        all_pred_df = all_pred_df[all_pred_df.p_id != 'a']
        
        # get performance metrics
        precision, recall, f1, support = precision_recall_fscore_support(all_pred_df.status, all_pred_df.predict, average='macro')
        
        # add performance metrics to list of metrics
        precision_scores.append(round(precision, ndigits=2))
        recall_scores.append(round(recall, ndigits=2))
        f1_scores.append(round(f1, ndigits=2))
        accuracies.append(round(accuracy_score(all_pred_df.status, all_pred_df.predict), ndigits=2))
        
    return precision_scores, recall_scores, f1_scores, accuracies

####################################################################
######### Testing RF Performance using SNV/INDEL Data Only #########
####################################################################
print('Training RF on SNV data only ...')
print('')

# random forest performance on SNV data only
precision_scores_snv_only, recall_scores_snv_only, f1_scores_snv_only, accuracies_snv_only = get_avg_metrics(X, y)

f_1 = open(snakemake.output.data_types_combinations_results, 'w')

print('SNVs/INDELs Only Avg. Precision:', round(np.mean(precision_scores_snv_only), ndigits=4), file=f_1)
print('SNVs/INDELs Only Std. Precision:', round(np.std(precision_scores_snv_only), ndigits=4), file=f_1)

print('SNVs/INDELs Only Avg. Recall:', round(np.mean(recall_scores_snv_only), ndigits=4), file=f_1)
print('SNVs/INDELs Only Std. Recall:', round(np.std(recall_scores_snv_only), ndigits=4), file=f_1)

print('SNVs/INDELs Only Avg. F1 Score:', round(np.mean(f1_scores_snv_only), ndigits=4), file=f_1)
print('SNVs/INDELs Only Std. F1 Score:', round(np.std(f1_scores_snv_only), ndigits=4), file=f_1)

print('SNVs/INDELs Only Avg. Accuracy:', round(np.mean(accuracies_snv_only), ndigits=4), file=f_1)
print('SNVs/INDELs Only Std. Accuracy:', round(np.std(accuracies_snv_only), ndigits=4), file=f_1)

#######################################################################
######### Testing RF Performance using SNV/INDEL and CNV Data #########
#######################################################################
print('Training RF on SNV and CNV data ...')
print('')

# random forest performance on SNV and CNV data
precision_scores_snv_and_cnv, recall_scores_snv_and_cnv, f1_scores_snv_and_cnv, accuracies_snv_and_cnv = get_avg_metrics(X_cnv, y_cnv)

print('SNVs and CNVs Avg. Precision:', round(np.mean(precision_scores_snv_and_cnv), ndigits=4), file=f_1)
print('SNVs and CNVs Std. Precision:', round(np.std(precision_scores_snv_and_cnv), ndigits=4), file=f_1)

print('SNVs and CNVs Avg. Recall:', round(np.mean(recall_scores_snv_and_cnv), ndigits=4), file=f_1)
print('SNVs and CNVs Std. Recall:', round(np.std(recall_scores_snv_and_cnv), ndigits=4), file=f_1)

print('SNVs and CNVs Avg. F1 Score:', round(np.mean(f1_scores_snv_and_cnv), ndigits=4), file=f_1)
print('SNVs and CNVs Std. F1 Score:', round(np.std(f1_scores_snv_and_cnv), ndigits=4), file=f_1)

print('SNVs and CNVs Avg. Accuracy:', round(np.mean(accuracies_snv_and_cnv), ndigits=4), file=f_1)
print('SNVs and CNVs Std. Accuracy:', round(np.std(accuracies_snv_and_cnv), ndigits=4), file=f_1)

######################################################################
######### Testing RF Performance using SNV/INDEL and SV Data #########
######################################################################
print('Training RF on SNV and SV data ...')
print('')

# random forest performance on SNV and SV data
precision_scores_snv_and_sv, recall_scores_snv_and_sv, f1_scores_snv_and_sv, accuracies_snv_and_sv = get_avg_metrics(X_sv, y_sv)

print('SNVs and SVs Avg. Precision:', round(np.mean(precision_scores_snv_and_sv), ndigits=4), file=f_1)
print('SNVs and SVs Std. Precision:', round(np.std(precision_scores_snv_and_sv), ndigits=4), file=f_1)

print('SNVs and SVs Avg. Recall:', round(np.mean(recall_scores_snv_and_sv), ndigits=4), file=f_1)
print('SNVs and SVs Std. Recall:', round(np.std(recall_scores_snv_and_sv), ndigits=4), file=f_1)

print('SNVs and SVs Avg. F1 Score:', round(np.mean(f1_scores_snv_and_sv), ndigits=4), file=f_1)
print('SNVs and SVs Std. F1 Score:', round(np.std(f1_scores_snv_and_sv), ndigits=4), file=f_1)

print('SNVs and SVs Avg. Accuracy:', round(np.mean(accuracies_snv_and_sv), ndigits=4), file=f_1)
print('SNVs and SVs Std. Accuracy:', round(np.std(accuracies_snv_and_sv), ndigits=4), file=f_1)

############################################################################
######### Testing RF Performance using SNV/INDEL, CNV, and SV Data #########
############################################################################
print('Training RF on SNV, CNV and SV data ...')
print('')

# random forest performance on SNV, CNV and SV data
precision_scores_snv_cnv_sv, recall_scores_snv_cnv_sv, f1_scores_snv_cnv_sv, accuracies_snv_cnv_sv = get_avg_metrics(X_cnv_sv, y_cnv_sv)

print('All Data Types Avg. Precision:', round(np.mean(precision_scores_snv_cnv_sv), ndigits=4), file=f_1)
print('All Data Types Std. Precision:', round(np.std(precision_scores_snv_cnv_sv), ndigits=4), file=f_1)

print('All Data Types Avg. Recall:', round(np.mean(recall_scores_snv_cnv_sv), ndigits=4), file=f_1)
print('All Data Types Std. Recall:', round(np.std(recall_scores_snv_cnv_sv), ndigits=4), file=f_1)

print('All Data Types Avg. F1 Score:', round(np.mean(f1_scores_snv_cnv_sv), ndigits=4), file=f_1)
print('All Data Types Std. F1 Score:', round(np.std(f1_scores_snv_cnv_sv), ndigits=4), file=f_1)

print('All Data Types Avg. Accuracy:', round(np.mean(accuracies_snv_cnv_sv), ndigits=4), file=f_1)
print('All Data Types Std. Accuracy:', round(np.std(accuracies_snv_cnv_sv), ndigits=4), file=f_1)
f_1.close()

print('All test settings are done running and results are saved.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
