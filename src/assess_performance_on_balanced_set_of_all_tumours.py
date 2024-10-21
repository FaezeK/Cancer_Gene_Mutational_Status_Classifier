################################################################################################################
## This is a script to test the performance of random forest on the balanced set of all tumour types in 
## permutations to find the best analysis settings for each gene of interest.
################################################################################################################

# import required libraries
import pandas as pd
import timeit
import sys
from sklearn.ensemble import RandomForestClassifier

# variables provided at run-time
gene_of_interest = sys.argv[1]

# timing the run-time
start_time = timeit.default_timer()

######################################
########## Read input files ##########
######################################
print('Reading input files ...')
print('')

# read feature matrix and label vector
X = pd.read_csv(snakemake.input.feature_matrix, delimiter = '\t', header=0)
y = pd.read_csv(snakemake.input.label_vector, delimiter = '\t', header=0)

X_cnv = pd.read_csv(snakemake.input.feature_matrix_cnv, delimiter = '\t', header=0)
y_cnv = pd.read_csv(snakemake.input.label_vector_cnv, delimiter = '\t', header=0)

# read the file with best hyperparameters
best_hp = pd.read_csv(snakemake.input.best_hyper_param, delimiter = '\t', header=0)

# read the file containing the best setting
best_setting = pd.read_csv(snakemake.input.best_setting, delimiter = '\t', header=0)

# read samples tumour types
tcga_t_type = pd.read_csv(snakemake.input.tcga_t_type, delimiter = '\t', header=0)
pog_t_type = pd.read_csv(snakemake.input.pog_t_type, delimiter = '\t', header=0)

################################################################################
######### Balancing the Mutational Categories before Running 5-fold CV #########
################################################################################
print('Balancing the Mutational Categories before Running 5-fold CV ...')
print('')

# find class ratios and balance per tumour type
tcga_pog_t_type = pd.concat([tcga_t_type, pog_t_type], axis=0)

if best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_only':
    y_df = pd.DataFrame({'p_id':X.index, 'label':y})
elif best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_CNV':
    y_df = pd.DataFrame({'p_id':X_cnv.index, 'label':y})

samples_w_t_types = tcga_pog_t_type.merge(y_df, on='p_id')

# find mut and wt cnts per tumour type
val_cnts = samples_w_t_types.value_counts(['tumour_type_abbv', 'label'])

# find tumour types with at least some mutant samples
t_types_w_mut = samples_w_t_types[samples_w_t_types.label=='mut'].tumour_type_abbv.unique()

# balance the dataset
all_samples_to_keep = pd.Series(dtype='str')

for t in t_types_w_mut:
    mut_cnt = val_cnts[(val_cnts.index.get_level_values(0)==t) & (val_cnts.index.get_level_values(1)=='mut')][0]
    wt_cnt = val_cnts[(val_cnts.index.get_level_values(0)==t) & (val_cnts.index.get_level_values(1)=='wt')][0]
    
    all_wt_samples = samples_w_t_types[(samples_w_t_types.tumour_type_abbv==t) & (samples_w_t_types.label=='wt')].p_id
    all_mut_samples = samples_w_t_types[(samples_w_t_types.tumour_type_abbv==t) & (samples_w_t_types.label=='mut')].p_id
    
    # down-samples wild-type category
    if mut_cnt < wt_cnt:
        wt_samples_to_keep = all_wt_samples.sample(n=mut_cnt)
        samples_to_keep = pd.concat([all_mut_samples, wt_samples_to_keep])
        all_samples_to_keep = pd.concat([all_samples_to_keep, samples_to_keep])
    
    # down-sample mutant category
    else:
        mut_samples_to_keep = all_mut_samples.sample(n=wt_cnt)
        samples_to_keep = pd.concat([all_wt_samples, mut_samples_to_keep])
        all_samples_to_keep = pd.concat([all_samples_to_keep, samples_to_keep])

# make feature matrix and label vector with the selected samples
if best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_only':
    X_new = X[X.index.isin(all_samples_to_keep)]
elif best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_CNV':
    X_new = X_cnv[X_cnv.index.isin(all_samples_to_keep)]

X_new = X_new.sort_index()
y_new = y_df[y_df.p_id.isin(all_samples_to_keep)]
y_new = y_new.sort_values(by=['p_id'])
y_new = y_new.label

#######################################################################################
######### Performing 5-fold CV to get predictions on all TCGA and POG samples #########
#######################################################################################
print('Performing 5-fold CV ...')
print('')

skf = StratifiedKFold(n_splits=5, shuffle=True)

clf = RandomForestClassifier(n_estimators=3000, max_depth=int(best_max_depth), max_features=float(best_max_features), 
                             max_samples=float(best_max_samples), min_samples_split=int(best_min_samples_split), 
                             min_samples_leaf=int(best_min_samples_leaf), n_jobs=40)

precision_scores = []
recall_scores = []
f1_scores = []
accuracies = []

for i in range(30):
    print('Round '+str(i+1)+' of CV ...')

    # random forest performance on tcga and pog
    all_pred_df = pd.DataFrame({'p_id':['a'], 'status':['mut_wt'], 'predict':['mut_wt']})

    for train_index, test_index in skf.split(X_new, y_new):

        y_train, y_test = y_new.iloc[train_index], y_new.iloc[test_index]
        X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]

        # train the model on 80% of samples from balanced set of all tumours
        clf.fit(X_train, y_train)

        sample_ids = X_test.index.values

        # test the model on the remaining 20%
        rf_predictions = clf.predict(X_test)
        rf_pred_df = pd.DataFrame({'p_id':sample_ids, 'status':y_test, 'predict':rf_predictions})
        all_pred_df = pd.concat([all_pred_df, rf_pred_df], axis=0)
    
    all_pred_df = all_pred_df[all_pred_df.p_id != 'a']
    
    # obtain performance metrics
    precision, recall, f1, support = precision_recall_fscore_support(all_pred_df.status, all_pred_df.predict, average='macro')
    
    # add metrics to the list of all permutations
    precision_scores.append(round(precision, ndigits=2))
    recall_scores.append(round(recall, ndigits=2))
    f1_scores.append(round(f1, ndigits=2))
    accuracies.append(round(accuracy_score(all_pred_df.status, all_pred_df.predict), ndigits=2))

# assess performance
f_1 = open(snakemake.output.str(gene_of_interest)+'_permut_balanced_results_all_tumours.txt', 'w')

print('Avg. Precision:', round(np.mean(precision_scores), ndigits=4), file=f_1)
print('Std. Precision:', round(np.std(precision_scores), ndigits=4), file=f_1)
        
print('Avg. Recall:', round(np.mean(recall_scores), ndigits=4), file=f_1)
print('Std. Recall:', round(np.std(recall_scores), ndigits=4), file=f_1)

print('Avg. F1 Score:', round(np.mean(f1_scores), ndigits=4), file=f_1)
print('Std. F1 Score:', round(np.std(f1_scores), ndigits=4), file=f_1)

print('Avg. Accuracy:', round(np.mean(accuracies), ndigits=4), file=f_1)
print('Std. Accuracy:', round(np.std(accuracies), ndigits=4), file=f_1)

f_1.close()

print('Classification is performed on the balanced set of all tumour types.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('') 
