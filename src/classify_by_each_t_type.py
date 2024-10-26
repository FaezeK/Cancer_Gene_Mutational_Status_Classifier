##############################################################################################################
## This is a script to test the performance of random forest on all tumour types separately to find the
## tumour types wehre classification led to the best results for each gene of interest.
##############################################################################################################

# import required libraries
import pandas as pd
import numpy as np
import timeit
import sys
import test_performance_helper as tph
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

# variables provided at run-time
gene_of_interest = snakemake.params.gene_name

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

# read the file with best hyperparameters
best_hp = pd.read_csv(snakemake.input.best_hyper_param, delimiter = '\t', header=0)

# read the file containing the best setting
best_setting = pd.read_csv(snakemake.input.best_setting, delimiter = '\t', header=0)

# read samples tumour types
tcga_t_type = pd.read_csv(snakemake.input.tcga_t_type, delimiter = '\t', header=0)
pog_t_type = pd.read_csv(snakemake.input.pog_t_type, delimiter = '\t', header=0)

# read expression files
tcga_tpm_impactful_mut = pd.read_csv(snakemake.input.tcga_tpm_impactful_mut, delimiter = '\t', header=0, index_col=0)
tcga_tpm_wt = pd.read_csv(snakemake.input.tcga_tpm_wt, delimiter = '\t', header=0, index_col=0)

pog_tpm_impactful_mut = pd.read_csv(snakemake.input.pog_tpm_impactful_mut, delimiter = '\t', header=0, index_col=0)
pog_tpm_wt = pd.read_csv(snakemake.input.pog_tpm_wt, delimiter = '\t', header=0, index_col=0)

#######################################################################################
######### Training RF with all samples containing impactful mutations or ##############
######### wilt-type copies from all tumour types separately ###########################
#######################################################################################
print('Training RF ...')
print('')

# extracting best hyperparameters
hps = best_hp.iloc[2,][0].split(',')

for i in range(len(hps)):
    if 'max_depth' in hps[i]:
        best_max_depth = int(hps[i].split(':')[1])
    elif 'max_features' in hps[i]:
        if 'sqrt' in hps[i].split(':')[1]:
            best_max_features = 'sqrt'
        else:
            best_max_features = float(hps[i].split(':')[1])
    elif 'max_samples' in hps[i]:
        best_max_samples = float(hps[i].split(':')[1])
    elif 'min_samples_leaf' in hps[i]:
        best_min_samples_leaf = int(hps[i].split(':')[1])
    elif 'min_samples_split' in hps[i]:
        best_min_samples_split = int(hps[i].split(':')[1])
    
# build random forest template
clf = RandomForestClassifier(n_estimators=3000, max_depth=best_max_depth, max_features=best_max_features, 
                             max_samples=best_max_samples, min_samples_split=best_min_samples_split, 
                             min_samples_leaf=best_min_samples_leaf, n_jobs=40)

# retrieve the output files passed from Snakemake so tumour types can be added to the end of file name
output_files_placeholders = sys.argv[14:18]

# extract TCGA sample ids that exist in tumour types with significant f1-score 
all_t_type = pd.concat([tcga_t_type, pog_t_type], axis=0)

# find all tumour types abbreviations
t_types = all_t_type.tumour_type_abbv.unique()

# run the analysis on each tumour type in a for loop
for t in t_types:
    spcfc_p_ids = all_t_type[all_t_type.tumour_type_abbv==t].p_id
    
    # find the number of samples with mutations 
    n_mut_tcga = tcga_tpm_impactful_mut[tcga_tpm_impactful_mut.index.isin(spcfc_p_ids)].shape[0]
    n_mut_pog = pog_tpm_impactful_mut[pog_tpm_impactful_mut.index.isin(spcfc_p_ids)].shape[0]
    n_mut = n_mut_tcga + n_mut_pog
    
    # run the analysis if there is at least 10 samples with mutations in the gene of interest
    if n_mut >= 10:
        # filter X and y based on extracted ids
        if best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_only':
            X_new = X.loc[(X.index.isin(spcfc_p_ids)==True),]
            y_new = y.loc[(y.index.isin(spcfc_p_ids)==True),]
        elif best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_CNV':
            X_new = X_cnv.loc[(X_cnv.index.isin(spcfc_p_ids)==True),]
            y_new = y_cnv.loc[(y_cnv.index.isin(spcfc_p_ids)==True),]
        
        # convert label dataframe to vector to train RF    
        y_new_rf = y_new.y
        
        # run 5-fold CV on the samples from each tumour type
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        
        # random forest performance on tcga and pog
        all_pred_df, all_prob, true_label_prob = tph.test_performance_5_fold_CV(clf, skf, X_new, y_new_rf)
        
        # assess performance
        #f_1 = open(snakemake.output.t_type_results, 'w')
        f_1 = open(output_files_placeholders[0].replace(".txt", "_"+str(t)+".txt"), 'w')

        print(confusion_matrix(all_pred_df.status, all_pred_df.predict, labels=['mut','wt']), file=f_1)
        print(classification_report(all_pred_df.status, all_pred_df.predict), file=f_1)

        f_1.close()
        
        # extract the genes contributing the most to classification in each tumour type
        # train on all samples
        clf.fit(X_new, y_new_rf)

        # important features in classification
        rand_f_scores = clf.feature_importances_
        indices = np.argsort(rand_f_scores)
        rand_f_scores_sorted = pd.Series(np.sort(rand_f_scores))
        rand_forest_importance_scores_true_df = pd.DataFrame({'gene':pd.Series(X_new.columns[indices]), 'importance_score':rand_f_scores_sorted})
        rand_forest_importance_scores_true_df = rand_forest_importance_scores_true_df.sort_values(by='importance_score', ascending=False)
        rand_forest_importance_scores_true_df.to_csv(output_files_placeholders[1].replace(".txt", "_"+str(t)+".txt"), sep='\t', index=False)
        
        ##########################################################
        ### Test the performance on balanced sets of tumour types
        # find the number of samples containing wild-type copies
        n_wt_tcga = tcga_tpm_wt[tcga_tpm_wt.index.isin(spcfc_p_ids)].shape[0]
        n_wt_pog = pog_tpm_wt[pog_tpm_wt.index.isin(spcfc_p_ids)].shape[0]
        n_wt = n_wt_tcga + n_wt_pog
        
        # run the analysis if there is at least 10 samples in each category (mutant and wild-type)
        if n_wt >= 10:
            all_wt_samples = pd.Series(y_new.loc[(y_new.index.isin(spcfc_p_ids)) & (y_new.y=='wt'),].index)
            all_mut_samples = pd.Series(y_new.loc[(y_new.index.isin(spcfc_p_ids)) & (y_new.y=='mut'),].index)
            
            # down-sample the category with higher number of samples
            if n_wt > n_mut:
                wt_samples_to_keep = all_wt_samples.sample(n=n_mut)
                samples_to_keep = pd.concat([all_mut_samples, wt_samples_to_keep])
            else:
                mut_samples_to_keep = all_mut_samples.sample(n=n_wt)
                samples_to_keep = pd.concat([all_wt_samples, mut_samples_to_keep])
            
            # filter data based on selected samples
            if best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_only':
                X_new2 = X[X.index.isin(samples_to_keep)]
            elif best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_CNV':
                X_new2 = X_cnv[X_cnv.index.isin(samples_to_keep)]
            
            X_new2 = X_new2.sort_index()
            
            y_new2 = y_new[y_new.index.isin(samples_to_keep)]
            y_new2 = y_new2.sort_values(by=['p_id'])
            y_new2 = y_new2.y
            
            # random forest performance on tcga and pog
            all_pred_df2, all_prob2, true_label_prob2 = tph.test_performance_5_fold_CV(clf, skf, X_new2, y_new2)
        
            # assess performance
            #f_2 = open(snakemake.output.t_type_results_balanced, 'w')
            f_2 = open(output_files_placeholders[2].replace(".txt", "_"+str(t)+".txt"), 'w')
            
            print(confusion_matrix(all_pred_df2.status, all_pred_df2.predict, labels=['mut','wt']), file=f_2)
            print(classification_report(all_pred_df2.status, all_pred_df2.predict), file=f_2)
            
            f_2.close()
            
            # extract the genes contributing the most to classification in balanced set of each tumour type
            # train on all samples
            clf.fit(X_new2, y_new2)

            # important features in classification
            rand_f_scores2 = clf.feature_importances_
            indices2 = np.argsort(rand_f_scores2)
            rand_f_scores_sorted2 = pd.Series(np.sort(rand_f_scores2))
            rand_forest_importance_scores_true_df2 = pd.DataFrame({'gene':pd.Series(X_new2.columns[indices2]), 'importance_score':rand_f_scores_sorted2})
            rand_forest_importance_scores_true_df2 = rand_forest_importance_scores_true_df2.sort_values(by='importance_score', ascending=False)
            rand_forest_importance_scores_true_df2.to_csv(output_files_placeholders[3].replace(".txt", "_"+str(t)+".txt"), sep='\t', index=False)

print('Classification is performed on both balanced and imbalanced sets of each tumour type.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
