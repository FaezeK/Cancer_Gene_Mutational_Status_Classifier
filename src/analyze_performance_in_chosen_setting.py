##############################################################################################################
## This is a script to test the performance of random forest model on classification of samples using the
## selected settings from the previous step.
##############################################################################################################

# import required libraries
import pandas as pd
import numpy as np
import timeit
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

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

# read expression data of samples with non-impactful mutations
tcga_tpm_not_impactful_mut = pd.read_csv(snakemake.input.tcga_tpm_not_impactful_mut, delimiter = '\t', header=0, index_col=0)
pog_tpm_not_impactful_mut = pd.read_csv(snakemake.input.pog_tpm_not_impactful_mut, delimiter = '\t', header=0, index_col=0)

# read processed mutation data
tcga_all_mut = pd.read_csv(snakemake.input.tcga_mut_prcssd, delimiter = '\t', header=0)
pog_all_mut = pd.read_csv(snakemake.input.pog_mut_prcssd, delimiter = '\t', header=0)

######################################################################################################
######### Training RF with TCGA and POG samples with impactful mutations or wilt-type copies #########
######################################################################################################
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

clf = RandomForestClassifier(n_estimators=3000, max_depth=best_max_depth, max_features=best_max_features, 
                             max_samples=best_max_samples, min_samples_split=best_min_samples_split, 
                             min_samples_leaf=best_min_samples_leaf, n_jobs=40)

# convert label dataframe to vector of just labels
y = y.y
y_cnv = y_cnv.y

# train the model using the appropriate feature matrix and label vector
if best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_only':
    clf.fit(X, y)
elif best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_CNV':
    clf.fit(X_cnv, y_cnv)

# important features in classification
rand_f_scores = clf.feature_importances_
indices = np.argsort(rand_f_scores)
rand_f_scores_sorted = pd.Series(np.sort(rand_f_scores))

if best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_only':
    rand_forest_importance_scores_df = pd.DataFrame({'gene':pd.Series(X.columns[indices]), 'importance_score':rand_f_scores_sorted})
elif best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_CNV':
    rand_forest_importance_scores_df = pd.DataFrame({'gene':pd.Series(X_cnv.columns[indices]), 'importance_score':rand_f_scores_sorted})

rand_forest_importance_scores_df = rand_forest_importance_scores_df.sort_values(by='importance_score', ascending=False)
rand_forest_importance_scores_df.to_csv(snakemake.output.gene_importance_scores_from_RF, sep='\t', index=False)

###################################################################
######### Testing on samples with not-impactful mutations #########
###################################################################
print('Testing RF on samples with not-impactful mutations ...')
print('')

both_tpm_not_impactful_mut = pd.concat([tcga_tpm_not_impactful_mut, pog_tpm_not_impactful_mut], axis=0)

not_impact_preds = clf.predict(both_tpm_not_impactful_mut)
not_impact_preds_df = pd.DataFrame({'p_id':both_tpm_not_impactful_mut.index, 'pred':not_impact_preds})

# add the mutation consequences of the above samples to the dataframe
tcga_not_impact_mut = tcga_all_mut[(tcga_all_mut.p_id.isin(both_tpm_not_impactful_mut.index)) & (tcga_all_mut.gene_name == gene_of_interest)]
pog_not_impact_mut = pog_all_mut[(pog_all_mut.p_id.isin(both_tpm_not_impactful_mut.index)) & (pog_all_mut.gene_name == gene_of_interest)]

not_impact_mut = pd.concat([tcga_not_impact_mut, pog_not_impact_mut], axis=0)

# when mutation types are combined, break them into separate lines
not_impact_mut['consequence_ls'] = not_impact_mut.consequence.str.split('\+')
not_impact_mut = not_impact_mut.explode('consequence_ls')
not_impact_mut = not_impact_mut.drop(columns=['consequence'])

not_impact_mut['consequence'] = not_impact_mut.consequence_ls.str.split('\&')
not_impact_mut = not_impact_mut.explode('consequence')
not_impact_mut = not_impact_mut.drop(columns=['consequence_ls'])

# merge prediction and consequence tables
not_impact_preds_w_conseq_df = not_impact_preds_df.merge(not_impact_mut, how='inner', on='p_id')

# write the above table into results directory
not_impact_preds_w_conseq_df.to_csv(snakemake.output.pred_on_sample_w_not_impact_mut, sep='\t', index=False)

######################################################################################################
######### Finding Not-impactful Mutation Categories where Most Samples are Labeled as Mutant #########
######################################################################################################

# extract the number of samples predicted as mutant or wt per consequence
val_cnts = not_impact_preds_w_conseq_df[['consequence','pred']].value_counts()
conseq = val_cnts.index.get_level_values(0)
pred = val_cnts.index.get_level_values(1)

# make table of consequence, prediction, and their counts
val_cnts_df = pd.DataFrame({'conseq':conseq, 'pred':pred, 'val':val_cnts})
val_cnts_df = val_cnts_df.reset_index(drop=True)

# add zero for mut or wt predictions if for a consequence all samples are predicted as mut or wt
for c in val_cnts_df.conseq.unique():
    temp = val_cnts_df[val_cnts_df.conseq == c]
    if temp.shape[0] == 1:
        if temp.pred.iloc[0] == 'wt':
            new_row = pd.DataFrame([[c, 'mut', 0]], columns=['conseq','pred','val'])
            val_cnts_df = pd.concat([val_cnts_df, new_row], axis=0)
        elif temp.pred.iloc[0] == 'mut':
            new_row = pd.DataFrame([[c, 'wt', 0]], columns=['conseq','pred','val'])
            val_cnts_df = pd.concat([val_cnts_df, new_row], axis=0)
            
# sort the above df and reset index
val_cnts_df = val_cnts_df.sort_values(by=['conseq','pred'])
val_cnts_df = val_cnts_df.reset_index(drop=True)

# add p-value from binomial test (per consequence)
p_val_col = []

for i in np.arange(0,val_cnts_df.shape[0],2):
    p_val = stats.binom_test(val_cnts_df.val[i], n=val_cnts_df.val[i]+val_cnts_df.val[i+1])
    if np.round(p_val, 2) == 0:
        p_val = np.round(p_val, 6)
    else:
        p_val = np.round(p_val, 2)
    p_val_col.append(str(p_val))
    p_val_col.append(str(p_val))
    
val_cnts_df['p_val'] = p_val_col

val_cnts_df.to_csv(snakemake.output.not_impact_mut_groups_pred_n_binom_p_val, sep='\t', index=False)

# based on above table find the mutation types that led to mutant prediction
val_cnt_not_impact_all_df = pd.DataFrame()

for conseq in val_cnts_df.conseq.unique():
    tmp = val_cnts_df[val_cnts_df.conseq == conseq]
    # only investigate consequences were p-value is significant and there are more mutant predictions
    if float(tmp.p_val.iloc[0]) < 0.05 and tmp[tmp.pred=='mut'].val.iloc[0] > tmp[tmp.pred=='wt'].val.iloc[0]:
        not_impact_predicted_as_mut = not_impact_preds_w_conseq_df[(not_impact_preds_w_conseq_df.consequence==conseq) & 
                                                           (not_impact_preds_w_conseq_df.pred=='mut')]
        # remove the rows with NA amino acid or base change
        not_impact_predicted_as_mut.loc[not_impact_predicted_as_mut.amino_acid_change.isna(),'amino_acid_change'] = 'NA'
        not_impact_predicted_as_mut.loc[not_impact_predicted_as_mut.base_change.isna(),'base_change'] = 'NA'
        if not_impact_predicted_as_mut.shape[0] != 0:
            # count the samples with different types of amino acid or base change
            val_cnt_not_impact = not_impact_predicted_as_mut[['amino_acid_change','base_change']].value_counts()
            val_cnt_not_impact_df = pd.DataFrame({'amino_acid_change':val_cnt_not_impact.index.get_level_values(0), 
                                        'base_change':val_cnt_not_impact.index.get_level_values(1), 'cnt':val_cnt_not_impact,
                                        'conseq':conseq})
            val_cnt_not_impact_df = val_cnt_not_impact_df.reset_index(drop=True)
            val_cnt_not_impact_all_df = pd.concat([val_cnt_not_impact_all_df, val_cnt_not_impact_df])
      
if val_cnt_not_impact_all_df.shape[0] != 0:
    val_cnt_not_impact_all_df.to_csv(snakemake.output.not_impact_conseq_base_n_aa_changes, sep='\t', index=False)
else: # this can happen when either none of the p-values are significant or most samples are predicted as wild-type as expected
        # or although a significant number of samples were predicted as mutant, amino acid change and base chage are missing
        # in these cases, only a message appears in the file indicating that no categories were found
    val_cnt_not_impact_df = pd.DataFrame({'message':'No mutations categories with the desired condition is found!'})
    val_cnt_not_impact_all_df = pd.concat([val_cnt_not_impact_all_df, val_cnt_not_impact_df])
    val_cnt_not_impact_all_df.to_csv(snakemake.output.not_impact_conseq_base_n_aa_changes, sep='\t', index=False)

print('Classification is performed using all samples and results are analyzed.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
