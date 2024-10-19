##############################################################################################################
## This is a script to find the significance threshold for the genes contributing the most to the 
## classification task.
##############################################################################################################

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

# read feature importance scores found in previous step
gene_importance_scores_from_RF = pd.read_csv(snakemake.input.gene_importance_scores_from_RF, delimiter = '\t', header=0)

# read feature matrix and label vector
X = pd.read_csv(snakemake.input.feature_matrix, delimiter = '\t', header=0)
y = pd.read_csv(snakemake.input.label_vector, delimiter = '\t', header=0)

X_cnv = pd.read_csv(snakemake.input.feature_matrix_cnv, delimiter = '\t', header=0)
y_cnv = pd.read_csv(snakemake.input.label_vector_cnv, delimiter = '\t', header=0)

# read the file with best hyperparameters
best_hp = pd.read_csv(snakemake.input.best_hyper_param, delimiter = '\t', header=0)

# read the file containing the best setting
best_setting = pd.read_csv(snakemake.input.best_setting, delimiter = '\t', header=0)

###################################################################################
######### Finding the threshold for the top genes important in prediction #########
###################################################################################
print('Finding the threshold for the number of important genes ...')
print('')

rand_forest_importance_scores_true_df = gene_importance_scores_from_RF.head(n=1000)
rand_forest_importance_scores_true_df['iter'] = 'true_lab'
rand_forest_importance_scores_true_df['rank'] = range(1,(len(rand_forest_importance_scores_true_df.gene)+1))
rand_forest_importance_scores_df_all = rand_forest_importance_scores_true_df

if best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_only':
    X_1000 = X.iloc[:,X.columns.isin(rand_forest_importance_scores_true_df.gene)]
elif best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_CNV':
    X_1000 = X_cnv.iloc[:,X_cnv.columns.isin(rand_forest_importance_scores_true_df.gene)]

# 100 permutations of shuffling labels to find random importance scores
for iter in range(100):
    
    if best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_only':
        y_imp_feat = y.sample(frac=1)
    elif best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_CNV':
        y_imp_feat = y_cnv.sample(frac=1)
    
    # train the model with top 1000 features and randomly shuffled labels
    clf.fit(X_1000, y_imp_feat)

    # extract the top features in classification
    rand_f_scores = clf.feature_importances_
    indices = np.argsort(rand_f_scores)
    rand_f_scores_sorted = pd.Series(np.sort(rand_f_scores))
    rand_forest_importance_scores_df = pd.DataFrame({'gene':pd.Series(X_1000.columns[indices]), 'importance_score':rand_f_scores_sorted, 'iter':iter})
    rand_forest_importance_scores_df = rand_forest_importance_scores_df[rand_forest_importance_scores_df.importance_score != 0]
    rand_forest_importance_scores_df = rand_forest_importance_scores_df.sort_values(by='importance_score', ascending=False)
    rand_forest_importance_scores_df['rank'] = range(1,(len(rand_forest_importance_scores_df.gene)+1))
    rand_forest_importance_scores_df_all = pd.concat([rand_forest_importance_scores_df_all, rand_forest_importance_scores_df], axis=0)
    
# find the mean and standard deviation of importance scores found by using shuffled labels
rand_frst_imp_scr_df = rand_forest_importance_scores_df_all.pivot(index='rank', columns='iter', values='importance_score')
rand_frst_imp_scr_df = rand_frst_imp_scr_df.dropna()

rand_frst_imp_scr_df.columns = ['iter' + str(s) for s in list(rand_frst_imp_scr_df.columns)]
rand_frst_imp_scr_df = rand_frst_imp_scr_df.rename(columns={"itertrue_lab": "true_lab"})
rand_frst_imp_scr_df = rand_frst_imp_scr_df.sort_values(by='true_lab', ascending=False)
n, d = rand_frst_imp_scr_df.shape

rand_frst_imp_scr_df_copy = rand_frst_imp_scr_df
rand_frst_imp_scr_df_copy['mean'] = rand_frst_imp_scr_df_copy.iloc[:,0:(d-1)].mean(axis=1)
rand_frst_imp_scr_df_copy['sd'] = rand_frst_imp_scr_df_copy.iloc[:,0:(d-1)].std(axis=1)

# make the graph of real importance scores vs mean of shuffled scores
plt.figure(figsize=(10,7))
plt.plot(rand_frst_imp_scr_df_copy.index, rand_frst_imp_scr_df_copy.true_lab, label='True labels', color='darkgreen')
plt.plot(rand_frst_imp_scr_df_copy.index, rand_frst_imp_scr_df_copy['mean'], label='Mean shuffled labels', color='indigo')
plt.legend()
plt.fill_between(rand_frst_imp_scr_df_copy.index, rand_frst_imp_scr_df_copy['mean']+rand_frst_imp_scr_df_copy.sd, rand_frst_imp_scr_df_copy['mean']-rand_frst_imp_scr_df_copy.sd, color='mediumslateblue')
plt.xlabel('Gene Rank', fontsize=16)
plt.ylabel('Importance (Gini) Score', fontsize=16)
plt.savefig(snakemake.output.str(gene_of_interest)+'true_vs_shuffled_importance_scores.jpg',format='jpeg', bbox_inches='tight', dpi=300)
plt.close()

# find the threshold value
true_score_compared_to_mean = rand_frst_imp_scr_df_copy.true_lab > rand_frst_imp_scr_df_copy['mean']
first_position_mean_gt_true_score = (true_score_compared_to_mean==False).idxmax()
if first_position_mean_gt_true_score == 1:
    first_position_mean_gt_true_score = (true_score_compared_to_mean==True).idxmax()

# record the number of top genes
first_position_mean_gt_true_score.to_csv(snakemake.output.num_important_genes, sep='\t', index=False)

# make a zoomed-in version of the previous graph
rand_frst_imp_feat_scr_df2_copy2 = rand_frst_imp_scr_df_copy.iloc[first_position_mean_gt_true_score-35:first_position_mean_gt_true_score+35,:]
plt.figure(figsize=(10,7))
plt.plot(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2.true_lab, label='True labels', color='darkgreen')
plt.plot(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2['mean'], label='Mean shuffled labels', color='indigo')
plt.legend()
plt.fill_between(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2['mean']+rand_frst_imp_feat_scr_df2_copy2.sd, rand_frst_imp_feat_scr_df2_copy2['mean']-rand_frst_imp_feat_scr_df2_copy2.sd, color='mediumslateblue')
plt.savefig(snakemake.output.str(gene_of_interest)+'true_vs_shuffled_importance_scores_zoomed_in.jpg',format='jpeg', bbox_inches='tight', dpi=300)
plt.close()

# make an even more zoomed-in version of the previous graph
rand_frst_imp_feat_scr_df2_copy2 = rand_frst_imp_scr_df_copy.iloc[first_position_mean_gt_true_score-8:first_position_mean_gt_true_score+8,:]
plt.figure(figsize=(10,7))
plt.plot(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2.true_lab, label='True labels', color='darkgreen')
plt.plot(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2['mean'], label='Mean shuffled labels', color='indigo')
plt.legend()
plt.fill_between(rand_frst_imp_feat_scr_df2_copy2.index, rand_frst_imp_feat_scr_df2_copy2['mean']+rand_frst_imp_feat_scr_df2_copy2.sd, rand_frst_imp_feat_scr_df2_copy2['mean']-rand_frst_imp_feat_scr_df2_copy2.sd, color='mediumslateblue')
plt.savefig(snakemake.output.str(gene_of_interest)+'true_vs_shuffled_importance_scores_zoomed_in2.jpg',format='jpeg', bbox_inches='tight', dpi=300)
plt.close()

print('The threshold for top genes is found and graphs are made.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
