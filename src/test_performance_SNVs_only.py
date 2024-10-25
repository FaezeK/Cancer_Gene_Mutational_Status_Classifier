##############################################################################################################
## This is a script to test the performance of random forest model on classification of samples using
## SNVs/INDELs data only.
##############################################################################################################

# import required libraries
import pandas as pd
import numpy as np
import timeit
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt
import test_performance_helper as tph
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

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

# read the file with best hyperparameters
best_hp = pd.read_csv(snakemake.input.best_hyper_param, delimiter = '\t', header=0)

# read TCGA tumour types
tcga_t_type = pd.read_csv(snakemake.input.tcga_t_type, delimiter = '\t', header=0)

#######################################################################################
######### Performing 5-fold CV to get predictions on all TCGA and POG samples #########
#######################################################################################
print('Extracting best hyperparameters ...')
print('')
hps = best_hp.iloc[2,][0].split(',')

for i in range(len(hps)):
    if 'max_depth' in hps[i]:
        best_max_depth = int(hps[i].split(':')[1])
    elif 'max_features' in hps[i]:
        if 'sqrt' in hps[i].split(':')[1]:
            best_max_features = hps[i].split(':')[1]
        else:
            best_max_features = float(hps[i].split(':')[1])
    elif 'max_samples' in hps[i]:
        best_max_samples = float(hps[i].split(':')[1])
    elif 'min_samples_leaf' in hps[i]:
        best_min_samples_leaf = int(hps[i].split(':')[1])
    elif 'min_samples_split' in hps[i]:
        best_min_samples_split = int(hps[i].split(':')[1])

# convert label dataframe to vector of just labels
y = y.y

print('Performing 5-fold CV ...')
print('')
skf = StratifiedKFold(n_splits=5, shuffle=True)

clf = RandomForestClassifier(n_estimators=3000, max_depth=best_max_depth, max_features=best_max_features, 
                             max_samples=best_max_samples, min_samples_split=best_min_samples_split, 
                             min_samples_leaf=best_min_samples_leaf, n_jobs=40)

# random forest performance on tcga and pog
all_pred_df, all_prob, true_label_prob = tph.test_performance_5_fold_CV(clf, skf, X, y)

# assess performance
f_1 = open(snakemake.output.classification_results_SNVs_only, 'w')

print(confusion_matrix(all_pred_df.status, all_pred_df.predict, labels=['mut','wt']), file=f_1)
print(classification_report(all_pred_df.status, all_pred_df.predict), file=f_1)

both_auprc = sklearn.metrics.average_precision_score(all_pred_df.status, true_label_prob, pos_label="wt")
both_auroc = sklearn.metrics.roc_auc_score(all_pred_df.status, true_label_prob)
print('AUPRC:', file=f_1)
print(both_auprc, file=f_1)
print('AUROC:', file=f_1)
print(both_auroc, file=f_1)

f_1.close()

# make auroc and auprc graphs
fig, axes =plt.subplots(1, 2, figsize=(8, 4), dpi=300)
sns.set_theme()
fnt_size = 18
tph.generate_auroc_curve(all_pred_df.status, true_label_prob, axes[0])
tph.generate_auprc_curve(all_pred_df.status, true_label_prob, axes[1])

axes[0].text(0.20, 0.60, 'AUROC='+str(round(both_auroc, 2)))
axes[1].text(0.50, 0.60, 'AUPRC='+str(round(both_auprc, 2)))

fig.savefig(snakemake.output.auroc_auprc_SNVs_only,format='jpeg',dpi=300,bbox_inches='tight')

#######################################################
### Make the graph of f1 score vs tumour types for TCGA
print('Making the graph of f1 scores vs tumour types ...')
print('')
all_pred_df_w_t_type = all_pred_df.merge(tcga_t_type, on='p_id')

# find all unique cancer types
tcga_cancer_types = all_pred_df_w_t_type.tumour_type_abbv.unique()

# calculate precision, recall, f1-score and ratio of mutant ot WT or WT to mutant samples (whichever that's smaller)
tcga_cancer_types_measures = pd.DataFrame({'type':['a'], 'num_mut':[-1], 'num_wt':[-1], 'precision':[-1.0], 'recall':[-1.0], 
                                            'f1_score':[-1.0], 'min_to_maj_clss_ratio':[-1.0]})

for i in tcga_cancer_types:
    df = all_pred_df_w_t_type[all_pred_df_w_t_type.tumour_type_abbv == i]
    measures = sklearn.metrics.precision_recall_fscore_support(df.status, df.predict, average='macro', zero_division=0)
    
    cancer_type = i
    num_mut_smpl = df[df.status == 'mut'].shape[0]
    num_wt_smpl = df[df.status == 'wt'].shape[0]
    prcsn = np.round(measures[0], 2)
    rcll = np.round(measures[1], 2)
    f1scr = np.round(measures[2], 2)
    clss_ratio = np.round(num_mut_smpl / num_wt_smpl, 2)
    if clss_ratio > 1:
        clss_ratio = np.round(num_wt_smpl / num_mut_smpl, 2)
    
    tcga_type_and_measure = pd.DataFrame({'type':[i], 'num_mut':num_mut_smpl, 'num_wt':num_wt_smpl, 'precision':prcsn, 'recall':rcll, 
                                            'f1_score':f1scr, 'min_to_maj_clss_ratio':clss_ratio})
    tcga_cancer_types_measures = pd.concat([tcga_cancer_types_measures, tcga_type_and_measure], axis=0)
        
tcga_cancer_types_measures = tcga_cancer_types_measures[tcga_cancer_types_measures.type != 'a']

# make the graph of F1 scores against the ratio of minor to major class size
fig_dims = (12, 9)
fig, ax = plt.subplots(figsize=fig_dims)
f1_to_ratio_fig = sns.scatterplot(data=tcga_cancer_types_measures, x='min_to_maj_clss_ratio', y='f1_score', hue='type')

# make lists to keep track of x and y locations on the graph for adding tumour type without overlapping text
x_y_loc_list = []

for t in range(tcga_cancer_types_measures.shape[0]):
    x_loc = tcga_cancer_types_measures.min_to_maj_clss_ratio.iloc[t]
    y_loc = tcga_cancer_types_measures.f1_score.iloc[t]
    
    adj_x_loc = np.max(tcga_cancer_types_measures.min_to_maj_clss_ratio) * 0.011
    adj_y_loc = 0.005
    
    # make sure labels do not overlap
    if (np.round(x_loc + adj_x_loc, 3), np.round(y_loc + adj_y_loc, 3)) in x_y_loc_list:
      new_x_loc = np.round(x_loc - adj_x_loc, 3)
      new_y_loc = np.round(y_loc - adj_y_loc, 3)
      new_h_aligh = 'right'
    else:
      new_x_loc = np.round(x_loc + adj_x_loc, 3)
      new_y_loc = np.round(y_loc + adj_y_loc, 3)
      new_h_aligh = 'left'
         
    f1_to_ratio_fig.text(new_x_loc, new_y_loc, tcga_cancer_types_measures.type.iloc[t], horizontalalignment=new_h_aligh, size='small')
    
    # add new locations to the list
    x_y_loc_list.append((new_x_loc, new_y_loc))
    
    # add more locations to the above list to avoid label overlap
    for num in np.round(np.arange(0.005,0.039,0.005), 3):
      x_y_loc_list.append((new_x_loc+num, new_y_loc))

plt.legend(bbox_to_anchor=(1, 1.02))
f1_to_ratio_fig.set(xlabel='The Ratio of Minor to Major Class Size', ylabel='F1 Score')

fig.savefig(snakemake.output.f1_to_min_maj_ratio_plot_SNVs_only,format='jpeg',dpi=400,bbox_inches='tight')

print('Model performance is evaluated and plots are made . . .')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
