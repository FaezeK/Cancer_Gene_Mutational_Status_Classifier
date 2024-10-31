################################################################################################################
## This is a script to test the performance of random forest on the balanced set of selected tumour types in 
## permutations to find the best analysis settings for each gene of interest.
################################################################################################################

# import required libraries
import pandas as pd
import numpy as np
import timeit
import test_performance_helper as tph
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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

# read expression files made using both SNV and CNV data
tcga_tpm_impactful_mut_cnv = pd.read_csv(snakemake.input.tcga_tpm_impactful_mut_cnv, delimiter = '\t', header=0, index_col=0)
tcga_tpm_wt_cnv = pd.read_csv(snakemake.input.tcga_tpm_wt_cnv, delimiter = '\t', header=0, index_col=0)

pog_tpm_impactful_mut_cnv = pd.read_csv(snakemake.input.pog_tpm_impactful_mut_cnv, delimiter = '\t', header=0, index_col=0)
pog_tpm_wt_cnv = pd.read_csv(snakemake.input.pog_tpm_wt_cnv, delimiter = '\t', header=0, index_col=0)

################################################################################################################
######### Training RF with samples containing impactful mutations or wilt-type copies from tumour types ########
######### that had a significant f1-score when balanced set of samples were used for training/testing. #########
################################################################################################################
print('Training RF ...')
print('')

# For each of the 50 genes, below dictionary contains the list of tumour types with significant f1-score compared across all 
# tumour types using a z-test with alpha=0.1
# Note that for genes without tumour types with significan f1-score, the tumour type with highest f1-score was added to the  
# below dictionary.
tumour_type_dict = {'APC':['COADREAD'],
                    'AR':['THCA'],
                    'ARID1A':['KICH', 'KIRP', 'LGG', 'PCPG', 'THYM', 'UVM'],
                    'ASXL1':['COADREAD'],
                    'ATM':['BRCA', 'CESC', 'PCPG', 'TGCT'],
                    'ATR':['CESC', 'KIRP', 'PCPG'],
                    'ATRX':['LGG', 'UCEC'],
                    'BRAF':['COADREAD', 'THCA'],
                    'BRCA1':['KICH', 'KIRP', 'UCEC'],
                    'BRCA2':['COADREAD'],
                    'CDH1':['BRCA', 'KIRP', 'LIHC', 'OV', 'PRAD', 'UVM'],
                    'CDK12':['KIRP', 'UCEC'],
                    'CDKN2A':['KIRC', 'KIRP', 'PAAD', 'THCA', 'UCEC'],
                    'CTCF':['BRCA', 'KIRP', 'LIHC', 'PRAD', 'THYM'],
                    'CTNNB1':['HNSC', 'KIRC', 'PCPG', 'UVM'],
                    'EGFR':['COADREAD', 'KIRP', 'LGG'],
                    'EP300':['THCA', 'UVM'],
                    'ERBB4':['CESC'],
                    'EZH2':['KIRP', 'LGG', 'THCA', 'THYM'],
                    'FBXW7':['LIHC', 'MESO'],
                    'FLT3':['COADREAD'],
                    'GATA3':['KIRP', 'LGG'],
                    'KDM6A':['KIRC'],
                    'KEAP1':['THCA'],
                    'KIT':['KIRC'],
                    'KRAS':['PAAD', 'STAD'],
                    'MAP3K1':['STAD', 'THCA'],
                    'MECOM':['LUSC', 'PCPG', 'UVM'],
                    'MTOR':['LGG', 'PCPG'],
                    'NCOR1':['BRCA', 'COADREAD', 'KICH', 'KIRP', 'LAML', 'LIHC', 'PCPG', 'THCA'],
                    'NF1':['KICH', 'KIRP', 'PCPG'],
                    'NFE2L2':['KICH', 'THCA'],
                    'NOTCH1':['KIRC'],
                    'NRAS':['LGG', 'PCPG', 'THCA', 'UCEC'],
                    'NSD1':['KICH', 'KIRC', 'PRAD', 'THYM'],
                    'PBRM1':['HNSC', 'KIRC', 'MESO', 'PCPG', 'THYM', 'UVM'],
                    'PDGFRA':['KICH'],
                    'PIK3CA':['HNSC','KIRP', 'LUSC', 'PCPG', 'UVM'],
                    'PIK3R1':['THYM'],
                    'POLQ':['BRCA', 'HNSC'],
                    'PTEN':['LGG', 'PRAD', 'SARC', 'SKCM'],
                    'RB1':['COADREAD', 'LGG', 'LIHC', 'PRAD', 'SARC'],
                    'SETBP1':['COADREAD', 'HNSC', 'PRAD'],
                    'SETD2':['HNSC', 'KIRC', 'PCPG', 'UVM'],
                    'SF3B1':['KICH'],
                    'SMAD4':['COADREAD', 'THYM'],
                    'SPOP':['KICH', 'KIRP', 'THCA', 'THYM'],
                    'STAG2':['THCA'],
                    'TET2':['MESO'],
                    'TP53':['BRCA', 'COADREAD', 'LGG', 'LIHC', 'LUAD', 'SKCM', 'UCEC']}

# extract sample ids that exist in tumour types with significant f1-score and balance sets
all_samples_to_keep = pd.Series(dtype='str')
all_t_type = pd.concat([tcga_t_type, pog_t_type], axis=0)

for t in tumour_type_dict[gene_of_interest]:
    t_type_smpls = all_t_type[all_t_type.tumour_type_abbv==t].p_id
    
    # obtain the number of samples with impactful mutations
    if best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_only':
        tcga_mut_smpls = tcga_tpm_impactful_mut[tcga_tpm_impactful_mut.index.isin(t_type_smpls)].index
        pog_mut_smpls = pog_tpm_impactful_mut[pog_tpm_impactful_mut.index.isin(t_type_smpls)].index
        mut_smpls = pd.concat([pd.Series(tcga_mut_smpls), pd.Series(pog_mut_smpls)])

        # obtain the number of samples with wild-type copies
        tcga_wt_smpls = tcga_tpm_wt[tcga_tpm_wt.index.isin(t_type_smpls)].index
        pog_wt_smpls = pog_tpm_wt[pog_tpm_wt.index.isin(t_type_smpls)].index
        wt_smpls = pd.concat([pd.Series(tcga_wt_smpls), pd.Series(pog_wt_smpls)])
        
    elif best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_CNV':
        tcga_mut_smpls = tcga_tpm_impactful_mut_cnv[tcga_tpm_impactful_mut_cnv.index.isin(t_type_smpls)].index
        pog_mut_smpls = pog_tpm_impactful_mut_cnv[pog_tpm_impactful_mut_cnv.index.isin(t_type_smpls)].index
        mut_smpls = pd.concat([pd.Series(tcga_mut_smpls), pd.Series(pog_mut_smpls)])

        # obtain the number of samples with wild-type copies
        tcga_wt_smpls = tcga_tpm_wt_cnv[tcga_tpm_wt_cnv.index.isin(t_type_smpls)].index
        pog_wt_smpls = pog_tpm_wt_cnv[pog_tpm_wt_cnv.index.isin(t_type_smpls)].index
        wt_smpls = pd.concat([pd.Series(tcga_wt_smpls), pd.Series(pog_wt_smpls)])
    
    # down-sample wild-type category
    if len(mut_smpls) < len(wt_smpls):
        wt_samples_to_keep = wt_smpls.sample(n=len(mut_smpls))
        samples_to_keep = pd.concat([mut_smpls, wt_samples_to_keep])
        all_samples_to_keep = pd.concat([all_samples_to_keep, samples_to_keep])
    
    # down-sample mutant category
    else:
        mut_samples_to_keep = mut_smpls.sample(n=len(wt_smpls))
        samples_to_keep = pd.concat([wt_smpls, mut_samples_to_keep])
        all_samples_to_keep = pd.concat([all_samples_to_keep, samples_to_keep])

# filter X and y based on the samples found above
if best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_only':
    X_new = X[X.index.isin(all_samples_to_keep)]
    y_new = y[y.index.isin(all_samples_to_keep)]
elif best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_CNV':
    X_new = X_cnv[X_cnv.index.isin(all_samples_to_keep)]
    y_new = y_cnv[y_cnv.index.isin(all_samples_to_keep)]
    
# convert label dataframe to vector
y_new = y_new.y

#######################################################################################
######### Performing 5-fold CV to get predictions on all TCGA and POG samples #########
#######################################################################################
print('Performing 5-fold CV on TCGA and POG samples ...')
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

skf = StratifiedKFold(n_splits=5, shuffle=True)

precision_scores = []
recall_scores = []
f1_scores = []
accuracies = []

for i in range(30):
    print('Round '+str(i+1)+' of CV ...')

    # random forest performance on tcga and pog
    all_pred_df, all_prob, true_label_prob = tph.test_performance_5_fold_CV(clf, skf, X_new, y_new)
    
    # obtain performance metrics
    precision, recall, f1, support = precision_recall_fscore_support(all_pred_df.status, all_pred_df.predict, average='macro')
    
    # add above metrics to the list of permutations
    precision_scores.append(round(precision, ndigits=2))
    recall_scores.append(round(recall, ndigits=2))
    f1_scores.append(round(f1, ndigits=2))
    accuracies.append(round(accuracy_score(all_pred_df.status, all_pred_df.predict), ndigits=2))

# assess performance
f_1 = open(snakemake.output.permut_balanced_results_selected_tumours, 'w')

print('Avg. Precision:', round(np.mean(precision_scores), ndigits=4), file=f_1)
print('Std. Precision:', round(np.std(precision_scores), ndigits=4), file=f_1)
        
print('Avg. Recall:', round(np.mean(recall_scores), ndigits=4), file=f_1)
print('Std. Recall:', round(np.std(recall_scores), ndigits=4), file=f_1)

print('Avg. F1 Score:', round(np.mean(f1_scores), ndigits=4), file=f_1)
print('Std. F1 Score:', round(np.std(f1_scores), ndigits=4), file=f_1)

print('Avg. Accuracy:', round(np.mean(accuracies), ndigits=4), file=f_1)
print('Std. Accuracy:', round(np.std(accuracies), ndigits=4), file=f_1)

f_1.close()

print('Classification is performed on the balanced set of selected tumour types.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
