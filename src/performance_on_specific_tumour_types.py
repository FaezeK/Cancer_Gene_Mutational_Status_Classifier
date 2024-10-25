##############################################################################################################
## This is a script to test the performance of random forest on the set of selected tumour types that showed
## the best performance for each gene of interest.
##############################################################################################################

# import required libraries
import pandas as pd
import numpy as np
import timeit
import test_performance_helper as tph
import sklearn.metrics
from scipy import stats
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

# read expression data of samples with non-impactful mutations
tcga_tpm_not_impactful_mut = pd.read_csv(snakemake.input.tcga_tpm_not_impactful_mut, delimiter = '\t', header=0, index_col=0)
pog_tpm_not_impactful_mut = pd.read_csv(snakemake.input.pog_tpm_not_impactful_mut, delimiter = '\t', header=0, index_col=0)

# read processed mutation data
tcga_all_mut = pd.read_csv(snakemake.input.tcga_mut_prcssd, delimiter = '\t', header=0)
pog_all_mut = pd.read_csv(snakemake.input.pog_mut_prcssd, delimiter = '\t', header=0)

################################################################################################################
######### Training RF with TCGA samples containing impactful mutations or wilt-type copies from tumour #########
######### types that had a f1-score > 0.728 when all samples were used for training/testing. ###################
################################################################################################################
print('Training RF ...')
print('')

# For each of the 50 genes, below dictionary contains the list of tumour types with significant f1-score compared across all 
# tumour types using a z-test with alpha=0.1
# Note that for genes without tumour types with significan f1-score, the tumour type with highest f1-score was added to the  
# below dictionary.
tumour_type_dict = {'APC':['COADREAD'],
                    'AR':['KIRP'],
                    'ARID1A':['BRCA', 'LGG', 'LIHC', 'PCPG', 'UVM'],
                    'ASXL1':['BRCA','CESC'],
                    'ATM':['BRCA', 'CESC', 'PCPG'],
                    'ATR':['HNSC', 'KIRP', 'PCPG'],
                    'ATRX':['LGG'],
                    'BRAF':['COADREAD', 'SKCM', 'THCA'],
                    'BRCA1':['BRCA', 'KICH', 'KIRP', 'UCEC'],
                    'BRCA2':['BRCA'],
                    'CDH1':['KIRP', 'LIHC', 'PRAD', 'SARC', 'THYM', 'UCEC', 'UVM'],
                    'CDK12':['BRCA', 'KICH', 'KIRP', 'SARC', 'UCEC'],
                    'CDKN2A':['HNSC', 'KIRC', 'LGG', 'MESO', 'PAAD', 'UCEC'],
                    'CTCF':['KIRP', 'LIHC', 'PRAD', 'SARC', 'THYM', 'UVM'],
                    'CTNNB1':['BRCA', 'CESC', 'HNSC', 'KIRC', 'LIHC', 'PCPG', 'UVM'],
                    'EGFR':['COADREAD', 'HNSC', 'KIRP', 'LGG', 'STAD', 'THCA'],
                    'EP300':['BRCA', 'PCPG', 'THCA', 'UCEC'],
                    'ERBB4':['CESC', 'PAAD'],
                    'EZH2':['COADREAD', 'KIRP', 'LGG', 'THCA', 'THYM'],
                    'FBXW7':['LIHC', 'MESO', 'UCEC'],
                    'FLT3':['UCEC'],
                    'GATA3':['LGG', 'SARC'],
                    'KDM6A':['KIRP', 'UCEC'],
                    'KEAP1':['GBM', 'LUAD', 'STAD', 'UCEC'],
                    'KIT':['ACC', 'KICH'],
                    'KRAS':['COADREAD', 'PAAD'],
                    'MAP3K1':['BLCA', 'HNSC', 'PRAD', 'STAD'],
                    'MECOM':['UCEC', 'UVM'],
                    'MTOR':['BRCA', 'LGG', 'PCPG'],
                    'NCOR1':['BRCA', 'COADREAD', 'KICH', 'KIRP', 'LIHC', 'PAAD', 'PCPG', 'UCEC'],
                    'NF1':['BRCA', 'KICH', 'KIRP', 'PCPG', 'SARC', 'UCEC'],
                    'NFE2L2':['KICH', 'THCA'],
                    'NOTCH1':['KIRC'],
                    'NRAS':['LGG', 'PCPG'],
                    'NSD1':['BLCA', 'HNSC', 'KIRC'],
                    'PBRM1':['BRCA', 'CESC', 'HNSC', 'KIRC', 'KIRP', 'PCPG', 'UVM'],
                    'PDGFRA':['KICH'],
                    'PIK3CA':['KIRP', 'PCPG', 'UVM'],
                    'PIK3R1':['BLCA', 'HNSC', 'STAD'],
                    'POLQ':['BRCA'],
                    'PTEN':['LGG', 'PRAD', 'SARC', 'SKCM'],
                    'RB1':['BRCA', 'CESC', 'COADREAD', 'GBM', 'LGG', 'LIHC', 'PRAD', 'SARC'],
                    'SETBP1':['HNSC', 'PRAD'],
                    'SETD2':['BRCA', 'CESC', 'HNSC', 'KIRC', 'KIRP', 'PCPG', 'UVM'],
                    'SF3B1':['UVM'],
                    'SMAD4':['STAD'],
                    'SPOP':['BRCA', 'KICH', 'KIRP'],
                    'STAG2':['KIRP'],
                    'TET2':['LIHC', 'MESO'],
                    'TP53':['BLCA', 'BRCA', 'COADREAD', 'HNSC', 'LGG', 'LIHC', 'LUAD', 'STAD', 'UCEC']}

# extract TCGA and POG sample ids that exist in tumour types with significant f1-score 
all_t_type = pd.concat([tcga_t_type, pog_t_type], axis=0)
spcfc_t_types = all_t_type[all_t_type.tumour_type_abbv.isin(tumour_type_dict[gene_of_interest])]
spcfc_p_ids = spcfc_t_types[spcfc_t_types.tumour_type_abbv.isin(tumour_type_dict[gene_of_interest])].p_id

# filter X and y based on above ids
if best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_only':
    X_new = X.loc[(X.index.isin(spcfc_p_ids)==True),]
    y_new = y.loc[(y.index.isin(spcfc_p_ids)==True),]
elif best_setting[best_setting.gene==gene_of_interest].best_setting.iloc[0] == 'SNV_CNV':
    X_new = X_cnv.loc[(X_cnv.index.isin(spcfc_p_ids)==True),]
    y_new = y_cnv.loc[(y_cnv.index.isin(spcfc_p_ids)==True),]

#######################################################################################
######### Performing 5-fold CV to get predictions on all TCGA and POG samples #########
#######################################################################################
print('Performing 5-fold CV on TCGA and POG samples from selected tumour types ...')
print('')

# extracting best hyperparameters
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

clf = RandomForestClassifier(n_estimators=3000, max_depth=best_max_depth, max_features=best_max_features, 
                             max_samples=best_max_samples, min_samples_split=best_min_samples_split, 
                             min_samples_leaf=best_min_samples_leaf, n_jobs=40)

skf = StratifiedKFold(n_splits=5, shuffle=True)

# random forest performance on tcga and pog
y_new = y_new.y
all_pred_df, all_prob, true_label_prob = tph.test_performance_5_fold_CV(clf, skf, X_new, y_new)

# assess performance
f_1 = open(snakemake.output.specific_t_types_cv_results, 'w')

print(confusion_matrix(all_pred_df.status, all_pred_df.predict, labels=['mut','wt']), file=f_1)
print(classification_report(all_pred_df.status, all_pred_df.predict), file=f_1)

both_auprc = sklearn.metrics.average_precision_score(all_pred_df.status, true_label_prob, pos_label="wt")
both_auroc = sklearn.metrics.roc_auc_score(all_pred_df.status, true_label_prob)
print('AUPRC:', file=f_1)
print(both_auprc, file=f_1)
print('AUROC:', file=f_1)
print(both_auroc, file=f_1)

f_1.close()

#############################################################################
######### Training the model on all samples and extracting features #########
#############################################################################
print('Training RF on all samples and extracting features ...')
print('')

clf.fit(X_new, y_new)

# important features in classification
rand_f_scores = clf.feature_importances_
indices = np.argsort(rand_f_scores)
rand_f_scores_sorted = pd.Series(np.sort(rand_f_scores))
rand_forest_importance_scores_true_df = pd.DataFrame({'gene':pd.Series(X_new.columns[indices]), 'importance_score':rand_f_scores_sorted})
rand_forest_importance_scores_true_df = rand_forest_importance_scores_true_df.sort_values(by='importance_score', ascending=False)
rand_forest_importance_scores_true_df.to_csv(snakemake.output.gene_importance_scores_specific_tumour_types, sep='\t', index=False)

###################################################################
######### Testing on samples with not-impactful mutations #########
###################################################################
print('Testing RF on samples with not-impactful mutations ...')
print('')

X_not_impact_tcga = tcga_tpm_not_impactful_mut.loc[tcga_tpm_not_impactful_mut.index.isin(spcfc_p_ids)==True,]
X_not_impact_pog = pog_tpm_not_impactful_mut.loc[pog_tpm_not_impactful_mut.index.isin(spcfc_p_ids)==True,]

X_not_impact = pd.concat([X_not_impact_tcga, X_not_impact_pog], axis=0)

if X_not_impact.shape[0] != 0:
    not_impact_preds_specific_tumour_types = clf.predict(X_not_impact)
    not_impact_preds_specific_tumour_types_df = pd.DataFrame({'p_id':X_not_impact.index, 'pred':not_impact_preds_specific_tumour_types})

    # add the mutation consequences of the above samples to the dataframe
    tcga_not_impact_mut = tcga_all_mut[(tcga_all_mut.p_id.isin(tcga_tpm_not_impactful_mut.index)) & (tcga_all_mut.gene_name == gene_of_interest)]
    pog_not_impact_mut = pog_all_mut[(pog_all_mut.p_id.isin(pog_tpm_not_impactful_mut.index)) & (pog_all_mut.gene_name == gene_of_interest)]

    tcga_not_impact_mut['consequence_ls'] = tcga_not_impact_mut.consequence.str.split('\&')
    tcga_not_impact_mut = tcga_not_impact_mut.explode('consequence_ls')
    tcga_not_impact_mut = tcga_not_impact_mut.drop(columns=['consequence'])
    tcga_not_impact_mut = tcga_not_impact_mut.rename(columns={"consequence_ls": "consequence"})
    
    pog_not_impact_mut['consequence_ls'] = pog_not_impact_mut.consequence.str.split('\+')
    pog_not_impact_mut = pog_not_impact_mut.explode('consequence_ls')
    pog_not_impact_mut = pog_not_impact_mut.drop(columns=['consequence'])
    pog_not_impact_mut = pog_not_impact_mut.rename(columns={"consequence_ls": "consequence"})
    pog_not_impact_mut = pog_not_impact_mut.drop(columns=['refseq_aa_change'])
    
    # merge tcga_not_impact_mut and pog_not_impact_mut
    all_not_impact_mut = pd.concat([tcga_not_impact_mut, pog_not_impact_mut], axis=0)

    not_impact_preds_w_conseq_df = not_impact_preds_specific_tumour_types_df.merge(all_not_impact_mut, how='inner', on='p_id')

    not_impact_preds_w_conseq_df.to_csv(snakemake.output.pred_on_sample_w_not_impact_mut_specific_tumour_types, sep='\t', index=False)

    # extract the number of samples predicted as mutant or wt per consequence
    val_cnts = not_impact_preds_w_conseq_df[['consequence','pred']].value_counts()
    conseq = val_cnts.index.get_level_values(0)
    pred = val_cnts.index.get_level_values(1)

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
    for conseq in val_cnts_df.conseq.unique():
        tmp = val_cnts_df[val_cnts_df.conseq == conseq]
        if float(tmp.p_val.iloc[0]) < 0.05 and tmp[tmp.pred=='mut'].val.iloc[0] > tmp[tmp.pred=='wt'].val.iloc[0]:
            not_impact_predicted_as_mut = not_impact_preds_w_conseq_df[(not_impact_preds_w_conseq_df.consequence==conseq) & 
                                                           (not_impact_preds_w_conseq_df.pred=='mut')]
            not_impact_predicted_as_mut.loc[not_impact_predicted_as_mut.amino_acid_change.isna(),'amino_acid_change'] = 'NA'
            not_impact_predicted_as_mut.loc[not_impact_predicted_as_mut.base_change.isna(),'base_change'] = 'NA'
            if not_impact_predicted_as_mut.shape[0] != 0:
                val_cnt_not_impact = not_impact_predicted_as_mut[['amino_acid_change','base_change']].value_counts()
                val_cnt_not_impact_df = pd.DataFrame({'amino_acid_change':val_cnt_not_impact.index.get_level_values(0), 
                                            'base_change':val_cnt_not_impact.index.get_level_values(1), 'cnt':val_cnt_not_impact,
                                            'conseq':conseq})
                val_cnt_not_impact_df = val_cnt_not_impact_df.reset_index(drop=True)
                val_cnt_not_impact_df.to_csv(snakemake.output.not_impact_base_n_aa_changes, sep='\t', index=False)

print('Classification is performed on imbalanced set of selected tumour types.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
