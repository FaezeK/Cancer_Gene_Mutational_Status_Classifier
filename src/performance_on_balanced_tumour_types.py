##############################################################################################################
## This is a script to test the performance of random forest on the balanced set of selected tumour types 
## that showed the best performance for each gene of interest.
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

# read expression data of samples with non-impactful mutations
tcga_tpm_not_impactful_mut = pd.read_csv(snakemake.input.tcga_tpm_not_impactful_mut, delimiter = '\t', header=0)
pog_tpm_not_impactful_mut = pd.read_csv(snakemake.input.pog_tpm_not_impactful_mut, delimiter = '\t', header=0)

# read processed mutation data
tcga_all_mut = pd.read_csv(snakemake.input.tcga_mut_prcssd, delimiter = '\t', header=0)
pog_all_mut = pd.read_csv(snakemake.input.pog_mut_prcssd, delimiter = '\t', header=0)

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

# extract TCGA and POG sample ids that exist in tumour types with significant f1-score and balance sets
all_samples_to_keep = pd.Series(dtype='str')
all_t_type = pd.concat([tcga_t_type, pog_t_type], axis=0)

for t in tumour_type_dict[gene_of_interest]:
    t_type_smpls = all_t_type[all_t_type.tumour_type_abbv==t].p_id
    
    # obtain samples with impactful mutations
    tcga_mut_smpls = tcga_tpm_impactful_mut[tcga_tpm_impactful_mut.index.isin(t_type_smpls)].index
    pog_mut_smpls = pog_tpm_impactful_mut[pog_tpm_impactful_mut.index.isin(t_type_smpls)].index
    mut_smpls = pd.concat([pd.Series(tcga_mut_smpls), pd.Series(pog_mut_smpls)])

    # obtain samples with wild-type copies
    tcga_wt_smpls = tcga_tpm_wt[tcga_tpm_wt.index.isin(t_type_smpls)].index
    pog_wt_smpls = pog_tpm_wt[pog_tpm_wt.index.isin(t_type_smpls)].index
    wt_smpls = pd.concat([pd.Series(tcga_wt_smpls), pd.Series(pog_wt_smpls)])
    
    # down-sample the wild-type category
    if len(mut_smpls) < len(wt_smpls):
        wt_samples_to_keep = wt_smpls.sample(n=len(mut_smpls))
        samples_to_keep = pd.concat([mut_smpls, wt_samples_to_keep])
        all_samples_to_keep = pd.concat([all_samples_to_keep, samples_to_keep])
    
    # down-sample the mutant category
    else:
        mut_samples_to_keep = mut_smpls.sample(n=len(wt_smpls))
        samples_to_keep = pd.concat([wt_smpls, mut_samples_to_keep])
        all_samples_to_keep = pd.concat([all_samples_to_keep, samples_to_keep])

# filter X and y based on the samples found above
if best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_only':
    X_new = X[X.index.isin(all_samples_to_keep)]
    y_df = pd.DataFrame({'p_id':X.index, 'label':y})
elif best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_CNV':
    X_new = X_cnv[X_cnv.index.isin(all_samples_to_keep)]
    y_df = pd.DataFrame({'p_id':X_cnv.index, 'label':y})

X_new = X_new.sort_index()
y_new = y_df[y_df.p_id.isin(all_samples_to_keep)]
y_new = y_new.sort_values(by=['p_id'])
y_new = y_new.label

# train the model with filtered X and y
clf = RandomForestClassifier(n_estimators=3000, max_depth=int(best_max_depth), max_features=float(best_max_features), 
                             max_samples=float(best_max_samples), min_samples_split=int(best_min_samples_split), 
                             min_samples_leaf=int(best_min_samples_leaf), n_jobs=40)

clf.fit(X_new, y_new)

# extract important features in classification
rand_f_scores = clf.feature_importances_
indices = np.argsort(rand_f_scores)
rand_f_scores_sorted = pd.Series(np.sort(rand_f_scores))
rand_forest_importance_scores_true_df = pd.DataFrame({'gene':pd.Series(X.columns[indices]), 'importance_score':rand_f_scores_sorted})
rand_forest_importance_scores_true_df = rand_forest_importance_scores_true_df.sort_values(by='importance_score', ascending=False)
rand_forest_importance_scores_true_df.to_csv(snakemake.output.str(gene_of_interest)+'_gene_importance_scores_specific_tumour_types_balanced.txt', sep='\t', index=False)

###################################################################
######### Testing on samples with not-impactful mutations #########
###################################################################
print('Testing RF on samples with not-impactful mutations ...')
print('')

# extract sample ids that exist in tumour types with significant f1-score
req_p_ids = all_t_type[all_t_type.tumour_type_abbv.isin(tumour_type_dict[gene_of_interest])].p_id

X_not_impact_tcga = tcga_tpm_not_impactful_mut.loc[tcga_tpm_not_impactful_mut.index.isin(req_p_ids)==True,]
X_not_impact_pog = pog_tpm_not_impactful_mut.loc[pog_tpm_not_impactful_mut.index.isin(req_p_ids)==True,]
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

    not_impact_preds_w_conseq_df.to_csv(snakemake.output.str(gene_of_interest)+'_pred_on_sample_w_not_impact_mut_specific_tumour_types_balanced.txt', sep='\t', index=False)

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

    val_cnts_df.to_csv(snakemake.output.str(gene_of_interest)+'_not_impact_mut_groups_pred_n_binom_p_val_balanced.txt', sep='\t', index=False)

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
                                            'base_change':val_cnt_not_impact.index.get_level_values(1), 'cnt':val_cnt_not_impact})
                val_cnt_not_impact_df = val_cnt_not_impact_df.reset_index(drop=True)
                val_cnt_not_impact_df.to_csv(snakemake.output.str(gene_of_interest)+'_not_impact_'+str(conseq)+'_base_n_aa_changes_balanced.txt', sep='\t', index=False)

#######################################################################################
######### Performing 5-fold CV to get predictions on all TCGA and POG samples #########
#######################################################################################
print('Performing 5-fold CV on TCGA and POG samples ...')
print('')

skf = StratifiedKFold(n_splits=5, shuffle=True)

clf = RandomForestClassifier(n_estimators=3000, max_depth=int(best_max_depth), max_features=float(best_max_features), 
                             max_samples=float(best_max_samples), min_samples_split=int(best_min_samples_split), 
                             min_samples_leaf=int(best_min_samples_leaf), n_jobs=40)

# random forest performance on tcga and pog
all_pred_df = pd.DataFrame({'p_id':['a'], 'status':['mut_wt'], 'predict':['mut_wt']})
all_prob = []
true_label_prob = np.empty([0,])

for train_index, test_index in skf.split(X_new, y_new):

    y_train, y_test = y_new.iloc[train_index], y_new.iloc[test_index]
    X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]

    # train model on 80% of samples from balanced set of selected tumour types
    clf.fit(X_train, y_train)

    sample_ids = X_test.index.values

    # test the model on the remaining 20% of samples
    rf_predictions = clf.predict(X_test)
    rf_pred_df = pd.DataFrame({'p_id':sample_ids, 'status':y_test, 'predict':rf_predictions})
    all_pred_df = pd.concat([all_pred_df, rf_pred_df], axis=0)

    tcga_pog_prob = clf.predict_proba(X_test)
    for i in tcga_pog_prob:
        rf_prob=max(i)
        all_prob.append(rf_prob)
    
    if clf.classes_[0]=="wt":
        true_prob = tcga_pog_prob[:,0]
    else:
        true_prob = tcga_pog_prob[:, 1]
    true_label_prob = np.concatenate((true_label_prob, true_prob), axis=0)
    
all_pred_df = all_pred_df[all_pred_df.p_id != 'a']

# assess performance
f_1 = open(snakemake.output.str(gene_of_interest)+'_balanced_t_types_cv_results.txt', 'w')

print(confusion_matrix(all_pred_df.status, all_pred_df.predict, labels=['mut','wt']), file=f_1)
print(classification_report(all_pred_df.status, all_pred_df.predict), file=f_1)

both_auprc = sklearn.metrics.average_precision_score(all_pred_df.status, true_label_prob, pos_label="wt")
both_auroc = sklearn.metrics.roc_auc_score(all_pred_df.status, true_label_prob)
print('AUPRC:', file=f_1)
print(both_auprc, file=f_1)
print('AUROC:', file=f_1)
print(both_auroc, file=f_1)

f_1.close()

print('Classification is performed on balanced set of selected tumour types.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
