##############################################################################################################
## This is a script to test the performance of random forest on the set of selected tumour types that showed
## the best performance for each gene of interest.
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
if best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_only':
    X_new = X.loc[(X.index.isin(spcfc_p_ids)==True),]
    y_df = pd.DataFrame({'p_id':X.index,'y':y})
elif best_setting[best_setting.gene==gene_of_interest].best_setting == 'SNV_CNV':
    X_new = X_cnv.loc[(X_cnv.index.isin(spcfc_p_ids)==True),]
    y_df = pd.DataFrame({'p_id':X_cnv.index,'y':y})

y_new = y_df.loc[(y_df.p_id.isin(spcfc_p_ids)),]
y_new = y_new.y

#######################################################################################
######### Performing 5-fold CV to get predictions on all TCGA and POG samples #########
#######################################################################################
print('Performing 5-fold CV on TCGA and POG samples from selected tumour types ...')
print('')

skf = StratifiedKFold(n_splits=5, shuffle=True)

clf = RandomForestClassifier(n_estimators=3000, max_depth=int(best_max_depth), max_features=float(best_max_features), 
                             max_samples=float(best_max_samples), min_samples_split=int(best_min_samples_split), 
                             min_samples_leaf=int(best_min_samples_leaf), n_jobs=40)

# extract sample ids that belong to tumour types of interest
pog_p_ids = pog_t_type[pog_t_type.tumour_type_abbv.isin(tumour_type_dict[gene_of_interest])].p_id

# make X and y with TCGA and POG samples from specific tumour types
if len(pog_p_ids) != 0:
    X_tcga_pog = pd.concat([X_new, X_pog_new], axis=0)
    y_tcga_pog = pd.concat([y_new, y_pog_new], axis=0)

    # random forest performance on tcga and pog
    all_pred_df = pd.DataFrame({'p_id':['a'], 'status':['mut_wt'], 'predict':['mut_wt']})
    all_prob = []
    true_label_prob = np.empty([0,])

    for train_index, test_index in skf.split(X_tcga_pog, y_tcga_pog):

        y_train, y_test = y_tcga_pog.iloc[train_index], y_tcga_pog.iloc[test_index]
        X_train, X_test = X_tcga_pog.iloc[train_index], X_tcga_pog.iloc[test_index]

        clf.fit(X_train, y_train)

        sample_ids = X_test.index.values

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
    f_3 = open('rf_results/'+str(gene_of_interest)+'/cv_tcga_pog_specific_tumour_types_results.txt', 'w')

    print(confusion_matrix(all_pred_df.status, all_pred_df.predict, labels=['mut','wt']), file=f_3)
    print(classification_report(all_pred_df.status, all_pred_df.predict), file=f_3)

    both_auprc = sklearn.metrics.average_precision_score(all_pred_df.status, true_label_prob, pos_label="wt")
    both_auroc = sklearn.metrics.roc_auc_score(all_pred_df.status, true_label_prob)
    print('AUPRC:', file=f_3)
    print(both_auprc, file=f_3)
    print('AUROC:', file=f_3)
    print(both_auroc, file=f_3)

    f_3.close()

print('Classification is performed on imbalanced set of selected tumour types.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
