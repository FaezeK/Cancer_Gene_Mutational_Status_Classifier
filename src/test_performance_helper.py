##############################################################################################################
## This is a script to provide the methods needed to test the performance of random forest model.
##############################################################################################################

import pandas as pd
import numpy as np
import sklearn.metrics

# function to perform 5-fold CV using the given the model, X and y
def test_performance_5_fold_CV(clf, skf, X, y):
    
    all_pred_df = pd.DataFrame({'p_id':['a'], 'status':['mut_wt'], 'predict':['mut_wt']})
    all_prob = []
    true_label_prob = np.empty([0,])
    
    # test in 5-fold CV
    for train_index, test_index in skf.split(X, y):

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        # train on 80% of data
        clf.fit(X_train, y_train)

        # test on 20% of data
        sample_ids = X_test.index.values

        rf_predictions = clf.predict(X_test)
        rf_pred_df = pd.DataFrame({'p_id':sample_ids, 'status':y_test, 'predict':rf_predictions})
        all_pred_df = pd.concat([all_pred_df, rf_pred_df], axis=0)

        # extract prediction probabilities
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
    
    return all_pred_df, all_prob, true_label_prob

# function to make AUROC graph
def generate_auroc_curve(tru_stat, tru_lab_prob, pos):
    data_fpr, data_tpr, data_thresholds = sklearn.metrics.roc_curve(tru_stat, tru_lab_prob, pos_label="wt")
    data_fpr_tpr = pd.DataFrame({'fpr':data_fpr, 'tpr':data_tpr})
    sns.lineplot(data=data_fpr_tpr, x='fpr', y='tpr', ax=pos)
    pos.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    pos.plot([0, 1], [0, 1], color='black', ls='--')

# function to make AUPRC graph
def generate_auprc_curve(tru_stat, tru_lab_prob, pos):
    data_prcsn, data_rcll, data_thrshlds = sklearn.metrics.precision_recall_curve(tru_stat, tru_lab_prob, pos_label="wt")
    data_prcsn_rcll = pd.DataFrame({'prcsn':data_prcsn, 'rcll':data_rcll})
    sns.lineplot(data=data_prcsn_rcll, x='rcll', y='prcsn', ax=pos)
    pos.set(xlabel='Recall', ylabel='Precision')
    pos.plot([0, 1], [1, 0], color='black', ls='--')
