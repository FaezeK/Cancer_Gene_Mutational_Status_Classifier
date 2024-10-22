##############################################################################################################
## This is a script to make feature matrix and label vector based on preprocessed expression and all gene
## alteration data (SNVs/INDELs, CNVs, and SVs).
##############################################################################################################

# import required libraries
import pandas as pd
import timeit

# variables provided at run-time
gene_of_interest = snakemake.params.gene_name

# timing the run-time
start_time = timeit.default_timer()

######################################
########## Read input files ##########
######################################
print('Reading input files ...')
print('')

# read expression files
tcga_tpm_impactful_mut = pd.read_csv(snakemake.input.tcga_tpm_impactful_mut, delimiter = '\t', header=0)
tcga_tpm_wt = pd.read_csv(snakemake.input.tcga_tpm_wt, delimiter = '\t', header=0)
tcga_tpm_not_impactful_mut = pd.read_csv(snakemake.input.tcga_tpm_not_impactful_mut, delimiter = '\t', header=0)

pog_tpm_impactful_mut = pd.read_csv(snakemake.input.pog_tpm_impactful_mut, delimiter = '\t', header=0)
pog_tpm_wt = pd.read_csv(snakemake.input.pog_tpm_wt, delimiter = '\t', header=0)
pog_tpm_not_impactful_mut = pd.read_csv(snakemake.input.pog_tpm_not_impactful_mut, delimiter = '\t', header=0)

# read copy number data
tcga_cnv_prcssd = pd.read_csv(snakemake.input.tcga_cnv_prcssd, delimiter = '\t', header=0, index_col=0)
pog_cnv_prcssd = pd.read_csv(snakemake.input.pog_cnv_prcssd, delimiter = '\t', header=0, index_col=0)

# read structral variataion data
tcga_sv_prcssd = pd.read_csv(snakemake.input.tcga_sv_prcssd, delimiter='\t', header=0)
pog_sv_prcssd = pd.read_csv(snakemake.input.pog_sv_prcssd, delimiter='\t', header=0)

######################################################################################
######### Making Updated Feature Matrix and Label Vector for Classification ##########
######################################################################################
print('Making X and y arrays ...')
print('')
# function to move samples from wt or not-impactful to impactful based on CNV data
def move_smpls_based_on_cnv(cnv_df, tpm_impactful_df, tpm_wt_df, tpm_not_impactful_df, gene_of_interest):
    cnv_df_g_of_intrst = cnv_df[cnv_df.index==gene_of_interest]
    cnv_df_g_of_intrst_T = cnv_df_g_of_intrst.T
    mut_smpls_based_cnv = pd.Series(cnv_df_g_of_intrst_T[(cnv_df_g_of_intrst_T[gene_of_interest] != 0 )].index)
    
    tpm_wt_df_to_impact = tpm_wt_df[tpm_wt_df.index.isin(mut_smpls_based_cnv)]
    tpm_not_impact_df_to_impact = tpm_not_impactful_df[tpm_not_impactful_df.index.isin(mut_smpls_based_cnv)]
    
    tpm_impactful_df = pd.concat([tpm_impactful_df, tpm_wt_df_to_impact, tpm_not_impact_df_to_impact])
    tpm_wt_df = tpm_wt_df.loc[~tpm_wt_df.index.isin(tpm_wt_df_to_impact.index),]
    tpm_not_impactful_df = tpm_not_impactful_df.loc[~tpm_not_impactful_df.index.isin(tpm_not_impact_df_to_impact.index),]

    return tpm_impactful_df, tpm_wt_df, tpm_not_impactful_df

# function to move samples from wt or not-impactful to impactful based on presence of SVs
def move_smpls_based_on_sv(sv_df, tpm_impactful_df, tpm_wt_df, tpm_not_impactful_df, gene_of_interest):
    sv_df_g_of_intrst = sv_df[(sv_df.gene1==gene_of_interest) | (sv_df.gene2==gene_of_interest)]
    mut_smpls_based_sv = pd.Series(sv_df_g_of_intrst.p_id.unique())

    tpm_wt_to_impact = tpm_wt_df[tpm_wt_df.index.isin(mut_smpls_based_sv)]
    tpm_not_impactful_to_impact = tpm_not_impactful_df[tpm_not_impactful_df.index.isin(mut_smpls_based_sv)]

    tpm_impactful_df = pd.concat([tpm_impactful_df, tpm_wt_to_impact, tpm_not_impactful_to_impact])
    tpm_wt_df = tpm_wt_df.loc[~tpm_wt_df.index.isin(tpm_wt_to_impact.index),]
    tpm_not_impactful_df = tpm_not_impactful_df.loc[~tpm_not_impactful_df.index.isin(tpm_not_impactful_to_impact.index),]

    return tpm_impactful_df, tpm_wt_df, tpm_not_impactful_df

# make_X_y_merged function creates the feature matrix and label array for merged data
def make_X_y_merged(exp_mut_df1, exp_mut_df2, exp_wt_df1, exp_wt_df2, mut_label, wt_label):

    X = pd.concat([exp_mut_df1, exp_mut_df2, exp_wt_df1, exp_wt_df2])
    y = pd.concat([pd.Series([mut_label]*exp_mut_df1.shape[0]), pd.Series([mut_label]*exp_mut_df2.shape[0]),
                   pd.Series([wt_label]*exp_wt_df1.shape[0]), pd.Series([wt_label]*exp_wt_df2.shape[0])])
    
    return X, y

################################################################
### utilize CNV data and update feature matrix and label vector
tcga_tpm_impactful_mut_w_cnv, tcga_tpm_wt_w_cnv, tcga_tpm_not_impactful_mut_w_cnv = move_smpls_based_on_cnv(tcga_cnv_prcssd, 
                        tcga_tpm_impactful_mut, tcga_tpm_wt, tcga_tpm_not_impactful_mut, gene_of_interest)

pog_tpm_impactful_mut_w_cnv, pog_tpm_wt_w_cnv, pog_tpm_not_impactful_mut_w_cnv = move_smpls_based_on_cnv(pog_cnv_prcssd,
                        pog_tpm_impactful_mut, pog_tpm_wt, pog_tpm_not_impactful_mut, gene_of_interest)

# expression matrix and target label for TCGA and POG samples based on SNV and CNV data
X_cnv, y_cnv = make_X_y_merged(tcga_tpm_impactful_mut_w_cnv, pog_tpm_impactful_mut_w_cnv, tcga_tpm_wt_w_cnv, pog_tpm_wt_w_cnv, 'mut', 'wt')

################################################################
### utilize SV data and update feature matrix and label vector
tcga_tpm_impactful_mut_w_sv, tcga_tpm_wt_w_sv, tcga_tpm_not_impactful_mut_w_sv = move_smpls_based_on_sv(tcga_sv_prcssd, 
                        tcga_tpm_impactful_mut, tcga_tpm_wt, tcga_tpm_not_impactful_mut, gene_of_interest)

pog_tpm_impactful_mut_w_sv, pog_tpm_wt_w_sv, pog_tpm_not_impactful_mut_w_sv = move_smpls_based_on_sv(pog_sv_prcssd, 
                        pog_tpm_impactful_mut, pog_tpm_wt, pog_tpm_not_impactful_mut, gene_of_interest)

# expression matrix and target label for TCGA and POG samples based on SNV and SV data
X_sv, y_sv = make_X_y_merged(tcga_tpm_impactful_mut_w_sv, pog_tpm_impactful_mut_w_sv, tcga_tpm_wt_w_sv, pog_tpm_wt_w_sv, 'mut', 'wt')

#######################################################################
### utilize CNV and SV data and update feature matrix and label vector
tcga_tpm_impactful_mut_all_data, tcga_tpm_wt_all_data, tcga_tpm_not_impactful_mut_all_data = move_smpls_based_on_sv(tcga_sv_prcssd, 
                        tcga_tpm_impactful_mut_w_cnv, tcga_tpm_wt_w_cnv, tcga_tpm_not_impactful_mut_w_cnv, gene_of_interest)

pog_tpm_impactful_mut_all_data, pog_tpm_wt_all_data, pog_tpm_not_impactful_mut_all_data = move_smpls_based_on_sv(pog_sv_prcssd, 
                        pog_tpm_impactful_mut_w_cnv, pog_tpm_wt_w_cnv, pog_tpm_not_impactful_mut_w_cnv, gene_of_interest)

# expression matrix and target label for TCGA and POG samples based on SNV and SV data
X, y = make_X_y_merged(tcga_tpm_impactful_mut_all_data, pog_tpm_impactful_mut_all_data, tcga_tpm_wt_all_data, pog_tpm_wt_all_data, 'mut', 'wt')

#######################################################################
##### Write feature matrices and label vectors into tmp directory #####
#######################################################################

X_cnv.to_csv(snakemake.output.feature_matrix_cnv, sep='\t', index=False)
y_cnv.to_csv(snakemake.output.label_vector_cnv, sep='\t', index=False)

X_sv.to_csv(snakemake.output.feature_matrix_sv, sep='\t', index=False)
y_sv.to_csv(snakemake.output.label_vector_sv, sep='\t', index=False)

X.to_csv(snakemake.output.feature_matrix_all, sep='\t', index=False)
y.to_csv(snakemake.output.label_vector_all, sep='\t', index=False)

print('Feature matrices and label vectors are made.')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
