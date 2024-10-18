##############################################################################################################
## This is a script to make feature matrix and label vector based on preprocessed expression and SNV/INDEL
## data (samples from TCGA and POG are combined)
##############################################################################################################

# import required libraries
import pandas as pd
import timeit
import sys

# variables provided at run-time
gene_of_interest = sys.argv[1]

# timing the run-time
start_time = timeit.default_timer()

#########################################
########## Read processed data ##########
#########################################
print('Reading preprocessed data ...')
print('')
# expression data
pog_tpm = pd.read_csv(snakemake.input.pog_tpm_prcssd, delimiter = '\t', header=0)
tcga_tpm = pd.read_csv(snakemake.input.tcga_tpm_prcssd, delimiter = '\t', header=0)

# somatic mutation data
pog_mut = pd.read_csv(snakemake.input.pog_mut_prcssd, delimiter='\t', header=0)
tcga_mut = pd.read_csv(snakemake.input.tcga_mut_prcssd, delimiter='\t', header=0)

print('Input data are read!')
print('')

##############################################################################
######### Making Feature Matrix and Label Vector for Classification ##########
##############################################################################
print('Making X and y arrays ...')
print('')
# function to find samples with mutated and wild-type copies of a given gene of interest
def find_mut_groups(expr_dat, snv_dat, gene_of_interest):

    snv_dat_fltr = snv_dat[snv_dat.gene_name == gene_of_interest]

    snv_dat_fltr = snv_dat_fltr.rename(columns={'consequence':'consequence_pre'})
    if snv_dat.p_id[0].startswith('POG'):
        snv_dat_fltr['consequence'] = snv_dat_fltr.consequence_pre.str.split('\+')
    else:
        snv_dat_fltr['consequence'] = snv_dat_fltr.consequence_pre.str.split('&')
    snv_dat_fltr_exp = snv_dat_fltr.explode('consequence')

    snv_impactful_mut = snv_dat_fltr_exp[snv_dat_fltr_exp.consequence.isin(['missense_variant', 'frameshift_variant', 
                                'splice_acceptor_variant', 'stop_gained', 'inframe_deletion', 'splice_donor_variant', 'inframe_insertion', 
                                'start_lost', 'stop_lost', 'conservative_inframe_deletion', 'conservative_inframe_insertion', 
                                'disruptive_inframe_deletion', 'disruptive_inframe_insertion', 'missense', 'deletion', 'indel',
                                'nonsense', 'splicing', 'insertion/deletion', 'in-frame indel'])]
    snv_not_impactful_mut = snv_dat_fltr_exp[snv_dat_fltr_exp.consequence.isin(['synonymous_variant', 'splice_region_variant', 
                                '3_prime_UTR_variant', 'intron_variant', '5_prime_UTR_variant', 'non_coding_transcript_exon_variant', 
                                'downstream_gene_variant', 'upstream_gene_variant', 'stop_retained_variant', 'intergenic_region', 
                                'non_coding_transcript_variant', 'splice_donor_region_variant', 'synonymous'])]

    snv_impactful_mut_samples = snv_impactful_mut.p_id.unique()
    snv_not_impactful_mut_samples = snv_not_impactful_mut.p_id.unique()
    snv_not_impactful_mut_samples = snv_not_impactful_mut_samples[np.isin(snv_not_impactful_mut_samples, snv_impactful_mut_samples)==False]

    all_snv_samples = snv_dat.p_id.unique()
    snv_wt_samples = all_snv_samples[np.isin(all_snv_samples, snv_impactful_mut_samples)==False]
    snv_wt_samples = snv_wt_samples[np.isin(snv_wt_samples, snv_not_impactful_mut_samples)==False]

    # divide tcga tpm dataset by above groups
    tpm_impactful_mut = expr_dat[expr_dat.index.isin(snv_impactful_mut_samples)]
    tpm_not_impactful_mut = expr_dat[expr_dat.index.isin(snv_not_impactful_mut_samples)]
    tpm_wt = expr_dat[expr_dat.index.isin(snv_wt_samples)]

    return tpm_impactful_mut, tpm_not_impactful_mut, tpm_wt

# transpose expression matrices
tcga_tpm_T = tcga_tpm.T
pog_tpm_T = pog_tpm.T

# sort expression data by sample id
tcga_tpm_T = tcga_tpm_T.sort_index(axis=1)
pog_tpm_T = pog_tpm_T.sort_index(axis=1)

# divide samples based on the mutational status of the given gene of interest
tcga_tpm_impactful_mut, tcga_tpm_not_impactful_mut, tcga_tpm_wt = find_mut_groups(tcga_tpm_T, tcga_all_mut, gene_of_interest)
pog_tpm_impactful_mut, pog_tpm_not_impactful_mut, pog_tpm_wt = find_mut_groups(pog_tpm_T, pog_all_mut, gene_of_interest)

# make_X_y_merged function creates the feature matrix and label vector using both TCGA and POG data
def make_X_y_merged(exp_mut_df1, exp_mut_df2, exp_wt_df1, exp_wt_df2, mut_label, wt_label):

    X = pd.concat([exp_mut_df1, exp_mut_df2, exp_wt_df1, exp_wt_df2])
    y = pd.concat([pd.Series([mut_label]*exp_mut_df1.shape[0]), pd.Series([mut_label]*exp_mut_df2.shape[0]),
                   pd.Series([wt_label]*exp_wt_df1.shape[0]), pd.Series([wt_label]*exp_wt_df2.shape[0])])
    
    return X, y

# make expression matrix and target label for TCGA and POG samples based on SNV data only
X, y = make_X_y_merged(tcga_tpm_impactful_mut, pog_tpm_impactful_mut, tcga_tpm_wt, pog_tpm_wt, 'mut', 'wt')

####################################################################
##### Write feature matrix and label vector into tmp directory #####
####################################################################

X.to_csv(snakemake.output.feature_matrix, sep='\t', index=False)
y.to_csv(snakemake.output.label_vector, sep='\t', index=False)

print('Files were written into tmp directory!')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
