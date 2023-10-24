##############################################################################################################
## This is a script to process initial data files for the classification task (based on gene mutations, 
## copy number variations, and structural variations).
##############################################################################################################

# import required libraries
import pandas as pd
import numpy as np
import timeit
import sys

# variables provided at run-time
gene_of_interest = sys.argv[1]

# timing the run-time
start_time = timeit.default_timer()

#####################################
########## Read input data ##########
#####################################
print('Reading input data ...')
print('')
# expression data
pog_tpm = pd.read_csv(snakemake.input.pog_tpm, delimiter = '\t', header=0)
tcga_tpm = pd.read_csv(snakemake.input.tcga_tpm, delimiter = '\t', header=0)
hartwig_tpm = pd.read_csv(snakemake.input.hartwig_tpm, delimiter = '\t', header=0)

# somatic mutation data
pog_snv = pd.read_csv(snakemake.input.pog_snv, delimiter='\t', header=0)
tcga_snv = pd.read_csv(snakemake.input.tcga_snv, delimiter='\t', header=0)
hartwig_snv = pd.read_csv(snakemake.input.hartwig_snv, delimiter='\t', header=0, dtype = {'CHROM':str})

# germline mutation data
tcga_germline = pd.read_csv(snakemake.input.tcga_germline, delimiter='\t', header=0)
pog_germline = pd.read_excel(snakemake.input.pog_germline)

# copy number variation data
tcga_cnv = pd.read_csv(snakemake.input.tcga_cnv, delimiter = '\t', header=0)

# structural variation data
tcga_sv = pd.read_csv(snakemake.input.tcga_sv, delimiter='\t', header=0)

# metadata
tcga_t_type = pd.read_csv(snakemake.input.tcga_t_type, delimiter='\t', header=0)
all_genes = pd.read_csv(snakemake.input.all_genes, delimiter='\t', header=0)
print('Input data are read!')
print('')

################################################
######### Process Expression Datasets ##########
################################################
print('Processing input files ...')
print('')

# the function to change ensembl ids to a combination of    
# gene names and ensembl ids in expression matrices
def assigne_ensembl_id_as_index(expr_df):
    expr_df['ensembl_ids'] = expr_df.gene_id.str.split('.').str[0]
    expr_df = expr_df.merge(all_genes, on='ensembl_ids')
    expr_df = expr_df.drop(columns=['gene_id','ensembl_ids','gene_names'])
    expr_df = expr_df.set_index('genes')
    return expr_df

# harmonize gene names in all three expression matrices
# change the sample column name to gene_id to match with the other two datasets
tcga_tpm = tcga_tpm.rename(columns={"sample":"gene_id"})

pog_tpm = assigne_ensembl_id_as_index(pog_tpm)
hartwig_tpm = assigne_ensembl_id_as_index(hartwig_tpm)
tcga_tpm = assigne_ensembl_id_as_index(tcga_tpm)

# remove the genes that do not exist in TCGA from POG and Hartwig expression data
pog_tpm = pog_tpm[pog_tpm.index.isin(tcga_tpm.index)]
hartwig_tpm = hartwig_tpm[hartwig_tpm.index.isin(tcga_tpm.index)]

# exclue non-primary tumours from TCGA (now there are two sets of TCGA
# samples, one with primary tumours only and the second one with all 
# TCGA samples including the ones from metastatic and recurrenct tumours)
tcga_tpm_primary_only = tcga_tpm.loc[:, tcga_tpm.columns.str.slice(13,15)=='01']

# convert tcga values to tpm from log2(tpm+0.001)
tcga_tpm = (2**tcga_tpm) - 0.001
tcga_tpm_primary_only = (2**tcga_tpm_primary_only) - 0.001

#################################################
######### Process SNVs/INDELS Datasets ##########
#################################################

# for samples with multiple mutations in the same gene, get the most important one
# order of importance/harmfulness: 
# frameshift_variant > stop_gained > start_lost > stop_lost > splice_acceptor_variant > splice_donor_variant > missense_variant > 
# inframe_deletion > inframe_insertion > protein_altering_variant > splice_region_variant > incomplete_terminal_codon_variant > 
# stop_retained_variant > synonymous_variant > 3_prime_UTR_variant > 5_prime_UTR_variant > regulatory_region_variant > coding_sequence_variant 
# > intron_variant > upstream_gene_variant > downstream_gene_variant > non_coding_transcript_exon_variant > mature_miRNA_variant
tcga_snv['conseq_score'] = 0
tcga_snv.loc[tcga_snv.One_Consequence=='frameshift_variant', 'conseq_score'] = 1
tcga_snv.loc[tcga_snv.One_Consequence=='stop_gained', 'conseq_score'] = 2
tcga_snv.loc[tcga_snv.One_Consequence=='start_lost', 'conseq_score'] = 3
tcga_snv.loc[tcga_snv.One_Consequence=='stop_lost', 'conseq_score'] = 4
tcga_snv.loc[tcga_snv.One_Consequence=='splice_acceptor_variant', 'conseq_score'] = 5
tcga_snv.loc[tcga_snv.One_Consequence=='splice_donor_variant', 'conseq_score'] = 6
tcga_snv.loc[tcga_snv.One_Consequence=='missense_variant', 'conseq_score'] = 7
tcga_snv.loc[tcga_snv.One_Consequence=='inframe_deletion', 'conseq_score'] = 8
tcga_snv.loc[tcga_snv.One_Consequence=='inframe_insertion', 'conseq_score'] = 9
tcga_snv.loc[tcga_snv.One_Consequence=='protein_altering_variant', 'conseq_score'] = 10
tcga_snv.loc[tcga_snv.One_Consequence=='splice_region_variant', 'conseq_score'] = 11
tcga_snv.loc[tcga_snv.One_Consequence=='incomplete_terminal_codon_variant', 'conseq_score'] = 12
tcga_snv.loc[tcga_snv.One_Consequence=='stop_retained_variant', 'conseq_score'] = 13
tcga_snv.loc[tcga_snv.One_Consequence=='synonymous_variant', 'conseq_score'] = 14
tcga_snv.loc[tcga_snv.One_Consequence=='3_prime_UTR_variant', 'conseq_score'] = 15
tcga_snv.loc[tcga_snv.One_Consequence=='5_prime_UTR_variant', 'conseq_score'] = 16
tcga_snv.loc[tcga_snv.One_Consequence=='regulatory_region_variant', 'conseq_score'] = 17
tcga_snv.loc[tcga_snv.One_Consequence=='coding_sequence_variant', 'conseq_score'] = 18
tcga_snv.loc[tcga_snv.One_Consequence=='intron_variant', 'conseq_score'] = 19
tcga_snv.loc[tcga_snv.One_Consequence=='upstream_gene_variant', 'conseq_score'] = 20
tcga_snv.loc[tcga_snv.One_Consequence=='downstream_gene_variant', 'conseq_score'] = 21
tcga_snv.loc[tcga_snv.One_Consequence=='non_coding_transcript_exon_variant', 'conseq_score'] = 22
tcga_snv.loc[tcga_snv.One_Consequence=='mature_miRNA_variant', 'conseq_score'] = 23

tcga_snv = tcga_snv.sort_values(by=['Tumor_Sample_Barcode','Hugo_Symbol','conseq_score'])

# for the same sample, same gene, keep the first row
tcga_snv = tcga_snv.drop_duplicates(subset=['Tumor_Sample_Barcode', 'Hugo_Symbol'], keep='first')

# change/create the patient id column to match across all three mutation datasets
pog_snv = pog_snv.rename(columns={"patient.participant_project_identifier":"p_id"})
tcga_snv['p_id'] = tcga_snv.Tumor_Sample_Barcode.str.slice(0,15)
hartwig_snv['p_id'] = hartwig_snv.P_id.str.slice(0,12)

# convert gene name column in TCGA mutation dataframe to match POG 
tcga_snv = tcga_snv.rename(columns={'Hugo_Symbol':'gene_name'})

# remove the samples from the same patient that has the same gene mutation
tcga_snv = tcga_snv.drop_duplicates(subset=['p_id', 'gene_name', 'One_Consequence'], keep='first')

# find samples that have expression, mutation and CNV data
def find_smpls_w_expr_n_mut_n_cnv_data(expr_df, mut_df, cnv_df):
    common_ids = list(set(mut_df.p_id) & set(cnv_df.columns) & set(expr_df.columns))
    
    mut_df = mut_df[mut_df.p_id.isin(common_ids)]
    cnv_df = cnv_df.set_index('Sample')
    cnv_df = cnv_df.loc[:,cnv_df.columns.isin(common_ids)]
    expr_df = expr_df.loc[:,expr_df.columns.isin(common_ids)]

    return expr_df, mut_df, cnv_df

#pog_tpm, pog_snv = find_smpls_w_expr_n_mut_n_cnv_data(pog_tpm, pog_snv)
tcga_tpm, tcga_snv, cnv_df = find_smpls_w_expr_n_mut_n_cnv_data(tcga_tpm, tcga_snv)
tcga_tpm_primary_only, tcga_snv, cnv_df = find_smpls_w_expr_n_mut_n_cnv_data(tcga_tpm_primary_only, tcga_snv)
#hartwig_tpm, hartwig_snv = find_smpls_w_expr_n_mut_n_cnv_data(hartwig_tpm, hartwig_snv)

# harmonize somatic mutation data for tcga and pog
pog_snv = pog_snv[['p_id','gene_name','effect']]
tcga_snv = tcga_snv[['p_id','gene_name','One_Consequence']]

pog_snv = pog_snv.rename(columns={'effect':'consequence'})
tcga_snv = tcga_snv.rename(columns={'One_Consequence':'consequence'})

#######################################################
######### Process Germline Mutation Datasets ##########
#######################################################

# extract required columns from germline mutation data
tcga_germline = tcga_germline[['Sample','HUGO_Symbol','Consequence','Overall_Classification']]
pog_germline = pog_germline[['pog_id','gene_name','function','reviewed_classification']]

# harmonize column names in germline mutation data
tcga_germline = tcga_germline.rename(columns={'Sample':'p_id','HUGO_Symbol':'gene_name','Consequence':'consequence','Overall_Classification':'classification'})
pog_germline = pog_germline.rename(columns={'pog_id':'p_id','function':'consequence','reviewed_classification':'classification'})

# filter germline data for pathogenic or likely pathogenic mutations and add to somatic mut data
def filter_and_add(germline_mut_df, somatic_mut_df):
    germline_mut_df = germline_mut_df[(germline_mut_df.classification == 'Pathogenic') | 
                                      (germline_mut_df.classification == 'Likely Pathogenic') | 
                                      (germline_mut_df.classification == 'Likely pathogenic')]
    germline_mut_df = germline_mut_df.drop(columns=['classification'])
    somatic_n_germline = pd.concat([somatic_mut_df, germline_mut_df])
    return somatic_n_germline

tcga_all_mut = filter_and_add(tcga_germline, tcga_snv)
pog_all_mut = filter_and_add(pog_germline, pog_snv)

tcga_all_mut = tcga_all_mut.reset_index(drop=True)
pog_all_mut = pog_all_mut.reset_index(drop=True)

##########################################################
######### Process Structural Variation Datasets ##########
##########################################################

# filter SV data for needed columns and samples with SVs in the gene of interest
tcga_sv_fltrd = tcga_sv[['Sample_Id','Site1_Hugo_Symbol','Site2_Hugo_Symbol','Site2_Effect_On_Frame','Event_Info']]
tcga_sv_gene_of_interest = tcga_sv_fltrd[(tcga_sv_fltrd.Site1_Hugo_Symbol==gene_of_interest) | 
                                         (tcga_sv_fltrd.Site2_Hugo_Symbol==gene_of_interest)]

print('Data has been processed . . .')
print('')

####################################################
##### Write processed files into tmp directory #####
####################################################

tcga_tpm.to_csv(snakemake.output.tcga_expr_prcssd, sep='\t', index=False)

print('Processed files were written into tmp directory!')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
