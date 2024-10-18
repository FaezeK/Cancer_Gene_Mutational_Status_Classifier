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

# somatic mutation data
pog_snv = pd.read_csv(snakemake.input.pog_snv, delimiter='\t', header=0)
tcga_snv = pd.read_csv(snakemake.input.tcga_snv, delimiter='\t', header=0)

# germline mutation data
tcga_germline = pd.read_csv(snakemake.input.tcga_germline, delimiter='\t', header=0)
pog_germline = pd.read_excel(snakemake.input.pog_germline)

# copy number variation data
tcga_cnv = pd.read_csv(snakemake.input.tcga_cnv, delimiter = '\t', header=0)
pog_cnv = pd.read_csv(snakemake.input.pog_cnv, delimiter = '\t', header=0)

# structural variation data
tcga_sv = pd.read_csv(snakemake.input.tcga_sv, delimiter='\t', header=0)
pog_sv = pd.read_csv(snakemake.input.pog_sv, delimiter='\t', header=0)

# metadata
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

# harmonize gene names in expression matrices
# change the sample column name to gene_id to match with the POG dataset
tcga_tpm = tcga_tpm.rename(columns={"sample":"gene_id"})

pog_tpm = assigne_ensembl_id_as_index(pog_tpm)
tcga_tpm = assigne_ensembl_id_as_index(tcga_tpm)

# remove the genes that do not exist in TCGA from POG expression data
pog_tpm = pog_tpm[pog_tpm.index.isin(tcga_tpm.index)]

# exclue non-primary tumours from TCGA (now there are two sets of TCGA
# samples, one with primary tumours only and the second one with all 
# TCGA samples including the ones from metastatic and recurrenct tumours)
tcga_tpm_primary_only = tcga_tpm.loc[:, tcga_tpm.columns.str.slice(13,15)=='01']

# convert tcga values to tpm from log2(tpm+0.001)
tcga_tpm = (2**tcga_tpm) - 0.001
tcga_tpm_primary_only = (2**tcga_tpm_primary_only) - 0.001

#########################################
######### Process CNV Datasets ##########
#########################################

# change the cnv data format of POG to match TCGA
pog_cnv = pog_cnv.set_index('p_id')
pog_cnv_T = pog_cnv.T
pog_cnv_T.insert(0, 'gene_id',pog_cnv_T.index)
pog_cnv_T = pog_cnv_T.reset_index(drop=True)

# add gene names to pog cnv
pog_cnv = assigne_ensembl_id_as_index(pog_cnv_T)
pog_cnv['gene_name'] = pog_cnv.index.str.split('_').str[0]
pog_cnv = pog_cnv.set_index('gene_name')

# assign genes as index for tcga cnv
tcga_cnv = tcga_cnv.set_index('Sample')

#################################################
######### Process SNVs/INDELs Datasets ##########
#################################################

# change/create the patient id column to match across all three mutation datasets
pog_snv = pog_snv.rename(columns={"patient.participant_project_identifier":"p_id"})
tcga_snv['p_id'] = tcga_snv.Tumor_Sample_Barcode.str.slice(0,15)

# harmonize the column names for SNV datasets
tcga_snv = tcga_snv.rename(columns={'Hugo_Symbol':'gene_name', 'One_Consequence':'consequence', 'HGVSc':'base_change', 
                                    'HGVSp':'amino_acid_change'})

pog_snv = pog_snv.rename(columns={'patient.participant_project_identifier':'p_id', 'effect':'consequence', 
                                  'amino_acid_c':'base_change', 'amino_acid_p':'amino_acid_change'})

# remove the samples that for the same patient has the same gene mutation
tcga_snv = tcga_snv.drop_duplicates(subset=['p_id', 'gene_name', 'consequence'], keep='first')
pog_snv = pog_snv.drop_duplicates(subset=['consequence','impact','gene_name','p_id'])

# find samples that have expression, mutation and CNV data
def find_smpls_w_expr_n_mut_n_cnv_data(expr_df, snv_df, cnv_df):
    common_ids = list(set(snv_df.p_id) & set(expr_df.columns) & set(cnv_df.columns))
    
    expr_df = expr_df.loc[:,expr_df.columns.isin(common_ids)]
    snv_df = snv_df[snv_df.p_id.isin(common_ids)]
    cnv_df = cnv_df.loc[:,cnv_df.columns.isin(common_ids)]

    return expr_df, snv_df, cnv_df

pog_tpm, pog_snv, pog_cnv = find_smpls_w_expr_n_mut_n_cnv_data(pog_tpm, pog_snv, pog_cnv)
tcga_tpm, tcga_snv, tcga_cnv = find_smpls_w_expr_n_mut_n_cnv_data(tcga_tpm, tcga_snv, tcga_cnv)

# select required columns from somatic mutation data for tcga and pog 
pog_snv = pog_snv[['p_id','gene_name','consequence','base_change','amino_acid_change']]
tcga_snv = tcga_snv[['p_id','gene_name','consequence','base_change','amino_acid_change']]

#######################################################
######### Process Germline Mutation Datasets ##########
#######################################################

# extract required columns from germline mutation data
tcga_germline = tcga_germline[['Sample','HUGO_Symbol','Consequence','Overall_Classification','HGVSc','HGVSp']]
pog_germline = pog_germline[['pog_id','gene_name','function','reviewed_classification','refseq_aa_change']]

# extract base change and aa change from refseq_aa_change column in pog data
pog_germline['base_change'] = pog_germline.refseq_aa_change.str.split(':').str[3]
pog_germline['amino_acid_change'] = pog_germline.refseq_aa_change.str.split(':').str[4]

# harmonize ids between tcga somatic and germline mut data
tcga_germline['p_id'] = tcga_germline.Sample.str.slice(0,15)
tcga_germline = tcga_germline.drop(columns=['Sample'])

# rename columns in germline mutation data
tcga_germline = tcga_germline.rename(columns={'HUGO_Symbol':'gene_name','Consequence':'consequence','Overall_Classification':'classification'
                                              , 'HGVSc':'base_change', 'HGVSp':'amino_acid_change'})
pog_germline = pog_germline.rename(columns={'pog_id':'p_id','function':'consequence','reviewed_classification':'classification'})
tcga_germline = tcga_germline.drop_duplicates()
pog_germline = pog_germline.drop_duplicates()

# extract pathogenic and likely pathogenic germline mutation data and add them to SNV data
# (to make sure all samples with disruptive mutations are labeled as mutant irrespective of the source)
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

# extract required columns
tcga_sv = tcga_sv[['Sample_Id','Site1_Hugo_Symbol','Site2_Hugo_Symbol']]
pog_sv = pog_sv[['Patient Participant_project_identifier','Gene1_aliases','Gene2_aliases']]
pog_sv = pog_sv.drop_duplicates()

# harmonize the column names
tcga_sv = tcga_sv.rename(columns={'Sample_Id':'p_id', 'Site1_Hugo_Symbol':'gene1', 'Site2_Hugo_Symbol':'gene2'})
pog_sv = pog_sv.rename(columns={'Patient Participant_project_identifier':'p_id','Gene1_aliases':'gene1', 'Gene2_aliases':'gene2'})

print('Data has been processed . . .')
print('')

####################################################
##### Write processed files into tmp directory #####
####################################################

pog_tpm.to_csv(snakemake.output.pog_tpm_prcssd, sep='\t', index=False)
pog_cnv.to_csv(snakemake.output.pog_cnv_prcssd, sep='\t', index=False)
pog_all_mut.to_csv(snakemake.output.pog_mut_prcssd, sep='\t', index=False)
pog_sv.to_csv(snakemake.output.pog_sv_prcssd, sep='\t', index=False)

tcga_tpm.to_csv(snakemake.output.tcga_tpm_prcssd, sep='\t', index=False)
tcga_cnv.to_csv(snakemake.output.tcga_cnv_prcssd, sep='\t', index=False)
tcga_all_mut.to_csv(snakemake.output.tcga_mut_prcssd, sep='\t', index=False)
tcga_sv.to_csv(snakemake.output.tcga_sv_prcssd, sep='\t', index=False)

print('Processed files were written into tmp directory!')
print('')

end_time = timeit.default_timer()
tot_time = (end_time - start_time) / 60
tot_time_h = (end_time - start_time) / 3600
print('')
print('Script ran in ' + str(tot_time) + ' minutes = ' + str(tot_time_h) + ' hours')
print('')
