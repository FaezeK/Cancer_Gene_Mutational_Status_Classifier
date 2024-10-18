rule all:
    input:
        'results/RF_better_clssfctn.jpg'


rule preprocess_data:
    input:
        pog_tpm = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/hg38/pog_tpm.tsv',
        pog_snv = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/hg38/new_download/pog_snps_indels_short.tsv',
        pog_germline = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/germline_mut/POG500.germline_variants.final.xlsx',
        pog_cnv = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/hg38/cnv/all_CN.tsv',
        pog_sv = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/pog_sv.tsv',

        tcga_tpm = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/hg38/tcga_rsem_gene_tpm.txt',
        tcga_snv = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/hg38/tcga_gdc_hg38_mut_fltr.txt',
        tcga_germline = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/germline_mut/PCA_pathVar_integrated_filtered_adjusted.tsv',
        tcga_cnv = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/cnv/TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes',
        tcga_sv = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/sv/all_sv.txt',
        all_genes = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/all_genes.txt'
    output:
        pog_tpm_prcssd = 'tmp_data/pog_tpm_prcssd.tsv',
        pog_mut_prcssd = 'tmp_data/pog_mut_prcssd.tsv',
        pog_cnv_prcssd = 'tmp_data/pog_cnv_prcssd.tsv',
        pog_sv_prcssd = 'tmp_data/pog_sv_prcssd.tsv',

        tcga_tpm_prcssd = 'tmp_data/tcga_tpm_prcssd.tsv',
        tcga_mut_prcssd = 'tmp_data/tcga_mut_prcssd.tsv',
        tcga_cnv_prcssd = 'tmp_data/tcga_cnv_prcssd.tsv',
        tcga_sv_prcssd = 'tmp_data/tcga_sv_prcssd.tsv'
    message: 'Preprocessing datasets!'
    script: 'process_input.py'

rule make_feature_matrix_label_vector:
    input:
        pog_tpm_prcssd = 'tmp_data/pog_tpm_prcssd.tsv',
        tcga_tpm_prcssd = 'tmp_data/tcga_tpm_prcssd.tsv',
        pog_mut_prcssd = 'tmp_data/pog_mut_prcssd.tsv',
        tcga_mut_prcssd = 'tmp_data/tcga_mut_prcssd.tsv'
    output:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt'
    message: 'Making feature matrix and label vector for analysis!'
    script: 'make_feature_mat_label_vec.py'

rule fine_tune:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt'
    output:
        best_hyper_param = 'results/best_hyper_param.txt'
    message: 'Fine-tuning hyperparameters for classification!'
    script: 'fine_tune.py'
