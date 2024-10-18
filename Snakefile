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
        label_vector = 'tmp_data/label_vector.txt',
        tcga_tpm_impactful_mut = 'tmp_data/tcga_tpm_impactful_mut.txt',
        tcga_tpm_wt = 'tmp_data/tcga_tpm_wt.txt',
        tcga_tpm_not_impactful_mut = 'tmp_data/tcga_tpm_not_impactful_mut.txt',
        pog_tpm_impactful_mut = 'tmp_data/pog_tpm_impactful_mut.txt',
        pog_tpm_wt = 'tmp_data/pog_tpm_wt.txt',
        pog_tpm_not_impactful_mut = 'tmp_data/pog_tpm_not_impactful_mut.txt'
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

rule test_performance_SNVs_only:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        best_hyper_param = 'results/best_hyper_param.txt',
        tcga_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/tcga_t_type.tsv'
    output:
        classification_results_SNVs_only = 'results/classification_results_SNVs_only.txt'
        auroc_auprc_SNVs_only = 'results/auroc_auprc_SNVs_only.jpg',
        f1_to_min_maj_ratio_plot_SNVs_only = 'results/f1_to_min_maj_ratio_plot_SNVs_only.jpg'
    message: 'Test RF performance using SNVs/INDELs data only'
    script: 'test_performance_SNVs_only.py'

rule make_feature_matrix_label_vector_w_additional_data:
    input:
        tcga_tpm_impactful_mut = 'tmp_data/tcga_tpm_impactful_mut.txt',
        tcga_tpm_wt = 'tmp_data/tcga_tpm_wt.txt',
        tcga_tpm_not_impactful_mut = 'tmp_data/tcga_tpm_not_impactful_mut.txt',
        tcga_cnv_prcssd = 'tmp_data/tcga_cnv_prcssd.tsv',
        tcga_sv_prcssd = 'tmp_data/tcga_sv_prcssd.tsv',
        
        pog_tpm_impactful_mut = 'tmp_data/pog_tpm_impactful_mut.txt',
        pog_tpm_wt = 'tmp_data/pog_tpm_wt.txt',
        pog_tpm_not_impactful_mut = 'tmp_data/pog_tpm_not_impactful_mut.txt',
        pog_cnv_prcssd = 'tmp_data/pog_cnv_prcssd.tsv',
        pog_sv_prcssd = 'tmp_data/pog_sv_prcssd.tsv'
    output:
        feature_matrix_updated = 'tmp_data/feature_matrix_updated.txt',
        label_vector_updated = 'tmp_data/label_vector_updated.txt'
    message: 'Utilizing CNV and SV data when contructing feature matrix and label vector'
    script: 'make_feature_mat_label_vec_w_all_data_types.py'
