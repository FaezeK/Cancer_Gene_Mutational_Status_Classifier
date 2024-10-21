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
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        feature_matrix_sv = 'tmp_data/feature_matrix_sv.txt',
        label_vector_sv = 'tmp_data/label_vector_sv.txt',
        feature_matrix_all = 'tmp_data/feature_matrix_all.txt',
        label_vector_all = 'tmp_data/label_vector_all.txt'
    message: 'Utilizing CNV and SV data when contructing feature matrix and label vector'
    script: 'make_feature_mat_label_vec_w_all_data_types.py'

rule test_data_type_combinations:
    input:
        feature_matrix  = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        feature_matrix_sv = 'tmp_data/feature_matrix_sv.txt',
        label_vector_sv = 'tmp_data/label_vector_sv.txt',
        feature_matrix_all = 'tmp_data/feature_matrix_all.txt',
        label_vector_all = 'tmp_data/label_vector_all.txt',
        best_hyper_param = 'results/best_hyper_param.txt'
    output:
        data_types_combinations_results = 'results/data_types_combinations_results.txt'
    message: 'Comparing the model performance on different data combinations'
    script: 'test_data_type_combinations.py'

rule analyze_performance_in_chosen_setting:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/best_hyper_param.txt',
        best_setting = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/best_classification_setting.tsv',
        tcga_tpm_not_impactful_mut = 'tmp_data/tcga_tpm_not_impactful_mut.txt',
        pog_tpm_not_impactful_mut = 'tmp_data/pog_tpm_not_impactful_mut.txt',
        tcga_mut_prcssd = 'tmp_data/tcga_mut_prcssd.tsv',
        pog_mut_prcssd = 'tmp_data/pog_mut_prcssd.tsv'
    output:
        gene_importance_scores_from_RF = 'results/gene_importance_scores_from_RF.txt',
        pred_on_sample_w_not_impact_mut = 'results/pred_on_sample_w_not_impact_mut.txt',
        not_impact_mut_groups_pred_n_binom_p_val = 'results/not_impact_mut_groups_pred_n_binom_p_val.txt',
        not_impact_conseq_base_n_aa_changes = 'results/not_impact_conseq_base_n_aa_changes.txt'
    message: 'Test RF performance using best setting'
    script: 'analyze_performance_in_chosen_setting.py'

rule find_threshold_for_important_genes:
    input:
        gene_importance_scores_from_RF = 'results/gene_importance_scores_from_RF.txt',
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/best_hyper_param.txt',
        best_setting = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/best_classification_setting.tsv'
    output:
        num_important_genes = 'results/num_important_genes.txt',
        true_vs_shuffled_importance_scores = 'results/true_vs_shuffled_importance_scores.jpg',
        true_vs_shuffled_importance_scores_zoomed_in = 'results/true_vs_shuffled_importance_scores_zoomed_in.jpg',
        true_vs_shuffled_importance_scores_zoomed_in2 = 'results/true_vs_shuffled_importance_scores_zoomed_in2.jpg'
    message: 'Find threshold for genes contributing the most to the classification'
    script: 'find_threshold.py'

rule classify_samples_by_tumour_types:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/best_hyper_param.txt',
        best_setting = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/best_classification_setting.tsv',
        tcga_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/hg38/pog_t_type.tsv'
    output:
        t_type_results = 'results/t_type_results.txt',
        t_type_gene_importance_scores = 'results/t_type_gene_importance_scores.txt',
        t_type_results_balanced = 'results/t_type_results_balanced.txt',
        t_type_gene_importance_scores_balanced = 'results/t_type_gene_importance_scores_balanced.txt'
    message: 'Run classification on each tumour type separately (for both balanced and imbalanced sets)'
    script: 'classify_by_each_t_type.py'

rule test_performance_on_tumour_types:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/best_hyper_param.txt',
        best_setting = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/best_classification_setting.tsv',
        tcga_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/hg38/pog_t_type.tsv',
        tcga_tpm_not_impactful_mut = 'tmp_data/tcga_tpm_not_impactful_mut.txt',
        pog_tpm_not_impactful_mut = 'tmp_data/pog_tpm_not_impactful_mut.txt',
        tcga_mut_prcssd = 'tmp_data/tcga_mut_prcssd.tsv',
        pog_mut_prcssd = 'tmp_data/pog_mut_prcssd.tsv'
    output:
        specific_t_types_cv_results = 'results/specific_t_types_cv_results.txt',
        gene_importance_scores_specific_tumour_types = 'results/gene_importance_scores_specific_tumour_types.txt',
        pred_on_sample_w_not_impact_mut_specific_tumour_types = 'results/pred_on_sample_w_not_impact_mut_specific_tumour_types.txt',
        not_impact_mut_groups_pred_n_binom_p_val = 'results/not_impact_mut_groups_pred_n_binom_p_val.txt',
        not_impact_base_n_aa_changes = 'results/not_impact_base_n_aa_changes.txt'
    message: 'Run classification on selected tumour types from the imbalanced analysis'
    script: 'performance_on_specific_tumour_types.py'

rule assess_performance_on_balanced_set_of_all_tumours:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/best_hyper_param.txt',
        best_setting = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/best_classification_setting.tsv',
        tcga_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/hg38/pog_t_type.tsv'
    output:
        permut_balanced_results_all_tumours = 'results/permut_balanced_results_all_tumours.txt'
    message: 'Run classification on balanced set of tumours in permutations!'
    script: 'assess_performance_on_balanced_set_of_all_tumours.py'
    
rule assess_performance_on_balanced_set_of_selected_tumours:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/best_hyper_param.txt',
        best_setting = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/best_classification_setting.tsv',
        tcga_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/hg38/pog_t_type.tsv'
    output:
        permut_balanced_results_selected_tumours = 'results/permut_balanced_results_selected_tumours.txt'
    message: 'Run classification on balanced set of tumours in permutations!'
    script: 'assess_performance_on_balanced_set_of_all_tumours.py'

rule test_performance_on_balanced_tumour_types:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/best_hyper_param.txt',
        best_setting = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/best_classification_setting.tsv',
        tcga_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/projects/fkeshavarz_prj/fkeshavarz_scratch/data/pog/hg38/pog_t_type.tsv'
    output:
        balanced_t_types_results = 'results/balanced_t_types_results.txt'
    message: 'Run classification on selected tumour types from the balanced analysis'
    script: 'performance_on_balanced_tumour_types.py'
