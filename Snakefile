genes_list = ['APC','AR','ARID1A','ASXL1','ATM','ATR','ATRX','BRAF','BRCA1','BRCA2','CDH1','CDK12','CDKN2A','CTCF',
              'CTNNB1','EGFR','EP300','ERBB4','EZH2','FBXW7','FLT3','GATA3','KDM6A','KEAP1','KIT','KRAS','MAP3K1',
              'MECOM','MTOR','NCOR1','NF1','NFE2L2','NOTCH1','NRAS','NSD1','PBRM1','PDGFRA','PIK3CA','PIK3R1',
              'POLQ','PTEN','RB1','SETBP1','SETD2','SF3B1','SMAD4','SPOP','STAG2','TET2','TP53']

for g in genes_list:
    gene_of_interest = g

def fetch_tumour_types():
    # Get output of classify_by_tumour_types rule
    checkpoint_output = checkpoints.classify_samples_by_tumour_types.get()
    output_paths = checkpoint_output.output 
    # Extract tumour types dynamically based on file names in output paths
    tumour_types = set(path.split("_")[-1].replace(".txt", "") for path in output_paths)
    # Remove the file endings that are not tumour types
    tumour_types -= set(['scores','results','balanced'])
    # Exapand file name using found tumour types
    return expand(
        f"results/{gene_of_interest}/individual_t_types/t_type_results_{{tumour_type}}.txt",
        tumour_type=tumour_types
    )

rule all:
    input:
        'results/'+str(gene_of_interest)+'/classification_results_SNVs_only.txt',
        'results/'+str(gene_of_interest)+'/data_types_combinations_results.txt',
        'results/'+str(gene_of_interest)+'/gene_importance_scores_from_RF.txt',
        'results/'+str(gene_of_interest)+'/true_vs_shuffled_importance_scores.jpg',
        lambda wildcards: fetch_tumour_types(),
        'results/'+str(gene_of_interest)+'/specific_t_types_cv_results.txt',
        'results/'+str(gene_of_interest)+'/permut_balanced_results_all_tumours.txt',
        'results/'+str(gene_of_interest)+'/permut_balanced_results_selected_tumours.txt',
        'results/'+str(gene_of_interest)+'/balanced_t_types_cv_results.txt'
    shell: 'rm -rf tmp_data'


rule preprocess_data:
    input:
        pog_tpm = '/data/pog/hg38/pog_tpm.tsv',
        pog_snv = '/data/pog/hg38/new_download/pog_snps_indels_short.tsv',
        pog_germline = '/data/pog/germline_mut/POG500.germline_variants.final.xlsx',
        pog_cnv = '/data/pog/hg38/cnv/all_CN.tsv',
        pog_sv = '/data/pog/pog_sv.tsv',

        tcga_tpm = '/data/tcga/hg38/tcga_rsem_gene_tpm.txt',
        tcga_snv = '/data/tcga/hg38/tcga_gdc_hg38_mut_fltr.txt',
        tcga_germline = '/data/tcga/germline_mut/PCA_pathVar_integrated_filtered_adjusted.tsv',
        tcga_cnv = '/data/tcga/cnv/TCGA.PANCAN.sampleMap_Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes',
        tcga_sv = '/data/tcga/sv/all_sv.txt',
        all_genes = '/data/all_genes.txt'
    output:
        pog_tpm_prcssd = 'tmp_data/pog_tpm_prcssd.tsv',
        pog_mut_prcssd = 'tmp_data/pog_mut_prcssd.tsv',
        pog_cnv_prcssd = 'tmp_data/pog_cnv_prcssd.tsv',
        pog_sv_prcssd = 'tmp_data/pog_sv_prcssd.tsv',

        tcga_tpm_prcssd = 'tmp_data/tcga_tpm_prcssd.tsv',
        tcga_mut_prcssd = 'tmp_data/tcga_mut_prcssd.tsv',
        tcga_cnv_prcssd = 'tmp_data/tcga_cnv_prcssd.tsv',
        tcga_sv_prcssd = 'tmp_data/tcga_sv_prcssd.tsv'
    params: gene_name = gene_of_interest
    message: 'Preprocessing datasets!'
    script: 'src/process_input.py'


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
    params: gene_name = gene_of_interest
    message: 'Making feature matrix and label vector for analysis!'
    script: 'src/make_feature_mat_label_vec.py'


rule fine_tune:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt'
    output:
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt'
    message: 'Fine-tuning hyperparameters for classification!'
    script: 'src/fine_tune.py'


rule test_performance_SNVs_only:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt',
        tcga_t_type = '/data/tcga/tcga_t_type.tsv'
    output:
        classification_results_SNVs_only = 'results/'+str(gene_of_interest)+'/classification_results_SNVs_only.txt',
        auroc_auprc_SNVs_only = 'results/'+str(gene_of_interest)+'/auroc_auprc_SNVs_only.jpg',
        f1_to_min_maj_ratio_plot_SNVs_only = 'results/'+str(gene_of_interest)+'/f1_to_min_maj_ratio_plot_SNVs_only.jpg'
    message: 'Test RF performance using SNVs/INDELs data only'
    script: 'src/test_performance_SNVs_only.py'


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
        label_vector_all = 'tmp_data/label_vector_all.txt',
        
        tcga_tpm_impactful_mut_cnv = 'tmp_data/tcga_tpm_impactful_mut_cnv.txt',
        tcga_tpm_wt_cnv = 'tmp_data/tcga_tpm_wt_cnv.txt',
        tcga_tpm_not_impactful_mut_cnv = 'tmp_data/tcga_tpm_not_impactful_mut_cnv.txt',
        pog_tpm_impactful_mut_cnv = 'tmp_data/pog_tpm_impactful_mut_cnv.txt',
        pog_tpm_wt_cnv = 'tmp_data/pog_tpm_wt_cnv.txt',
        pog_tpm_not_impactful_mut_cnv = 'tmp_data/pog_tpm_not_impactful_mut_cnv.txt'
    params: gene_name = gene_of_interest
    message: 'Utilizing CNV and SV data when contructing feature matrix and label vector'
    script: 'src/make_feature_mat_label_vec_w_all_data_types.py'


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
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt'
    output:
        data_types_combinations_results = 'results/'+str(gene_of_interest)+'/data_types_combinations_results.txt'
    message: 'Comparing the model performance on different data combinations'
    script: 'src/test_data_type_combinations.py'


rule analyze_performance_in_chosen_setting:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt',
        best_setting = '/data/best_classification_setting.tsv',
        tcga_tpm_not_impactful_mut = 'tmp_data/tcga_tpm_not_impactful_mut.txt',
        pog_tpm_not_impactful_mut = 'tmp_data/pog_tpm_not_impactful_mut.txt',
        tcga_tpm_not_impactful_mut_cnv = 'tmp_data/tcga_tpm_not_impactful_mut_cnv.txt',
        pog_tpm_not_impactful_mut_cnv = 'tmp_data/pog_tpm_not_impactful_mut_cnv.txt',
        tcga_mut_prcssd = 'tmp_data/tcga_mut_prcssd.tsv',
        pog_mut_prcssd = 'tmp_data/pog_mut_prcssd.tsv'
    output:
        gene_importance_scores_from_RF = 'results/'+str(gene_of_interest)+'/gene_importance_scores_from_RF.txt',
        pred_on_sample_w_not_impact_mut = 'results/'+str(gene_of_interest)+'/pred_on_sample_w_not_impact_mut.txt',
        not_impact_mut_groups_pred_n_binom_p_val = 'results/'+str(gene_of_interest)+'/not_impact_mut_groups_pred_n_binom_p_val.txt',
        not_impact_conseq_base_n_aa_changes = 'results/'+str(gene_of_interest)+'/not_impact_conseq_base_n_aa_changes.txt'
    params: gene_name = gene_of_interest
    message: 'Test RF performance using best setting'
    script: 'src/analyze_performance_in_chosen_setting.py'


rule find_threshold_for_important_genes:
    input:
        gene_importance_scores_from_RF = 'results/'+str(gene_of_interest)+'/gene_importance_scores_from_RF.txt',
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt',
        best_setting = '/data/best_classification_setting.tsv'
    output:
        num_important_genes = 'results/'+str(gene_of_interest)+'/num_important_genes.txt',
        true_vs_shuffled_importance_scores = 'results/'+str(gene_of_interest)+'/true_vs_shuffled_importance_scores.jpg',
        true_vs_shuffled_importance_scores_zoomed_in = 'results/'+str(gene_of_interest)+'/true_vs_shuffled_importance_scores_zoomed_in.jpg',
        true_vs_shuffled_importance_scores_zoomed_in2 = 'results/'+str(gene_of_interest)+'/true_vs_shuffled_importance_scores_zoomed_in2.jpg'
    params: gene_name = gene_of_interest
    message: 'Find threshold for genes contributing the most to the classification'
    script: 'src/find_threshold.py'


checkpoint classify_samples_by_tumour_types:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt',
        best_setting = '/data/best_classification_setting.tsv',
        tcga_t_type = '/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/data/pog/hg38/pog_t_type.tsv',
        tcga_tpm_impactful_mut = 'tmp_data/tcga_tpm_impactful_mut.txt',
        tcga_tpm_wt = 'tmp_data/tcga_tpm_wt.txt',
        pog_tpm_impactful_mut = 'tmp_data/pog_tpm_impactful_mut.txt',
        pog_tpm_wt = 'tmp_data/pog_tpm_wt.txt',
        tcga_tpm_impactful_mut_cnv = 'tmp_data/tcga_tpm_impactful_mut_cnv.txt',
        tcga_tpm_wt_cnv = 'tmp_data/tcga_tpm_wt_cnv.txt',
        pog_tpm_impactful_mut_cnv = 'tmp_data/pog_tpm_impactful_mut_cnv.txt',
        pog_tpm_wt_cnv = 'tmp_data/pog_tpm_wt_cnv.txt'
    output:
        temp('results/'+str(gene_of_interest)+'/individual_t_types/t_type_results.txt'),
        temp('results/'+str(gene_of_interest)+'/individual_t_types/t_type_gene_importance_scores.txt'),
        temp('results/'+str(gene_of_interest)+'/individual_t_types/t_type_results_balanced.txt'),
        temp('results/'+str(gene_of_interest)+'/individual_t_types/t_type_gene_importance_scores_balanced.txt')
    params: gene_name = gene_of_interest
    message: 'Run classification on each tumour type separately (for both balanced and imbalanced sets)'
    script: 'src/classify_by_each_t_type.py'


rule test_performance_on_tumour_types:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt',
        best_setting = '/data/best_classification_setting.tsv',
        tcga_t_type = '/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/data/pog/hg38/pog_t_type.tsv',
        tcga_tpm_not_impactful_mut = 'tmp_data/tcga_tpm_not_impactful_mut.txt',
        pog_tpm_not_impactful_mut = 'tmp_data/pog_tpm_not_impactful_mut.txt',
        tcga_tpm_not_impactful_mut_cnv = 'tmp_data/tcga_tpm_not_impactful_mut_cnv.txt',
        pog_tpm_not_impactful_mut_cnv = 'tmp_data/pog_tpm_not_impactful_mut_cnv.txt',
        tcga_mut_prcssd = 'tmp_data/tcga_mut_prcssd.tsv',
        pog_mut_prcssd = 'tmp_data/pog_mut_prcssd.tsv'
    output:
        specific_t_types_cv_results = 'results/'+str(gene_of_interest)+'/specific_t_types_cv_results.txt',
        gene_importance_scores_specific_tumour_types = 'results/'+str(gene_of_interest)+'/gene_importance_scores_specific_tumour_types.txt',
        pred_on_sample_w_not_impact_mut_specific_tumour_types = 'results/'+str(gene_of_interest)+'/pred_on_sample_w_not_impact_mut_specific_tumour_types.txt',
        not_impact_mut_groups_pred_n_binom_p_val = 'results/'+str(gene_of_interest)+'/not_impact_mut_groups_pred_n_binom_p_val_specific_tumour_types.txt',
        not_impact_base_n_aa_changes = 'results/'+str(gene_of_interest)+'/not_impact_conseq_base_n_aa_changes_specific_tumour_types.txt'
    params: gene_name = gene_of_interest
    message: 'Run classification on selected tumour types from the imbalanced analysis'
    script: 'src/performance_on_specific_tumour_types.py'


rule assess_performance_on_balanced_set_of_all_tumours:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt',
        best_setting = '/data/best_classification_setting.tsv',
        tcga_t_type = '/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/data/pog/hg38/pog_t_type.tsv'
    output:
        permut_balanced_results_all_tumours = 'results/'+str(gene_of_interest)+'/permut_balanced_results_all_tumours.txt'
    params: gene_name = gene_of_interest
    message: 'Run classification on balanced set of tumours in permutations!'
    script: 'src/assess_performance_on_balanced_set_of_all_tumours.py'
    

rule assess_performance_on_balanced_set_of_selected_tumours:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt',
        best_setting = '/data/best_classification_setting.tsv',
        tcga_t_type = '/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/data/pog/hg38/pog_t_type.tsv',
        tcga_tpm_impactful_mut = 'tmp_data/tcga_tpm_impactful_mut.txt',
        tcga_tpm_wt = 'tmp_data/tcga_tpm_wt.txt',
        pog_tpm_impactful_mut = 'tmp_data/pog_tpm_impactful_mut.txt',
        pog_tpm_wt = 'tmp_data/pog_tpm_wt.txt',
        tcga_tpm_impactful_mut_cnv = 'tmp_data/tcga_tpm_impactful_mut_cnv.txt',
        tcga_tpm_wt_cnv = 'tmp_data/tcga_tpm_wt_cnv.txt',
        pog_tpm_impactful_mut_cnv = 'tmp_data/pog_tpm_impactful_mut_cnv.txt',
        pog_tpm_wt_cnv = 'tmp_data/pog_tpm_wt_cnv.txt'
    output:
        permut_balanced_results_selected_tumours = 'results/'+str(gene_of_interest)+'/permut_balanced_results_selected_tumours.txt'
    params: gene_name = gene_of_interest
    message: 'Run classification on balanced set of tumours in permutations!'
    script: 'src/assess_performance_on_balanced_set_of_selected_tumours.py'


rule test_performance_on_balanced_tumour_types:
    input:
        feature_matrix = 'tmp_data/feature_matrix.txt',
        label_vector = 'tmp_data/label_vector.txt',
        feature_matrix_cnv = 'tmp_data/feature_matrix_cnv.txt',
        label_vector_cnv = 'tmp_data/label_vector_cnv.txt',
        best_hyper_param = 'results/'+str(gene_of_interest)+'/best_hyper_param.txt',
        best_setting = '/data/best_classification_setting.tsv',
        tcga_t_type = '/data/tcga/tcga_t_type.tsv',
        pog_t_type = '/data/pog/hg38/pog_t_type.tsv',
        tcga_tpm_not_impactful_mut = 'tmp_data/tcga_tpm_not_impactful_mut.txt',
        pog_tpm_not_impactful_mut = 'tmp_data/pog_tpm_not_impactful_mut.txt',
        tcga_tpm_not_impactful_mut_cnv = 'tmp_data/tcga_tpm_not_impactful_mut_cnv.txt',
        pog_tpm_not_impactful_mut_cnv = 'tmp_data/pog_tpm_not_impactful_mut_cnv.txt',
        tcga_tpm_impactful_mut = 'tmp_data/tcga_tpm_impactful_mut.txt',
        tcga_tpm_wt = 'tmp_data/tcga_tpm_wt.txt',
        pog_tpm_impactful_mut = 'tmp_data/pog_tpm_impactful_mut.txt',
        pog_tpm_wt = 'tmp_data/pog_tpm_wt.txt',
        tcga_tpm_impactful_mut_cnv = 'tmp_data/tcga_tpm_impactful_mut_cnv.txt',
        tcga_tpm_wt_cnv = 'tmp_data/tcga_tpm_wt_cnv.txt',
        pog_tpm_impactful_mut_cnv = 'tmp_data/pog_tpm_impactful_mut_cnv.txt',
        pog_tpm_wt_cnv = 'tmp_data/pog_tpm_wt_cnv.txt',
        tcga_mut_prcssd = 'tmp_data/tcga_mut_prcssd.tsv',
        pog_mut_prcssd = 'tmp_data/pog_mut_prcssd.tsv'
    output:
        gene_importance_scores_specific_tumour_types_balanced = 'results/'+str(gene_of_interest)+'/gene_importance_scores_specific_tumour_types_balanced.txt',
        pred_on_sample_w_not_impact_mut_specific_tumour_types_balanced = 'results/'+str(gene_of_interest)+'/pred_on_sample_w_not_impact_mut_specific_tumour_types_balanced.txt',
        not_impact_mut_groups_pred_n_binom_p_val_balanced = 'results/'+str(gene_of_interest)+'/not_impact_mut_groups_pred_n_binom_p_val_balanced.txt',
        not_impact_base_n_aa_changes_balanced = 'results/'+str(gene_of_interest)+'/not_impact_base_n_aa_changes_balanced.txt',
        balanced_t_types_cv_results = 'results/'+str(gene_of_interest)+'/balanced_t_types_cv_results.txt'
    params: gene_name = gene_of_interest
    message: 'Run classification on selected tumour types from the balanced analysis'
    script: 'src/performance_on_balanced_tumour_types.py'
