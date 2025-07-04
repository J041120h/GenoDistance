import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy as hm
import matplotlib.pyplot as plt
import subprocess
import sys
import platform
import shutil
from pseudo_adata import compute_pseudobulk_adata
from Harmony import harmony
from EMD import EMD_distances
from VectorDistance import sample_distance
from ChiSquare import chi_square_distance
from jensenshannon import jensen_shannon_distance
from Visualization import visualization
from DR import process_anndata_with_pca
from CCA import CCA_Call
from CellType import cell_types, cell_type_assign
from CCA_test import find_optimal_cell_resolution, cca_pvalue_test
from TSCAN import TSCAN
from trajectory_diff_gene import identify_pseudoDEGs, summarize_results, run_differential_analysis_for_all_paths
from cluster import cluster
from sample_clustering.RAISIN import *
from sample_clustering.RAISIN_TEST import *
from ATAC_general_pipeline import *
from ATAC_visualization import *
from ATAC_CCA_test import *
from cell_proportion_pseudotime import visualize_cell_proportion_pseudotime

def wrapper(
    # ===== Harmony Preprocessing Parameters =====
    h5ad_path,
    sample_meta_path,
    output_dir,
    sample_col = 'sample',
    grouping_columns = ['sev.level'],
    cell_column='cell_type',
    cell_meta_path=None,
    batch_col=[],
    markers=None,
    cluster_resolution=0.8,
    num_PCs=20,
    num_harmony=30,
    num_features=2000,
    min_cells=500,
    min_features=500,
    pct_mito_cutoff=20,
    exclude_genes=None,
    doublet=True,
    combat=True,
    method='average',
    metric='euclidean',
    distance_mode='centroid',
    vars_to_regress=None,
    verbose=True,

    # ===== Cell Type Clustering Parameters =====
    existing_cell_types=False,
    n_target_cell_clusters = None,
    umap=False,
    cell_type_save=True,

    # ===== Cell Type Assignment Parameters =====
    assign_save=True,

    # ===== Pseudobulk Parameters =====
    celltype_col='cell_type',
    pseudobulk_output_dir=None,
    n_features=2000,
    frac=0.3,
    pseudobulk_verbose=True,

    # ===== PCA Parameters =====
    n_expression_pcs=10,
    n_proportion_pcs=10,
    pca_output_dir=None,
    AnnData_sample_path=None,
    pca_verbose=True,

    # ===== Trajectory Analysis Parameters =====
    trajectory_supervised = False,

    cca_output_dir=None,
    sev_col_cca = "sev.level",
    cca_optimal_cell_resolution=False,
    cca_pvalue=False,
    trajectory_verbose = True, 
    TSCAN_origin = None,

    # ===== Trajectory Differential Gene Parameters =====
    fdr_threshold = 0.05,
    effect_size_threshold = 1,
    top_n_genes = 100,
    trajectory_diff_gene_covariate = None,
    num_splines = 5,
    spline_order = 3,
    trajectory_diff_gene_output_dir = None,
    visualization_gene_list = None,
    visualize_all_deg = True,
    top_n_heatmap = 50,
    trajectory_diff_gene_verbose = True,
    top_gene_number = 30,
    n_pcs_for_null = 10,

    # ===== Paths for Skipping Preprocessing =====
    AnnData_cell_path=None,
    # ===== Distance Methods =====
    summary_sample_csv_path = None,
    sample_distance_methods=None,

    # ===== Distance Methods =====
    verbose_Visualization = True,
    trajectory_visualization_label=['sev.level'],
    age_bin_size=None,
    dot_size=3,

    plot_dendrogram_flag=True,
    plot_cell_umap_by_plot_group_flag=True,
    plot_umap_by_cell_type_flag=True,
    plot_pca_2d_flag=True,
    plot_pca_3d_flag=True,
    plot_3d_cells_flag=True,

    plot_cell_type_proportions_pca_flag=True,
    plot_cell_type_expression_pca_flag=True,
    plot_pseudobulk_batch_test_expression_flag=False,
    plot_pseudobulk_batch_test_proportion_flag=False,

    # ===== Cluster Based DEG ===== 
    Kmeans_based_cluster_flag = False,
    Tree_building_method = ['HRA_VEC', 'HRC_VEC', 'NN', 'UPGMA'],
    proportion_test = False,
    RAISIN_analysis = False,
    cluster_distance_method = 'cosine',
    cluster_number = 4,
    user_provided_sample_to_clade = None,

    # ATAC-seq Pipeline Parameters with Default Values
    atac_preprocessing = True,
    atac_pseudobulk_dimensionality_reduction = True,
    atac_visualization_processing = True,

    # File paths and directories
    atac_file_path = "data/atac_data.h5ad",
    atac_output_dir = None,
    atac_metadata_path = None,
    atac_pseudobulk_output_dir = None,
    atac_pca_output_dir = None,

    # Column specifications
    atac_sample_col = "sample",
    atac_batch_col = None,
    atac_cell_type_column = "cell_type",

    # Pipeline configuration
    atac_pipeline_verbose = True,
    use_snapatac2_dimred = False,

    # QC and filtering parameters
    atac_min_cells = 3,
    atac_min_genes = 2000,
    atac_max_genes = 15000,
    atac_min_cells_per_sample = 10,

    atac_doublet = True,
    atac_tfidf_scale_factor = 1e4,
    atac_num_features = 40000,
    atac_n_lsi_components = 30,
    atac_drop_first_lsi = True,
    atac_harmony_max_iter = 30,
    atac_n_neighbors = 15,
    atac_n_pcs = 30,

    # Leiden clustering parameters
    atac_leiden_resolution = 0.8,
    atac_existing_cell_types = False,
    atac_n_target_cell_clusters = None,

    # UMAP parameters
    atac_umap_min_dist = 0.3,
    atac_umap_spread = 1.0,
    atac_umap_random_state = 42,
    atac_plot_dpi = 300,

    atac_cell_path = None,
    atac_sample_path = None,

    # Pseudobulk parameters
    atac_pseudobulk_n_features = 50000,
    atac_pseudobulk_verbose = True,

    # PCA parameters
    atac_pca_n_expression_pcs = 30,
    atac_pca_n_proportion_pcs = 30,
    atac_pca_verbose = True,

    # Trajectory analysis parameters
    atac_cca_output_dir = None,
    trajectory_supervised_atac = True,
    atac_cca_optimal_cell_resolution = False,
    n_pcs_for_null_atac = 10,

    #visualization parameters
    atac_figsize=(10, 8),
    atac_point_size=50, 
    atac_visualization_grouping_columns=['current_severity'], 
    atac_show_sample_names = True,
    atac_visualization_age_size=None,
    # ===== Process Control Flags =====
    preprocessing=True,
    cell_type_cluster=True,
    sample_distance_calculation = True,
    DimensionalityReduction=True,
    trajectory_analysis=True,
    trajectory_differential_gene = True,
    cluster_and_DGE = True,
    ATAC_data = True,
    trajectory_analysis_atac = True, 
    visualize_data = True,
    initialization=True,
    use_gpu= False
):
    ## ====== Preprocessing to add ungiven parameter======
    linux_system = False
    print("Start of Process\n")
    if vars_to_regress is None:
        vars_to_regress = []

    if sample_distance_methods is None:
        sample_distance_methods = [
            'cosine', 'correlation'
        ]

    if pseudobulk_output_dir is None:
        pseudobulk_output_dir = output_dir

    if pca_output_dir is None:
        pca_output_dir = output_dir

    if cca_output_dir is None:
        cca_output_dir = output_dir
    
    if summary_sample_csv_path is None:
        summary_sample_csv_path = os.path.join(output_dir, 'summary_sample.csv')
    
    if trajectory_diff_gene_output_dir is None:
        trajectory_diff_gene_output_dir = os.path.join(output_dir, 'trajectoryDEG')
    #Check the status of previous processing, to ensure consistent data processing
    status_file_path = os.path.join(output_dir, "sys_log", "process_status.json")
    status_flags = {
        "preprocessing": False,
        "cell_type_cluster": False,
        "sample_distance_calculation": False,
        "DimensionalityReduction": False,
        "trajectory_analysis": False,
        "visualize_data": False
    }

    # Ensure the sys_log directory exists
    os.makedirs(os.path.dirname(status_file_path), exist_ok=True)

    if os.path.exists(status_file_path) and not initialization:
        try:
            with open(status_file_path, 'r') as f:
                saved_status = json.load(f)
                status_flags.update(saved_status)
            print("Resuming process from previous progress:")
            print(json.dumps(status_flags, indent=4))
        except Exception as e:
            print(f"Error reading status file: {e}")
            print("Reinitializing status from scratch.")
    else:
        check_output_dir = os.path.join(output_dir, "result")
        if os.path.exists(check_output_dir):
            try:
                shutil.rmtree(check_output_dir)
                print(f"Removed existing result directory: {check_output_dir}")
            except Exception as e:
                print(f"Failed to remove existing result directory: {e}")
        print("Initializing process status file.")
        os.makedirs(output_dir, exist_ok=True)
        with open(status_file_path, 'w') as f:
            json.dump(status_flags, f, indent=4)

    # Add new function that if system dependent:
    if platform.system() == "Linux" and use_gpu:
        print("Linux system detected.")
        if initialization:
            # Install dependencies for rapids-singlecell
            subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "rapids-singlecell[rapids12]",
            "--extra-index-url=https://pypi.nvidia.com"
            ])
        linux_system = True
        from linux.harmony_linux import harmony_linux
        from linux.CellType_linux import cell_types_linux, cell_type_assign_linux
        from linux.CCA_test_linux import find_optimal_cell_resolution_linux
        from pseudo_adata_linux import compute_pseudobulk_adata_linux
        # Enable `managed_memory`
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        import cupy as cp
        print("\n\nEnabling managed memory for RMM...\n\n")
        rmm.reinitialize(
            managed_memory=True,
            pool_allocator=False,
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
    else: 
        print("nNon-Linux system detected.")
    
    # Step 1: Harmony Preprocessing
    if preprocessing:
        if linux_system and use_gpu:
            AnnData_cell, AnnData_sample = harmony_linux(
            h5ad_path=h5ad_path,
            sample_meta_path=sample_meta_path,
            output_dir=output_dir,
            sample_column = sample_col,
            cell_column=cell_column,
            cell_meta_path=cell_meta_path,
            batch_key = batch_col,
            markers=markers,
            cluster_resolution=cluster_resolution,
            num_PCs=num_PCs,
            num_harmony=num_harmony,
            num_features=num_features,
            min_cells=min_cells,
            min_features=min_features,
            pct_mito_cutoff=pct_mito_cutoff,
            exclude_genes=exclude_genes,
            doublet=doublet,
            combat=combat,
            method=method,
            metric=metric,
            distance_mode=distance_mode,
            vars_to_regress=vars_to_regress,
            verbose=verbose
        )
        else:
            AnnData_cell, AnnData_sample = harmony(
                h5ad_path=h5ad_path,
                sample_meta_path=sample_meta_path,
                output_dir=output_dir,
                sample_column = sample_col,
                cell_column=cell_column,
                cell_meta_path=cell_meta_path,
                batch_key = batch_col,
                markers=markers,
                cluster_resolution=cluster_resolution,
                num_PCs=num_PCs,
                num_harmony=num_harmony,
                num_features=num_features,
                min_cells=min_cells,
                min_features=min_features,
                pct_mito_cutoff=pct_mito_cutoff,
                exclude_genes=exclude_genes,
                doublet=doublet,
                combat=combat,
                method=method,
                metric=metric,
                distance_mode=distance_mode,
                vars_to_regress=vars_to_regress,
                verbose=verbose
            )
        status_flags["preprocessing"] = True
        with open(status_file_path, 'w') as f:
            json.dump(status_flags, f, indent=4)
    elif not ATAC_data:
        if not status_flags["preprocessing"]:
            raise ValueError("Preprocessing is skipped, but no preprocessed data found.")
        if not AnnData_cell_path or not AnnData_sample_path:
            temp_cell_path = os.path.join(output_dir, "harmony", "adata_cell.h5ad")
            temp_sample_path = os.path.join(output_dir, "harmony", "adata_sample.h5ad")
            if not os.path.exists(temp_cell_path) or not os.path.exists(temp_sample_path):
                raise ValueError("Preprocessed data paths are not provided and default files path do not exist.")
            AnnData_cell_path = temp_cell_path
            AnnData_sample_path = temp_sample_path    
        AnnData_cell = sc.read(AnnData_cell_path)
        AnnData_sample = sc.read(AnnData_sample_path)

    if not ATAC_data:
        if sample_col != 'sample':
            AnnData_sample.obs.rename(columns={sample_col: 'sample'}, inplace=True)
        if batch_col != 'batch':
            AnnData_sample.obs.rename(columns={batch_col: 'batch'}, inplace=True)
        sample_col = 'sample'
        batch_col = 'batch'
    # Step 2: Cell Type Clustering
    if cell_type_cluster:
        if linux_system:
            AnnData_cell = cell_types_linux(
                adata=AnnData_cell,
                cell_column=cell_column,
                existing_cell_types=existing_cell_types,
                n_target_clusters = n_target_cell_clusters,
                umap=umap,
                Save=cell_type_save,
                output_dir=output_dir,
                cluster_resolution=cluster_resolution,
                markers=markers,
                method=method,
                metric=metric,
                distance_mode=distance_mode,
                num_PCs=num_PCs,
                verbose=verbose
            )
            cell_type_assign_linux(
                adata_cluster=AnnData_cell,
                adata=AnnData_sample,
                Save=assign_save,
                output_dir=output_dir,
                verbose=verbose
            )
        else:
            AnnData_cell = cell_types(
                adata=AnnData_cell,
                cell_column=cell_column,
                existing_cell_types=existing_cell_types,
                n_target_clusters = n_target_cell_clusters,
                umap=umap,
                Save=cell_type_save,
                output_dir=output_dir,
                cluster_resolution=cluster_resolution,
                markers=markers,
                method=method,
                metric=metric,
                distance_mode=distance_mode,
                num_PCs=num_PCs,
                verbose=verbose
            )

            cell_type_assign(
                adata_cluster=AnnData_cell,
                adata=AnnData_sample,
                Save=assign_save,
                output_dir=output_dir,
                verbose=verbose
            )
        status_flags["cell_type_cluster"] = True
        with open(status_file_path, 'w') as f:
            json.dump(status_flags, f, indent=4)


    # Step 3: Pseudobulk and PCA
    if DimensionalityReduction:
        if status_flags["cell_type_cluster"] == False:
            raise ValueError("Cell type clustering is required before dimension reduction.")

        if linux_system:
            pseudobulk_df,pseudobulk_adata = compute_pseudobulk_adata_linux(
                adata=AnnData_sample,
                batch_col=batch_col,
                sample_col=sample_col,
                celltype_col=celltype_col,
                output_dir=pseudobulk_output_dir,
                n_features=n_features,
                verbose=pseudobulk_verbose
            )
        else:
            pseudobulk_df,pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_sample,
                batch_col=batch_col,
                sample_col=sample_col,
                celltype_col=celltype_col,
                output_dir=pseudobulk_output_dir,
                n_features=n_features,
                verbose=pseudobulk_verbose
            )

        pseudobulk_anndata = process_anndata_with_pca(
            adata=AnnData_sample,
            pseudobulk=pseudobulk_df,
            pseudobulk_anndata = pseudobulk_adata,
            sample_col = sample_col,
            n_expression_pcs=n_expression_pcs,
            n_proportion_pcs=n_proportion_pcs,
            output_dir=pca_output_dir,
            verbose=pca_verbose
        )

        status_flags["DimensionalityReduction"] = True
        with open(status_file_path, 'w') as f:
            json.dump(status_flags, f, indent=4)

    if (trajectory_analysis or sample_distance_calculation or cluster_and_DGE) and not DimensionalityReduction:
        pseudobulk_anndata = sc.read(os.path.join(pca_output_dir, "pseudobulk", "pseudobulk_sample.h5ad"))

    # Step 4: CCA
    if trajectory_analysis:
        if not status_flags["DimensionalityReduction"]:
            raise ValueError("Dimensionality reduction is required before trajectory analysis.")
        if trajectory_supervised:
            if sev_col_cca not in pseudobulk_anndata.obs.columns:
                raise ValueError(f"Severity column '{sev_col_cca}' not found in AnnData_sample.")
            first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression= CCA_Call(adata = pseudobulk_anndata, output_dir=cca_output_dir, sev_col = sev_col_cca, ptime = True, verbose = trajectory_verbose)
            if cca_pvalue:
                cca_pvalue_test(
                    pseudo_adata = pseudobulk_anndata,
                    column = "X_DR_proportion",
                    input_correlation = first_component_score_proportion,
                    output_directory = cca_output_dir,
                    num_simulations = 1000,
                    sev_col = sev_col_cca,
                    verbose = trajectory_verbose
                )

                cca_pvalue_test(
                    pseudo_adata = pseudobulk_anndata,
                    column = "X_DR_expression",
                    input_correlation = first_component_score_expression,
                    output_directory = cca_output_dir,
                    num_simulations = 1000,
                    sev_col = sev_col_cca,
                    verbose = trajectory_verbose
                )
            visualize_cell_proportion_pseudotime(ptime_expression,output_dir)
            if cca_optimal_cell_resolution:
                if linux_system:
                    find_optimal_cell_resolution_linux(
                        AnnData_cell = AnnData_cell,
                        AnnData_sample = AnnData_sample,
                        output_dir = cca_output_dir,
                        column = "X_DR_proportion",
                        sample_col = sample_col
                    )
                    find_optimal_cell_resolution_linux(
                        AnnData_cell = AnnData_cell,
                        AnnData_sample = AnnData_sample,
                        output_dir = cca_output_dir,
                        column = "X_DR_expression",
                        sample_col = sample_col
                    )
                else:
                    find_optimal_cell_resolution(
                        AnnData_cell = AnnData_cell,
                        AnnData_sample = AnnData_sample,
                        output_dir = cca_output_dir,
                        column = "X_DR_expression",
                        sample_col = sample_col,
                        n_features = n_features,
                        sev_col = sev_col_cca, 
                        batch_col = batch_col,
                        n_pcs_for_null = n_pcs_for_null

                    )
                    find_optimal_cell_resolution(
                        AnnData_cell = AnnData_cell,
                        AnnData_sample = AnnData_sample,
                        output_dir = cca_output_dir,
                        column = "X_DR_proportion",
                        sample_col = sample_col,
                        n_features = n_features,
                        sev_col = sev_col_cca, 
                        batch_col = batch_col,
                        n_pcs_for_null = n_pcs_for_null
                    )
            status_flags["trajectory_analysis"] = True
            with open(status_file_path, 'w') as f:
                json.dump(status_flags, f, indent=4)
        else:
            TSCAN_result_expression = TSCAN(AnnData_sample = AnnData_sample, column = "X_DR_expression", n_clusters = 8, output_dir = output_dir, grouping_columns = trajectory_visualization_label, verbose = trajectory_verbose, origin=TSCAN_origin)
            TSCAN_result_proportion = TSCAN(AnnData_sample = AnnData_sample, column = "X_DR_proportion", n_clusters = 8, output_dir = output_dir, grouping_columns = trajectory_visualization_label, verbose = trajectory_verbose, origin=TSCAN_origin)

        if trajectory_differential_gene:
            if trajectory_supervised:
                results = identify_pseudoDEGs(
                    pseudobulk= pseudobulk_df,
                    sample_meta_path=sample_meta_path,
                    ptime_expression=ptime_expression,
                    fdr_threshold= fdr_threshold,
                    effect_size_threshold= effect_size_threshold,
                    top_n_genes = top_n_genes,
                    covariate_columns = trajectory_diff_gene_covariate,
                    sample_col=sample_col,
                    num_splines = num_splines,
                    spline_order = spline_order,
                    visualization_gene_list = visualization_gene_list,
                    visualize_all_deg = visualize_all_deg,
                    top_n_heatmap = top_n_heatmap,
                    output_dir=trajectory_diff_gene_output_dir,
                    verbose=trajectory_diff_gene_verbose
                )
                if trajectory_diff_gene_verbose:
                    print("Finish finding DEG, summarizing results")
                summarize_results(
                    results = results, 
                    top_n=top_gene_number, 
                    output_file=os.path.join(trajectory_diff_gene_output_dir,"differential_gene_result.txt"),
                    verbose=trajectory_diff_gene_verbose
                )
            else:
                print("Running differential analysis for main path...")
                all_path_results = run_differential_analysis_for_all_paths(
                    TSCAN_results=TSCAN_result_expression,
                    pseudobulk_df=pseudobulk_df,
                    sample_meta_path=sample_meta_path,
                    sample_col= sample_col,
                    fdr_threshold=fdr_threshold,
                    effect_size_threshold=effect_size_threshold,
                    top_n_genes = top_n_genes,
                    covariate_columns=trajectory_diff_gene_covariate,
                    num_splines=num_splines,
                    spline_order=spline_order,
                    base_output_dir=trajectory_diff_gene_output_dir,
                    top_gene_number=top_gene_number,
                    visualization_gene_list = visualization_gene_list,
                    visualize_all_deg = visualize_all_deg,
                    top_n_heatmap = top_n_heatmap,
                    verbose=trajectory_diff_gene_verbose
                )

    if os.path.exists(summary_sample_csv_path):
            os.remove(summary_sample_csv_path)

    # Step 5: Sample Distance
    if sample_distance_calculation:
        if not status_flags["DimensionalityReduction"]:
            raise ValueError("Dimensionality reduction is required before sample distance calculation.")
        for md in sample_distance_methods:
            print(f"\nRunning sample distance: {md}\n")
            sample_distance(
                adata = AnnData_sample,
                output_dir = os.path.join(output_dir, 'Sample'),
                method = f'{md}',
                summary_csv_path = summary_sample_csv_path,
                pseudobulk = pseudobulk_df,
                sample_column = sample_col,
                grouping_columns = grouping_columns,
                )
        
        if "EMD" in sample_distance_methods:
            EMD_distances(
                adata = AnnData_sample,
                output_dir = os.path.join(output_dir, 'sample_level_EMD'),
                summary_csv_path = summary_sample_csv_path,
                cell_type_column = 'cell_type',
                sample_column = sample_col,
            )

        if "chi_square" in sample_distance_methods:
            chi_square_distance(
                adata = AnnData_sample,
                output_dir = os.path.join(output_dir, 'Chi_square_sample'),
                summary_csv_path = summary_sample_csv_path,
                sample_column = sample_col,
            )

        if "jensen_shannon" in sample_distance_methods:
            jensen_shannon_distance(
                adata = AnnData_sample,
                output_dir = os.path.join(output_dir, 'jensen_shannon_sample'),
                summary_csv_path = summary_sample_csv_path,
                sample_column = sample_col,
            )
        status_flags["sample_distance_calculation"] = True
        if verbose:
            print(f"Sample distance calculation completed. Results saved in {os.path.join(output_dir, 'Sample')}")
        with open(status_file_path, 'w') as f:
            json.dump(status_flags, f, indent=4)
    
    if cluster_and_DGE:
        if cluster_distance_method not in sample_distance_methods:
            raise ValueError(f"Distance method '{cluster_distance_method}' not found in sample distance methods.")
        expr_results, prop_results = cluster(
            Kmeans=Kmeans_based_cluster_flag,
            methods=Tree_building_method,
            prportion_test=proportion_test,
            generalFolder=output_dir,
            distance_method=cluster_distance_method,
            number_of_clusters=cluster_number,
            sample_to_clade_user=user_provided_sample_to_clade
        )
        
        if RAISIN_analysis:
            print("Running RAISIN analysis...")
            if expr_results is not None:
                unique_expr_clades = len(set(expr_results.values()))
                if unique_expr_clades <= 1:
                    print("Only one clade found in expression results. Skipping RAISIN analysis.")
                else:
                    fit = raisinfit(
                        adata_path = os.path.join(output_dir, 'harmony', 'adata_sample.h5ad'), 
                        sample_col = sample_col, 
                        batch_key = batch_col, 
                        sample_to_clade=expr_results, 
                        verbose = verbose,
                        intercept=True, 
                        n_jobs=-1,
                        )
                    run_pairwise_raisin_analysis(
                        fit=fit,
                        output_dir= os.path.join(output_dir, 'raisin_results_expression'),
                        min_samples=2,
                        fdrmethod='fdr_bh',
                        n_permutations=10,
                        fdr_threshold=0.05,
                        verbose=True
                    )
            else:
                print("No expression results available. Skipping RAISIN analysis.")
            if prop_results is not None:
                unique_prop_clades = len(set(prop_results.values()))
                if unique_prop_clades <= 1:
                    print("Only one clade found in proportion results. Skipping RAISIN analysis.")
                else:
                    fit = raisinfit(
                        adata_path = os.path.join(output_dir, 'harmony', 'adata_sample.h5ad'), 
                        sample_col = sample_col, 
                        batch_key = batch_col, 
                        sample_to_clade=prop_results, 
                        intercept=True, 
                        n_jobs=-1,
                        )
                    run_pairwise_raisin_analysis(
                        fit=fit,
                        output_dir= os.path.join(output_dir, 'raisin_results_proportion'),
                        min_samples=2,
                        fdrmethod='fdr_bh',
                        n_permutations=10,
                        fdr_threshold=0.05,
                        verbose=True
                    )
            else:
                print("No proportion results available. Skipping RAISIN analysis.")
    
    if ATAC_data:
        if atac_output_dir is None:
            atac_output_dir = os.path.join(output_dir, 'ATAC')
        if atac_pseudobulk_output_dir is None:
            atac_pseudobulk_output_dir = atac_output_dir
        if atac_pca_output_dir is None:
            atac_pca_output_dir = atac_output_dir
        if atac_cca_output_dir is None:
            atac_cca_output_dir = atac_output_dir
        os.makedirs(atac_output_dir, exist_ok=True)
        os.makedirs(atac_pseudobulk_output_dir, exist_ok=True)
        os.makedirs(atac_pca_output_dir, exist_ok=True)

        if atac_preprocessing:
            atac_sample, atac_cell = run_scatac_pipeline(
                filepath=atac_file_path,
                output_dir=atac_output_dir,
                metadata_path=atac_metadata_path,
                sample_column=atac_sample_col,
                batch_key=atac_batch_col,
                verbose=atac_pipeline_verbose,
                use_snapatac2_dimred=use_snapatac2_dimred,
                
                # QC and filtering parameters
                min_cells=atac_min_cells,
                min_genes=atac_min_genes,
                max_genes=atac_max_genes,
                min_cells_per_sample=atac_min_cells_per_sample,
                
                # Doublet detection
                doublet=atac_doublet,
                
                # TF-IDF parameters
                tfidf_scale_factor=atac_tfidf_scale_factor,
                
                # Highly variable features
                num_features=atac_num_features,
                
                # LSI / dimensionality reduction
                n_lsi_components=atac_n_lsi_components,
                drop_first_lsi=atac_drop_first_lsi,
                
                # Harmony integration
                harmony_max_iter=atac_harmony_max_iter,
                harmony_use_gpu=use_gpu,
                
                # Nearest neighbors
                n_neighbors=atac_n_neighbors,
                n_pcs=atac_n_pcs,
                
                # UMAP visualization
                umap_min_dist=atac_umap_min_dist,
                umap_spread=atac_umap_spread,
                umap_random_state=atac_umap_random_state,
            )
        else:
            if not atac_cell_path or not atac_sample_path:
                temp_cell_path = os.path.join(atac_output_dir, "harmony", "adata_cell.h5ad")
                temp_sample_path = os.path.join(atac_output_dir, "harmony", "adata_sample.h5ad")
                if not os.path.exists(temp_cell_path) or not os.path.exists(temp_sample_path):
                    raise ValueError("Preprocessed data paths are not provided and default files path do not exist.")
                AnnData_cell_path = temp_cell_path
                AnnData_sample_path = temp_sample_path    
            atac_cell = sc.read(AnnData_cell_path)
            atac_sample = sc.read(AnnData_sample_path)

        atac_sample = cell_types_atac(
            adata = atac_sample,
            cell_column=atac_cell_type_column, 
            existing_cell_types=atac_existing_cell_types,
            n_target_clusters=atac_n_target_cell_clusters,
            cluster_resolution=atac_leiden_resolution,
            use_rep='X_DM_harmony',
            method='average', 
            metric='euclidean', 
            distance_mode='centroid',
            num_DMs=atac_n_lsi_components, 
            output_dir = atac_output_dir,
            verbose=verbose
        )

        if atac_pseudobulk_dimensionality_reduction:
            atac_pseudobulk_df, pseudobulk_adata = compute_pseudobulk_adata(
                adata=atac_sample,
                batch_col=atac_batch_col,
                sample_col=atac_sample_col,
                celltype_col=atac_cell_type_column,
                output_dir=atac_pseudobulk_output_dir,
                n_features=atac_pseudobulk_n_features,
                atac = True,
                verbose=atac_pseudobulk_verbose
            )

            pseudobulk_anndata = process_anndata_with_pca(
                adata=atac_sample,
                pseudobulk=atac_pseudobulk_df,
                pseudobulk_anndata = pseudobulk_adata,
                sample_col = sample_col,
                n_expression_pcs=atac_pca_n_expression_pcs,
                n_proportion_pcs=atac_pca_n_proportion_pcs,
                output_dir=atac_pca_output_dir,
                atac = True,
                use_snapatac2_dimred = use_snapatac2_dimred,
                verbose=atac_pca_verbose
            )
        
        if trajectory_analysis_atac:
            if not atac_pseudobulk_dimensionality_reduction:
                pseudobulk_anndata = sc.read(os.path.join(atac_pseudobulk_output_dir, "pseudobulk", "pseudobulk_sample.h5ad"))
            if trajectory_supervised_atac:
                if sev_col_cca not in pseudobulk_anndata.obs.columns:
                    raise ValueError(f"Severity column '{sev_col_cca}' not found in AnnData_sample.")
                first_component_score_proportion, first_component_score_expression, ptime_proportion, ptime_expression= CCA_Call(adata = pseudobulk_anndata, output_dir=cca_output_dir, sev_col = sev_col_cca, ptime = True, verbose = trajectory_verbose)
                if cca_pvalue:
                    cca_pvalue_test(
                        pseudo_adata = pseudobulk_anndata,
                        column = "X_DR_expression",
                        input_correlation = first_component_score_expression,
                        output_directory = atac_cca_output_dir,
                        num_simulations = 1000,
                        sev_col = sev_col_cca,
                        verbose = trajectory_verbose
                    )
                    
                    cca_pvalue_test(
                        pseudo_adata = pseudobulk_anndata,
                        column = "X_DR_proportion",
                        input_correlation = first_component_score_proportion,
                        output_directory = atac_cca_output_dir,
                        num_simulations = 1000,
                        sev_col = sev_col_cca,
                        verbose = trajectory_verbose
                    )

                status_flags["trajectory_analysis"] = True
                with open(status_file_path, 'w') as f:
                    json.dump(status_flags, f, indent=4)
                
                if atac_cca_optimal_cell_resolution:
                    find_optimal_cell_resolution_atac(
                        AnnData_cell = atac_cell,
                        AnnData_sample = atac_sample,
                        output_dir = atac_cca_output_dir,
                        column = "X_DR_expression",
                        n_features = atac_pseudobulk_n_features,
                        sample_col = atac_sample_col,
                        batch_col = atac_batch_col,
                        num_DR_components = atac_pca_n_expression_pcs,
                        num_DMs = atac_n_lsi_components,
                        n_pcs = n_pcs_for_null_atac
                    )
                    
                    find_optimal_cell_resolution_atac(
                        AnnData_cell = atac_cell,
                        AnnData_sample = atac_sample,
                        output_dir = atac_cca_output_dir,
                        column = "X_DR_proportion",
                        n_features = atac_pseudobulk_n_features,
                        sample_col = atac_sample_col,
                        batch_col = atac_batch_col,
                        num_DR_components = atac_pca_n_proportion_pcs,
                        num_DMs = atac_n_lsi_components,
                        n_pcs = n_pcs_for_null_atac
                    )
                    
            else:
                TSCAN_result_expression = TSCAN(AnnData_sample = AnnData_sample, column = "X_pca_expression", n_clusters = 8, output_dir = output_dir, grouping_columns = trajectory_visualization_label, verbose = trajectory_verbose, origin=TSCAN_origin)
                TSCAN_result_proportion = TSCAN(AnnData_sample = AnnData_sample, column = "X_pca_proportion", n_clusters = 8, output_dir = output_dir, grouping_columns = trajectory_visualization_label, verbose = trajectory_verbose, origin=TSCAN_origin)

        if atac_visualization_processing:
            DR_visualization_all(
                atac_sample, 
                figsize=atac_figsize, 
                point_size=atac_point_size, 
                alpha=0.7, 
                output_dir=os.path.join(atac_output_dir, "visualization"), 
                grouping_columns=atac_visualization_grouping_columns,
                age_bin_size=atac_visualization_age_size, 
                show_sample_names = atac_show_sample_names,
                sample_col=atac_sample_col
            )


    if visualize_data:
        # if not status_flags["preprocessing"]:
        #     if plot_cell_umap_by_plot_group_flag or plot_umap_by_cell_type_flag:
        #         raise ValueError("Preprocessing is required before the required visualization.")
        # if not status_flags["cell_type_cluster"] and plot_cell_umap_by_plot_group_flag:
        #     raise ValueError("Cell type clustering is required before the required visualization.")
        
        if not status_flags["DimensionalityReduction"]:
            if plot_pca_2d_flag or plot_pca_3d_flag or plot_3d_cells_flag or plot_cell_type_proportions_pca_flag or plot_cell_type_expression_pca_flag or plot_pseudobulk_batch_test_expression_flag or plot_pseudobulk_batch_test_proportion_flag:
                raise ValueError("Dimensionality reduction is required before the required visualization.")
            raise ValueError("Dimensionality reduction is required before the required visualization.")

        
        visualization(
            adata_sample_diff = AnnData_sample,
            output_dir = os.path.join(output_dir, 'visualization'),
            grouping_columns = grouping_columns,
            age_bin_size = age_bin_size,
            verbose = verbose_Visualization,
            dot_size = dot_size,

            plot_dendrogram_flag=plot_dendrogram_flag,
            plot_umap_by_plot_group_flag=plot_cell_umap_by_plot_group_flag,
            plot_umap_by_cell_type_flag=plot_umap_by_cell_type_flag,
            plot_pca_2d_flag=plot_pca_2d_flag,
            plot_pca_3d_flag=plot_pca_3d_flag,
            plot_3d_cells_flag=plot_3d_cells_flag,

            plot_cell_type_proportions_pca_flag=plot_cell_type_proportions_pca_flag,
            plot_cell_type_expression_pca_flag=plot_cell_type_expression_pca_flag,
            plot_pseudobulk_batch_test_expression_flag=plot_pseudobulk_batch_test_expression_flag,
            plot_pseudobulk_batch_test_proportion_flag=plot_pseudobulk_batch_test_proportion_flag
        )
        status_flags["visualize_data"] = True
        with open(status_file_path, 'w') as f:
            json.dump(status_flags, f, indent=4)
        
        status_flags["initialization"] = False
        with open(status_file_path, 'w') as f:
            json.dump(status_flags, f, indent=4)
    print("End of Process\n")


if __name__ == '__main__':
    wrapper(
        h5ad_path = "/Users/harry/Desktop/GenoDistance/Data/count_data.h5ad", 
        output_dir = "/Users/harry/Desktop/GenoDistance/result", 
        sample_meta_path = "/Users/harry/Desktop/GenoDistance/Data/sample_data.csv", 
        preprocessing=True, 
        cell_type_cluster=True, 
        sample_distance_calculation = True, 
        DimensionalityReduction=True, 
        trajectory_analysis=True,
        visualize_data = True
        )