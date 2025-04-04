import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy as hm
import matplotlib.pyplot as plt
from pseudobulk import compute_pseudobulk_dataframes
from Harmony import harmony
from EMD import EMD_distances
from VectorDistance import sample_distance
from ChiSquare import chi_square_distance
from jensenshannon import jensen_shannon_distance
from Test import sample_anndata_by_sample, treecor_seurat_mapping,count_samples_in_adata, test_harmony
from Visualization import visualization_harmony, plot_cell_type_proportions_pca, plot_pseudobulk_batch_test_pca
from PCA import process_anndata_with_pca
from CCA import CCA_Call
from CellType import cell_types, cell_type_assign
from CCA_test import find_optimal_cell_resolution, cca_pvalue_test

def wrapper(
    # ===== Harmony Preprocessing Parameters =====
    h5ad_path,
    sample_meta_path,
    output_dir,
    cell_column='cell_type',
    cell_meta_path=None,
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
    umap=False,
    cell_type_save=True,

    # ===== Cell Type Assignment Parameters =====
    assign_save=True,

    # ===== Pseudobulk Parameters =====
    batch_col='batch',
    sample_col='sample',
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

    # ===== CCA Parameters =====
    summary_sample_csv_path=None,
    summary_cell_csv_path=None,
    cca_output_dir=None,
    cca_optimal_cell_resolution=False,
    cca_pvalue=False,

    # ===== Paths for Skipping Preprocessing =====
    AnnData_cell_path=None,
    # ===== Distance Methods =====
    methods=None,
    EMD = False,
    chi_square = False,
    jensen_shannon = False,

    # ===== Distance Methods =====
    verbose_Visualization = True,
    grouping_columns=['sev.level'],
    age_bin_size=None,
    verbose_visualization=True,
    dot_size=3,

    plot_dendrogram_flag=True,
    plot_umap_by_plot_group_flag=True,
    plot_umap_by_cell_type_flag=True,
    plot_pca_2d_flag=True,
    plot_pca_3d_flag=True,
    plot_3d_cells_flag=True,

    plot_cell_type_proportions_pca_flag=False,
    plot_pseudobulk_expression_pca_flag=False,
    plot_pseudobulk_batch_test_pca_flag=False,

    # ===== Process Control Flags =====
    preprocessing=True,
    cell_type_cluster=True,
    sample_distance = True,
    pseudobulk=True,
    cca=True,
    Visualization = True

):
    if vars_to_regress is None:
        vars_to_regress = []

    if methods is None:
        methods = [
            'cosine', 'correlation'
        ]

    if pseudobulk_output_dir is None:
        pseudobulk_output_dir = output_dir

    if pca_output_dir is None:
        pca_output_dir = output_dir

    if cca_output_dir is None:
        cca_output_dir = output_dir

    # Step 1: Harmony Preprocessing
    if preprocessing:
        AnnData_cell, AnnData_sample = harmony(
            h5ad_path=h5ad_path,
            sample_meta_path=sample_meta_path,
            output_dir=output_dir,
            cell_column=cell_column,
            cell_meta_path=cell_meta_path,
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
        AnnData_cell = sc.read(AnnData_cell_path)
        AnnData_sample = sc.read(AnnData_sample_path)

    # Step 2: Cell Type Clustering
    if cell_type_cluster:
        AnnData_cell = cell_types(
            adata=AnnData_cell,
            cell_column=cell_column,
            existing_cell_types=existing_cell_types,
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

    # Step 3: Pseudobulk
    if pseudobulk:
        pseudobulk_df = compute_pseudobulk_dataframes(
            adata=AnnData_sample,
            batch_col=batch_col,
            sample_col=sample_col,
            celltype_col=celltype_col,
            output_dir=pseudobulk_output_dir,
            n_features=n_features,
            frac=frac,
            verbose=pseudobulk_verbose
        )

        process_anndata_with_pca(
            adata=AnnData_sample,
            pseudobulk=pseudobulk_df,
            n_expression_pcs=n_expression_pcs,
            n_proportion_pcs=n_proportion_pcs,
            output_dir=pca_output_dir,
            adata_path=AnnData_sample_path,
            verbose=pca_verbose
        )

    # Step 4: CCA
    if cca:
        CCA_Call(AnnData_sample, summary_sample_csv_path, cca_output_dir)

        cca_pvalue_test(
            AnnData_sample,
            sample_meta_path,
            "X_pca_expression",
            0.8449111150006337,  # could be parameterized
            cca_output_dir
        )

        find_optimal_cell_resolution(
            AnnData_cell,
            AnnData_sample,
            cca_output_dir,
            sample_meta_path,
            AnnData_sample_path,
            "X_pca_expression"
        )

        if cca_optimal_cell_resolution:
            column = "X_pca_proportion"
            find_optimal_cell_resolution(AnnData_cell, AnnData_sample, output_dir, sample_meta_path, AnnData_sample_path, column) 
            column = "X_pca_expression"
            find_optimal_cell_resolution(AnnData_cell, AnnData_sample, output_dir, sample_meta_path, AnnData_sample_path, column) 
        
        if cca_pvalue:
            cca_pvalue_test(
                AnnData_sample,
                sample_meta_path,
                "X_pca_expression",
                0.5807686668238389,  # could be parameterized
                cca_output_dir
            )

        plot_cell_type_proportions_pca(AnnData_sample, cca_output_dir)
        plot_pseudobulk_batch_test_pca(AnnData_sample, cca_output_dir)

        if os.path.exists(summary_sample_csv_path):
            os.remove(summary_sample_csv_path)
        if os.path.exists(summary_cell_csv_path):
            os.remove(summary_cell_csv_path)

    # Step 5: Sample Distance
    if sample_distance:
        for md in methods:
            print(f"\nRunning sample distance: {md}\n")
            sample_distance(
                AnnData_sample,
                os.path.join(output_dir, 'Sample'),
                md,
                summary_sample_csv_path,
                pseudobulk_df
            )
        
        if EMD:
            EMD_distances(
                AnnData_sample,
                os.path.join(output_dir, 'sample_level_EMD'),
                summary_sample_csv_path
            )

        if chi_square:
            chi_square_distance(
                AnnData_sample,
                os.path.join(output_dir, 'Chi_square_sample'),
                summary_sample_csv_path
            )

        if jensen_shannon:
            jensen_shannon_distance(
                AnnData_sample,
                os.path.join(output_dir, 'jensen_shannon_sample'),
                summary_sample_csv_path
            )
        
    #Visualization
    if Visualization:
        visualization_harmony(
            AnnData_sample,
            output_dir,
            grouping_columns=grouping_columns,
            age_bin_size=age_bin_size,
            verbose=verbose_Visualization,
            dot_size=dot_size,

            plot_dendrogram_flag=plot_dendrogram_flag,
            plot_umap_by_plot_group_flag=plot_umap_by_plot_group_flag,
            plot_umap_by_cell_type_flag=plot_umap_by_cell_type_flag,
            plot_pca_2d_flag=plot_pca_2d_flag,
            plot_pca_3d_flag=plot_pca_3d_flag,
            plot_3d_cells_flag=plot_3d_cells_flag,

            plot_cell_type_proportions_pca_flag=plot_cell_type_proportions_pca_flag,
            plot_pseudobulk_expression_pca_flag=plot_pseudobulk_expression_pca_flag,
            plot_pseudobulk_batch_test_pca_flag=plot_pseudobulk_batch_test_pca_flag
        )

    print("End of Process\n")