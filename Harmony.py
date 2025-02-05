import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Local imports from your project
from HierarchicalConstruction import cell_type_dendrogram
from HVG import find_hvgs

def treecor_harmony(h5ad_path,
                    sample_meta_path,
                    output_dir,
                    cell_meta_path=None,
                    markers=None,
                    cluster_resolution=1,
                    num_PCs=30,
                    num_harmony=10,
                    num_features=2000,
                    min_cells=5,
                    min_features=5,
                    pct_mito_cutoff=20,
                    exclude_genes=None,
                    method='average',
                    metric='euclidean',
                    distance_mode='centroid',
                    vars_to_regress=[],
                    verbose=True):
    """
    Harmony Integration with proportional HVG selection by cell type,
    now reading an existing H5AD file that only contains raw counts (no meta).
    """

    # 0. Create output directories if not present
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating output directory")

    # Append 'harmony' subdirectory
    output_dir = os.path.join(output_dir, 'harmony')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print("Automatically generating harmony subdirectory")

    # 1. Read the raw count data from an existing H5AD
    if verbose:
        print('=== Read input dataset ===')
    adata = sc.read_h5ad(h5ad_path)
    if verbose:
        print(f'Dimension of raw data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # 2. Filter out any genes that have zero expression across all cells
    #    If your AnnData already uses the shape (cells, genes), do:
    sc.pp.filter_genes(adata, min_cells=1)

    # Optionally, also filter out any cells that have zero genes expressed
    sc.pp.filter_cells(adata, min_genes=1)

    # Additional user-defined filtering by min_cells and min_features
    #   (Note: These can be redundant if you've already filtered above;
    #    but we keep them for completeness.)
    if min_cells > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells)
    if min_features > 0:
        sc.pp.filter_cells(adata, min_genes=min_features)

    # 3. Calculate mitochondrial gene percentage & filter
    #    We'll assume your gene symbols for mitochondrial genes start with "MT-"
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Filter out cells with > pct_mito_cutoff% mitochondrial counts
    adata = adata[adata.obs['pct_counts_mt'] < pct_mito_cutoff].copy()

    # Exclude (remove) mitochondrial genes and any other user-specified genes
    mt_genes = adata.var_names[adata.var_names.str.startswith('MT-')]
    if exclude_genes is not None:
        genes_to_exclude = set(exclude_genes) | set(mt_genes)
    else:
        genes_to_exclude = set(mt_genes)
    adata = adata[:, ~adata.var_names.isin(genes_to_exclude)].copy()

    if verbose:
        print(f'Dimension of processed data (cells x genes): {adata.shape[0]} x {adata.shape[1]}')

    # 4. Attach cell metadata if provided
    if cell_meta_path is None:
        # If no cell metadata, try to parse sample IDs from cell names
        # e.g. if cell name is "SAMPLE1:ATCGG"
        adata.obs['sample'] = adata.obs_names.str.split(':').str[0]
    else:
        cell_meta = pd.read_csv(cell_meta_path)
        # Assuming 'barcode' is the column with cell IDs
        cell_meta.set_index('barcode', inplace=True)
        # Join with adata.obs
        adata.obs = adata.obs.join(cell_meta, how='left')

    # 5. Attach sample metadata
    sample_meta = pd.read_csv(sample_meta_path)
    # Assuming 'sample' is the column to merge with
    adata.obs = adata.obs.merge(sample_meta, on='sample', how='left')

    # ----------------------------------------------------------------------
    # We now split the data for two analyses:
    #    (a) Clustering with Harmony (adata_cluster)
    #    (b) Sample differences (adata_sample_diff)
    # ----------------------------------------------------------------------
    # adata_cluster = adata.copy()      # with batch effect correction
    adata_cluster = adata.copy()
    adata_sample_diff = adata.copy()  # without batch effect correction

    # ==================== (a) Clustering with Harmony ====================
    if verbose:
        print('=== Processing data for clustering (mediating batch effects) ===')

    sc.pp.normalize_total(adata_cluster, target_sum=1e4)
    # Normalize and log transform
    sc.pp.log1p(adata_cluster)
    adata_cluster.raw = adata_cluster.copy()

    # Find highly variable genes (HVGs)
    sc.pp.highly_variable_genes(
        adata_cluster,
        n_top_genes=num_features,
        flavor='seurat_v3',
        batch_key='sample'
    )

    adata_cluster = adata_cluster[:, adata_cluster.var['highly_variable']].copy()

    # Scale data
    sc.pp.scale(adata_cluster, max_value=10)
    # PCA
    sc.pp.regress_out(adata_cluster, ['batch'])
    sc.tl.pca(adata_cluster, n_comps=num_PCs, svd_solver='arpack')
    # Harmony batch correction
    if verbose:
        print('=== Running Harmony integration for clustering ===')
        print('Variables to be regressed out: ', ','.join(vars_to_regress))
        print(f'Clustering cluster_resolution: {cluster_resolution}')

    vars_to_regress_for_harmony = vars_to_regress.copy()
    if "sample" not in vars_to_regress_for_harmony:
        vars_to_regress_for_harmony.append("sample")

    # ho = hm.run_harmony(adata_cluster.obsm['X_pca'], adata_cluster.obs, vars_to_regress_for_harmony)
    # adata_cluster.obsm['X_pca_harmony'] = ho.Z_corr.T

    sc.external.pp.harmony_integrate(adata_cluster, vars_to_regress_for_harmony)

    # Clustering
    # If "celltype" is not in the metadata, we perform Leiden clustering.
    if 'celltype' in adata_cluster.obs.columns:
        adata_cluster.obs['cell_type'] = adata_cluster.obs['celltype'].astype('category')
        if markers is not None:
            marker_dict = {i: markers[i - 1] for i in range(1, len(markers) + 1)}
            adata_cluster.obs['cell_type'] = adata_cluster.obs['cell_type'].map(marker_dict)
    else:
        sc.tl.leiden(
            adata_cluster,
            resolution=cluster_resolution,
            flavor='igraph',
            n_iterations=1,
            directed=False,
            key_added='cell_type'
        )
        # Convert cluster IDs to "1, 2, 3..."
        adata_cluster.obs['cell_type'] = (adata_cluster.obs['cell_type'].astype(int) + 1).astype(str)

    #----------------log error----------------
    # Marker genes for dendrogram
    sc.tl.rank_genes_groups(adata_cluster, groupby='cell_type', method='wilcoxon', n_genes=100)
    rank_results = adata_cluster.uns['rank_genes_groups']
    groups = rank_results['names'].dtype.names
    all_marker_genes = []
    for group in groups:
        all_marker_genes.extend(rank_results['names'][group])
    all_marker_genes = list(set(all_marker_genes))

    # Neighbors and UMAP
    sc.pp.neighbors(adata_cluster, use_rep='X_pca_harmony', n_pcs=num_harmony)
    sc.tl.umap(adata_cluster, min_dist=0.5)

    # Save results
    adata_cluster.write(os.path.join(output_dir, 'adata_cell.h5ad'))

    # ============== (b) Sample Differences (No batch correction) ==========
    if verbose:
        print('=== Processing data for sample differences (without batch effect correction) ===')

    # Ensure 'cell_type' is carried over
    if 'cell_type' not in adata_cluster.obs.columns or adata_cluster.obs['cell_type'].nunique() == 0:
        adata_cluster.obs['cell_type'] = '1'
    adata_sample_diff.obs['cell_type'] = adata_cluster.obs['cell_type']

    sc.pp.normalize_total(adata_sample_diff, target_sum=1e4)
    sc.pp.log1p(adata_sample_diff)
    adata_sample_diff.raw = adata_sample_diff.copy()

    # Find HVGs on raw counts
    find_hvgs(
        adata=adata_sample_diff,
        sample_column='sample',
        num_features=num_features,
        batch_key='cell_type',
        check_values=True,
        inplace=True
    )
    adata_sample_diff = adata_sample_diff[:, adata_sample_diff.var['highly_variable']].copy()

    sc.pp.scale(adata_sample_diff, max_value=10)

    # PCA + Harmony (optional for sample difference dimension reduction)
    sc.pp.regress_out(adata_sample_diff, ['batch'])
    sc.tl.pca(adata_sample_diff, n_comps=num_PCs, svd_solver='arpack', zero_center=True)
    # ha = hm.run_harmony(
    #     adata_sample_diff.obsm['X_pca'],
    #     adata_sample_diff.obs,
    #     vars_to_regress
    # )
    # adata_sample_diff.obsm['X_pca_harmony'] = ha.Z_corr.T

    sc.external.pp.harmony_integrate(adata_sample_diff, vars_to_regress_for_harmony)

    sc.pp.neighbors(adata_sample_diff, use_rep='X_pca_harmony', n_pcs=num_harmony, n_neighbors=15, metric='cosine')
    sc.tl.umap(adata_sample_diff, min_dist=0.3, spread=1.0)

    # Save final integrated data
    adata_sample_diff.write(os.path.join(output_dir, 'adata_sample.h5ad'))

    # === Visualizing sample relationship in 2D (PCA on average HVG expression) ===
    if verbose:
        print('=== Computing average HVG expression per sample ===')

    # HVG genes
    hvg_genes = adata_sample_diff.var_names[adata_sample_diff.var['highly_variable']]
    hvg_data = adata_sample_diff[:, hvg_genes].to_df()
    hvg_data['sample'] = adata_sample_diff.obs['sample'].values
    sample_means = hvg_data.groupby('sample').mean()

    if verbose:
        print(f'Computed average expression for {sample_means.shape[0]} samples and {sample_means.shape[1]} HVGs.')

    if verbose:
        print('=== End of processing ===')

    return adata_cluster, adata_sample_diff


def visualization_harmony(
    adata_cluster,
    adata_sample_diff,
    output_dir,
    severity_col= None,
    verbose=True
):
    """
    Generate harmony-related plots, coloring samples by severity level (stored in severity_col)
    or, if not provided, by the first two letters of the sample name.

    This version also computes the sample-level PCA (based on average HVG expression)
    within the function, so there's no need for an external pca_df.

    Parameters
    ----------
    adata_cluster : anndata.AnnData
        Batch-corrected data for clustering (Harmony corrected).
    adata_sample_diff : anndata.AnnData
        Data used for comparing sample differences (optionally Harmony corrected).
    output_dir : str
        The directory where plots will be saved.
    severity_col : str, optional
        The name of the column in `adata.obs` that indicates severity/grouping.
        If None, defaults to the first two letters of the 'sample' name.
    verbose : bool, optional
        If True, prints extra messages.
    """

    # -----------------------------
    # 1. Define Plotting Groups
    # -----------------------------
    # We'll create a column 'plot_group' in each AnnData to store the grouping (severity).
    output_dir = os.path.join(output_dir, 'harmony')
    
    if severity_col is not None and severity_col in adata_cluster.obs.columns:
        adata_cluster.obs['plot_group'] = adata_cluster.obs[severity_col].astype(str)
        adata_sample_diff.obs['plot_group'] = adata_sample_diff.obs[severity_col].astype(str)
        if verbose:
            print(f"[visualization_harmony] Using '{severity_col}' as severity grouping.")
    else:
        # Fall back on the first two letters of the 'sample' column
        adata_cluster.obs['plot_group'] = adata_cluster.obs['sample'].str[:2]
        adata_sample_diff.obs['plot_group'] = adata_sample_diff.obs['sample'].str[:2]
        if verbose:
            print("[visualization_harmony] No valid severity column provided; "
                  "defaulting to first two letters of sample name.")

    # Make sure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --------------------------------
    # 2. Dendrogram (by cell_type)
    # --------------------------------
    sc.pl.dendrogram(
        adata_cluster,
        groupby='cell_type',
        show=False
    )
    plt.savefig(os.path.join(output_dir, 'phylo_tree.pdf'))
    plt.close()

    # --------------------------------
    # 3. UMAP colored by plot_group (Harmony-corrected clusters)
    # --------------------------------
    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_cluster,
        color='plot_group',
        legend_loc='right margin',
        frameon=False,
        size=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_umap_by_plot_group.pdf'), bbox_inches='tight')
    plt.close()

    # --------------------------------
    # 4. UMAP colored by plot_group (Sample-differences AnnData)
    # --------------------------------
    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata_sample_diff,
        color='plot_group',
        legend_loc='right margin',
        frameon=False,
        size=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_umap_by_plot_group.pdf'), bbox_inches='tight')
    plt.close()

    # --------------------------------
    # 5. PCA of Average HVG Expression (computed here)
    # --------------------------------
    if verbose:
        print("[visualization_harmony] Computing sample-level PCA from average HVG expression.")

    df = pd.DataFrame(adata_sample_diff.X, index=adata_sample_diff.obs_names, columns=adata_sample_diff.var_names)
    df['sample'] = adata_sample_diff.obs['sample']
    sample_means = df.groupby('sample').mean()

    # Perform PCA on sample-level means
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(sample_means)
    # Build a dataframe of PC coordinates
    pca_df = pd.DataFrame(pca_coords, index=sample_means.index, columns=['PC1', 'PC2'])
    print("\n PCA complete")
    
    # Join the plot_group to pca_df for color labeling
    sample_to_group = (
        adata_sample_diff.obs[['sample', 'plot_group']]
        .drop_duplicates()
        .set_index('sample')
    )
    pca_df = pca_df.join(sample_to_group, how='left')

    # Plot the 2D PCA, color by plot_group
    plt.figure(figsize=(10, 8))
    for grp in pca_df['plot_group'].unique():
        mask = (pca_df['plot_group'] == grp)
        plt.scatter(
            pca_df.loc[mask, 'PC1'],
            pca_df.loc[mask, 'PC2'],
            label=grp,
            s=100,
            alpha=0.8
        )
        # Optionally label each point with the sample name
        for sample_name, row in pca_df.loc[mask].iterrows():
            plt.text(
                row['PC1'],
                row['PC2'],
                sample_name,
                fontsize=8,
                ha='right'
            )
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Average HVG Expression per Sample')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_relationship_pca_by_plot_group.pdf'))
    plt.close()

    if verbose:
        print("[visualization_harmony] All harmony visualizations have been saved.")