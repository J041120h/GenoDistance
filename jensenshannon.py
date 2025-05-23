import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import warnings
from scipy.sparse import issparse
from scipy.spatial.distance import jensenshannon 
from anndata._core.aligned_df import ImplicitModificationWarning
from Visualization import plot_cell_type_expression_heatmap, plot_cell_type_abundances, visualizeGroupRelationship, visualizeDistanceMatrix
from distanceTest import distanceCheck

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

def calculate_sample_distances_cell_proportion_jensenshannon(
    adata: AnnData,
    output_dir: str,
    summary_csv_path: str = "/users/harry/desktop/GenoDistance/result/summary.csv",
    cell_type_column: str = 'cell_type',
    sample_column: str = 'sample',
) -> pd.DataFrame:
    """
    Calculate distances between samples based on the proportions of each cell type using Jensen-Shannon Divergence.

    This function computes the Jensen-Shannon Divergence between each pair of samples by considering the distribution of cell types within each sample.

    Parameters:
    ----------
    adata : AnnData
        The integrated single-cell dataset obtained from the previous analysis.
    output_dir : str
        Directory to save the output files.
    cell_type_column : str, optional
        Column name in `adata.obs` that contains the cell type assignments (default: 'cell_type').
    sample_column : str, optional
        Column name in `adata.obs` that contains the sample information (default: 'sample').
    summary_csv_path : str, optional
        Path to the summary CSV file to record distance checks.

    Returns:
    -------
    sample_distance_matrix : pandas.DataFrame
        A symmetric matrix of distances between samples.
    """

    # Check if output directory exists and create if necessary
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating output directory")

    # Append 'cell_proportion' to the output directory path
    output_dir = os.path.join(output_dir, 'cell_proportion')

    # Create the new subdirectory if it doesn’t exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Automatically generating cell_proportion subdirectory")

    # 1. Compute cell type proportions in each sample
    samples = adata.obs[sample_column].unique()
    cell_types = adata.obs[cell_type_column].unique()

    # Create a DataFrame to hold proportions
    proportions = pd.DataFrame(0, index=samples, columns=cell_types, dtype=np.float64)

    for sample in samples:
        sample_data = adata.obs[adata.obs[sample_column] == sample]
        total_cells = sample_data.shape[0]
        counts = sample_data[cell_type_column].value_counts()
        proportions.loc[sample, counts.index] = counts.values / total_cells

    # 2. Compute Jensen-Shannon Divergence between each pair of samples
    num_samples = len(samples)
    sample_distance_matrix = pd.DataFrame(0, index=samples, columns=samples, dtype=np.float64)

    for i, sample_i in enumerate(samples):
        hist_i = proportions.loc[sample_i].values
        for j, sample_j in enumerate(samples):
            if i < j:
                hist_j = proportions.loc[sample_j].values

                # Compute Jensen-Shannon Divergence
                js_divergence = jensenshannon(hist_i, hist_j, base=2)
                sample_distance_matrix.loc[sample_i, sample_j] = js_divergence
                sample_distance_matrix.loc[sample_j, sample_i] = js_divergence

    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'sample_distance_proportion_matrix_jensenshannon.csv')
    sample_distance_matrix.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path, "cell_proportion", "Jensen-Shannon", summary_csv_path, adata)
    print(f"Sample distance proportion matrix saved to {distance_matrix_path}")

    # Save the cell type distribution map
    cell_type_distribution_map = os.path.join(output_dir, 'cell_type_distribution.pdf')
    plot_cell_type_abundances(proportions, output_dir)
    print(f"Cell type distribution in Sample saved to {cell_type_distribution_map}")

    # Generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_proportion_heatmap.pdf')
    visualizeDistanceMatrix(sample_distance_matrix, heatmap_path)
    visualizeGroupRelationship(
        sample_distance_matrix,
        outputDir=output_dir,
        adata = adata
    )

    return sample_distance_matrix

def jensen_shannon_distance(
    adata: AnnData,
    output_dir: str,
    summary_csv_path: str = "/users/harry/desktop/GenoDistance/result/summary.csv",
    sample_column: str = 'sample',
    normalize: bool = True,
    log_transform: bool = True
) -> pd.DataFrame:
    """
    Compute sample distances using Jensen-Shannon Divergence.

    This function computes distances between samples based on cell proportions, average expression, and weighted expression using Jensen-Shannon Divergence.

    Parameters:
    ----------
    adata : AnnData
        The integrated single-cell dataset.
    output_dir : str
        Directory to save the output files.
    summary_csv_path : str, optional
        Path to the summary CSV file to record distance checks.
    sample_column : str, optional
        Column name in `adata.obs` that contains the sample information (default: 'sample').
    normalize : bool, optional
        Whether to normalize the data (default: True).
    log_transform : bool, optional
        Whether to log-transform the data (default: True).

    Returns:
    -------
    None
    """
    method = "Jensen-Shannon"
    calculate_sample_distances_cell_proportion_jensenshannon(adata, output_dir, summary_csv_path, sample_column = sample_column)