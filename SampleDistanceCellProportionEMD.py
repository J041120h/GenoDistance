import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import seaborn as sns
from pyemd import emd
from anndata import AnnData
import warnings
from anndata._core.aligned_df import ImplicitModificationWarning
from Visualization import plot_cell_type_abundances, visualizeGroupRelationship, visualizeDistanceMatrix
from distanceTest import distanceCheck

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

def calculate_sample_distances_cell_proprotion(
    adata: AnnData,
    output_dir: str,
    cell_type_column: str = 'leiden',
    sample_column: str = 'sample',
) -> pd.DataFrame:
    """
    Calculate distances between samples based on the proportions of each cell type using Earth Mover's Distance (EMD).

    This function computes the EMD between each pair of samples by considering the distribution of cell types within each sample.
    The ground distance between cell types is defined based on the Euclidean distances between their centroids in PCA space.

    Parameters:
    ----------
    adata : AnnData
        The integrated single-cell dataset obtained from the previous analysis.
    output_dir : str
        Directory to save the output files.
    cell_type_column : str, optional
        Column name in `adata.obs` that contains the cell type assignments (default: 'leiden').
    sample_column : str, optional
        Column name in `adata.obs` that contains the sample information (default: 'sample').

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
    
        # Normalization
    min_val = proportions.min().min()
    max_val = proportions.max().max()

    # Avoid division by zero
    if max_val > min_val:
        # Normalize the DataFrame
        proportions = (proportions - min_val) / (max_val - min_val)
    else:
        # If all values are the same, set proportions to zero or handle accordingly
        proportions = proportions - min_val  # This will set all values to zero
    
    # 2. Compute ground distance matrix between cell types
    # We'll use the centroids of cell types in PCA space
    cell_type_centroids = {}
    for cell_type in cell_types:
        indices = adata.obs[adata.obs[cell_type_column] == cell_type].index
        # Get PCA coordinates
        if 'X_pca' in adata.obsm:
            coords = adata.obsm['X_pca'][adata.obs_names.isin(indices)]
        else:
            raise ValueError("PCA coordinates not found in adata.obsm['X_pca']")
        centroid = np.mean(coords, axis=0)
        cell_type_centroids[cell_type] = centroid

    # Now compute pairwise distances between cell type centroids
    centroids_matrix = np.vstack([cell_type_centroids[ct] for ct in cell_types])
    nd_distance = cdist(centroids_matrix, centroids_matrix, metric='euclidean')

    # Ensure that the ground distance matrix is of type float64
    ground_distance = nd_distance.astype(np.float64)
    max_distance = ground_distance.max()
    if max_distance > 0:
        ground_distance /= max_distance

    # 3. Compute EMD between each pair of samples
    num_samples = len(samples)
    sample_distance_matrix = pd.DataFrame(0, index=samples, columns=samples, dtype=np.float64)

    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i < j:
                hist_i = proportions.loc[sample_i].values
                hist_j = proportions.loc[sample_j].values

                # EMD requires histograms to sum to the same value
                # Since they are normalized proportions, they sum to 1
                distance = emd(hist_i, hist_j, ground_distance)
                sample_distance_matrix.loc[sample_i, sample_j] = distance
                sample_distance_matrix.loc[sample_j, sample_i] = distance
    
    # Save the distance matrix
    distance_matrix_path = os.path.join(output_dir, 'sample_distance_proportion_matrix.csv')
    sample_distance_matrix.to_csv(distance_matrix_path)
    distanceCheck(distance_matrix_path)
    print(f"Sample distance proportion matrix saved to {distance_matrix_path}")

    #save the cell type distribution map
    cell_type_distribution_map = os.path.join(output_dir, 'cell_type_distribution.pdf')
    plot_cell_type_abundances(proportions, output_dir)
    print(f"Cell type distirbution in Sample saved to {cell_type_distribution_map}")

    # generate a heatmap for sample distance
    heatmap_path = os.path.join(output_dir, 'sample_distance_proportion_heatmap.pdf')
    visualizeDistanceMatrix(sample_distance_matrix, heatmap_path)
    visualizeGroupRelationship(sample_distance_matrix, outputDir=output_dir, heatmap_path=os.path.join(output_dir, 'sample_proportion_relationship.pdf'))

    return sample_distance_matrix