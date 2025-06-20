import os
import contextlib
import io
import time
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Visualization import visualization
from combat.pycombat import pycombat
import warnings
import contextlib
from HVG import highly_variable_gene_selection, select_hvf_loess
from scipy.sparse import issparse, csr_matrix, vstack


def _normalize_and_log(vec: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Library-size normalisation followed by log1p."""
    tot = vec.sum()
    if tot > 0:
        vec = vec / tot * target_sum
    return np.log1p(vec)

def _tfidf_normalize(vec: np.ndarray, idf_weights: np.ndarray = None) -> np.ndarray:
    """
    Apply TF-IDF normalization to a vector.
    
    TF (Term Frequency) = count / total_counts
    IDF (Inverse Document Frequency) should be pre-computed across all samples
    
    Parameters:
    -----------
    vec : np.ndarray
        Raw count vector for a single sample
    idf_weights : np.ndarray
        Pre-computed IDF weights for each feature (gene/peak)
    
    Returns:
    --------
    np.ndarray
        TF-IDF normalized vector
    """
    # Compute TF (term frequency)
    total_counts = vec.sum()
    if total_counts > 0:
        tf = vec / total_counts
    else:
        tf = vec
    
    # Apply IDF weights if provided
    if idf_weights is not None:
        tfidf = tf * idf_weights
    else:
        tfidf = tf
    
    return tfidf


def compute_idf_weights(cell_expression_df: pd.DataFrame, verbose: bool = False) -> pd.Series:
    """
    Compute IDF weights across all samples for TF-IDF normalization.
    
    IDF = log(1 + N / (1 + df)) where:
    - N is the total number of samples
    - df is the number of samples containing the feature
    
    Parameters:
    -----------
    cell_expression_df : pd.DataFrame
        DataFrame with cell types as rows and samples as columns
        Each cell contains a Series of gene/peak expression values
    
    Returns:
    --------
    pd.Series
        IDF weights for each gene/peak
    """
    # Get gene names from the first non-empty cell
    gene_names = None
    for row in cell_expression_df.index:
        for col in cell_expression_df.columns:
            expr_series = cell_expression_df.at[row, col]
            if expr_series is not None and len(expr_series) > 0:
                gene_names = expr_series.index
                break
        if gene_names is not None:
            break
    
    if gene_names is None:
        raise ValueError("No valid expression data found in cell_expression_df")
    
    # Count document frequency for each gene/peak
    n_samples = 0
    doc_freq = np.zeros(len(gene_names))
    
    for row in cell_expression_df.index:
        for col in cell_expression_df.columns:
            expr_series = cell_expression_df.at[row, col]
            if expr_series is not None and len(expr_series) > 0:
                n_samples += 1
                # Count non-zero features
                doc_freq += (expr_series.values > 0).astype(int)
    
    # Compute IDF weights
    # Adding 1 to avoid division by zero and log(0)
    idf_weights = np.log(1 + n_samples / (1 + doc_freq))
    
    if verbose:
        print(f"Computed IDF weights for {len(gene_names)} features across {n_samples} samples")
        print(f"IDF weight range: [{idf_weights.min():.3f}, {idf_weights.max():.3f}]")
    
    return pd.Series(idf_weights, index=gene_names)

def check_nan_and_negative_in_lists(df: pd.DataFrame, verbose=False) -> bool:
    found_nan = False
    found_negative = False
    for row_index, row in df.iterrows():
        for col in df.columns:
            cell = row[col]
            if isinstance(cell, np.ndarray):
                if np.isnan(cell).any():
                    if verbose:
                        print(f"Found NaN value in cell at row {row_index}, column '{col}'.")
                    found_nan = True
                if (cell < 0).any():
                    if verbose:
                        print(f"Found negative value(s) in cell at row {row_index}, column '{col}'.")
                    found_negative = True
    if verbose and not found_nan and not found_negative:
        print("No NaN or negative values found in any numpy array cell.")
    return found_nan or found_negative


def vector_to_string(vector):
    """Convert a vector (list, np.array, etc.) to a full string representation without truncation."""
    arr = np.array(vector)
    return np.array2string(arr, threshold=np.inf, separator=', ')

def save_dataframe_as_strings(df: pd.DataFrame, pseudobulk_dir: str, filename: str, verbose=False):
    """Convert all cells in a DataFrame (where each cell is a vector) to strings without truncation, and save to a CSV file."""
    
    np.set_printoptions(threshold=np.inf)
    df_as_strings = df.applymap(lambda x: vector_to_string(x) if isinstance(x, (list, np.ndarray)) else str(x))

    file_path = os.path.join(pseudobulk_dir, filename)
    df_as_strings.to_csv(file_path, index=True)
    
    if verbose:
        print(f"DataFrame saved as strings to {file_path}")

def contains_nan_in_lists(df: pd.DataFrame, verbose=False) -> bool:
    found_nan = False
    for row_index, row in df.iterrows():
        for col in df.columns:
            cell = row[col]
            if isinstance(cell, np.ndarray) and np.isnan(cell).any():
                if verbose:
                    print(f"Found NaN in row {row_index}, column '{col}'.")
                found_nan = True
    return found_nan

def combat_correct_cell_expressions(
    adata: sc.AnnData,
    cell_expression_df: pd.DataFrame,
    cell_proportion_df: pd.DataFrame,
    pseudobulk_dir: str, 
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    parametric: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    
    start_time = time.time() if verbose else None

    check_nan_and_negative_in_lists(cell_expression_df)
    
    sample_batch_map = (
        adata.obs[[sample_col, batch_col]]
        .drop_duplicates()
        .set_index(sample_col)[batch_col]
        .to_dict()
    )
    
    example_series = next(
        (s for s in cell_expression_df.iloc[0].dropna() if s is not None and isinstance(s, pd.Series) and len(s) > 0),
        None
    )
    if example_series is None:
        raise ValueError("No valid Series found in cell_expression_df.")
    gene_names = example_series.index.tolist()
    n_genes = len(gene_names)
    
    corrected_df = cell_expression_df.copy(deep=True)
    cell_types_to_drop = set()
    
    for ctype in corrected_df.index:
        if ctype in cell_types_to_drop:
            continue
        
        row_data = corrected_df.loc[ctype]
        batch_labels = []
        arrays_for_this_ctype = []
        valid_sample_ids = []
        
        for sample_id in row_data.index:
            expr_series = row_data[sample_id]
            if expr_series is None or len(expr_series) == 0 or np.allclose(expr_series.values, 0):
                continue
            expr_array = np.nan_to_num(expr_series.values, nan=0, posinf=0, neginf=0)
            arrays_for_this_ctype.append(expr_array)
            batch_labels.append(sample_batch_map.get(sample_id, "missing_batch"))
            valid_sample_ids.append(sample_id)
        
        if len(arrays_for_this_ctype) == 0:
            if verbose:
                print(f"Deleting '{ctype}' because no samples with non-zero expression were found.")
            cell_types_to_drop.add(ctype)
            continue
        
        batch_labels_array = np.array(batch_labels)
        unique_batches = pd.unique(batch_labels_array)
        batch_counts = pd.Series(batch_labels).value_counts()
        
        if len(unique_batches) < 2 or any(batch_counts < 2):
            if verbose:
                print(f"Deleting '{ctype}' due to insufficient batch diversity or small batch sizes: {batch_counts.to_dict()}.")
            cell_types_to_drop.add(ctype)
            continue
        
        expr_matrix = np.vstack(arrays_for_this_ctype).T
        var_per_gene = np.var(expr_matrix, axis=1)
        zero_var_idx = np.where(var_per_gene == 0)[0]
        if len(zero_var_idx) == expr_matrix.shape[0]:
            if verbose:
                print(f"Deleting '{ctype}' because all genes have zero variance.")
            cell_types_to_drop.add(ctype)
            continue
        
        expr_matrix_sub = np.delete(expr_matrix, zero_var_idx, axis=0)
        expr_df_t = pd.DataFrame(expr_matrix_sub, columns=valid_sample_ids)
        batch_series = pd.Series(batch_labels, index=valid_sample_ids, name='batch')
        
        if expr_df_t.isnull().values.any() and verbose:
            print(f"Warning: NaN values detected in expression data for '{ctype}' before ComBat.")
            
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            corrected_df_sub = pycombat(
                expr_df_t,
                batch=batch_series,
                parametric=parametric
            )
        corrected_values_sub = corrected_df_sub.values
        
        corrected_expr_matrix = expr_matrix.copy()
        corrected_expr_matrix[zero_var_idx, :] = expr_matrix[zero_var_idx, :]
        non_zero_idx = np.delete(np.arange(n_genes), zero_var_idx)
        corrected_expr_matrix[non_zero_idx, :] = corrected_values_sub
        corrected_expr_matrix_t = corrected_expr_matrix.T
        
        for i, sample_id in enumerate(valid_sample_ids):
            corrected_df.at[ctype, sample_id] = pd.Series(corrected_expr_matrix_t[i], index=gene_names)
        if verbose:
            print(f"ComBat correction applied for '{ctype}'.")
    
    if cell_types_to_drop:
        if verbose:
            print(f"Removing {len(cell_types_to_drop)} cell types that failed checks: {cell_types_to_drop}")
        corrected_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
        cell_proportion_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
    
    for idx in corrected_df.index:
        row = corrected_df.loc[idx]
        if any(isinstance(cell, pd.Series) and cell.isnull().any() for cell in row):
            if verbose:
                print(f"Row '{idx}' contains NaNs after ComBat correction. Removing this cell type.")
            cell_types_to_drop.add(idx)
    
    if cell_types_to_drop:
        if verbose:
            print(f"Drop '{cell_types_to_drop}'")
        corrected_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
        cell_proportion_df.drop(labels=cell_types_to_drop, inplace=True, errors='ignore')
    
    save_dataframe_as_strings(corrected_df, pseudobulk_dir, "corrected_expression.csv")
    
    if verbose:
        print("ComBat correction completed.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[combat] Total runtime: {elapsed_time:.2f} seconds")
    return corrected_df

def compute_sparse_mean(X_sparse, indices):
    """Compute mean of sparse matrix for given indices efficiently."""
    if len(indices) == 0:
        return np.zeros(X_sparse.shape[1])
    
    # Subset the sparse matrix
    subset = X_sparse[indices, :]
    
    # Compute mean efficiently
    if issparse(subset):
        # For sparse matrices, sum and divide
        sum_vec = np.asarray(subset.sum(axis=0)).flatten()
        return sum_vec / len(indices)
    else:
        # For dense arrays
        return subset.mean(axis=0)

def validate_sparse_data(X_sparse, verbose=False):
    """Validate sparse matrix data for NaN, Inf, and negative values."""
    if issparse(X_sparse):
        # Check the data attribute of sparse matrix
        data = X_sparse.data
        
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        has_neg = (data < 0).any()
        
        if has_nan or has_inf or has_neg:
            if verbose:
                if has_nan:
                    print("\n\n\n\nWarning: Sparse matrix contains NaN values.\n\n\n\n")
                if has_inf:
                    print("\n\n\n\nWarning: Sparse matrix contains Inf values.\n\n\n\n")
                if has_neg:
                    print("\n\n\n\nWarning: Sparse matrix contains negative values.\n\n\n\n")
                print("\n\n\n\nWarning: Found invalid values in sparse matrix. Cleaning data.\n\n\n\n")
            
            # Clean the data
            data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
            data[data < 0] = 0  # Set negative values to 0
            
            # Create new sparse matrix with cleaned data
            X_sparse = csr_matrix((data, X_sparse.indices, X_sparse.indptr), shape=X_sparse.shape)
        else:
            if verbose:
                print("\n\n\n\nNo NaN, negative, or Inf values found in sparse matrix.\n\n\n\n")
    
    return X_sparse

def compute_gene_variance_sparse(X_sparse):
    """Compute variance for each gene (column) in a sparse matrix efficiently."""
    if issparse(X_sparse):
        n_cells = X_sparse.shape[0]
        
        # Compute mean for each gene
        mean_vec = np.asarray(X_sparse.mean(axis=0)).flatten()
        
        # Compute E[X^2] - (E[X])^2
        # For sparse matrix, we can compute sum of squares efficiently
        X_squared = X_sparse.copy()
        X_squared.data **= 2
        mean_squared = np.asarray(X_squared.mean(axis=0)).flatten()
        
        # Variance = E[X^2] - (E[X])^2
        variance = mean_squared - (mean_vec ** 2)
        
        # Handle numerical errors
        variance[variance < 0] = 0
        
        return variance
    else:
        return np.var(X_sparse, axis=0)

def compute_pseudobulk_dataframes(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './',
    n_features: int = 2000,
    frac: float = 0.3,
    normalize: bool = True,
    target_sum: float = 1e4,
    atac: bool = False,
    verbose: bool = False
    ):
    start_time = time.time() if verbose else None
    batch_correction = True

    # Check if batch correction should be applied
    if batch_col is None or batch_col not in adata.obs.columns:
        if verbose:
            print(f"Column '{batch_col}' not found or is None — skipping batch correction.")
        batch_correction = False
    elif adata.obs[batch_col].isnull().all():
        if verbose:
            print(f"Column '{batch_col}' contains only null values — skipping batch correction.")
        batch_correction = False

    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)
    
    # Allow overwriting to prevent issues with existing files
    keys_to_remove = ['X_DR_expression', 'X_pca_expression', 'X_pca_proportion']
    for key in keys_to_remove:
        if key in adata.uns:
            del adata.uns[key]
            print(f"Removed key '{key}' from adata.uns to prevent overwriting issues.") if verbose else None

    # Only process batch column if batch correction is enabled
    if batch_correction:
        if adata.obs[batch_col].isnull().any():
            if verbose:
                print("Warning: Missing batch labels found. Filling missing values with 'Unknown'.")
            adata.obs[batch_col].fillna("Unknown", inplace=True)

        batch_counts = adata.obs[batch_col].value_counts()
        small_batches = batch_counts[batch_counts < 5]
        if not small_batches.empty and verbose:
            print(f"Warning: The following batches have fewer than 5 samples: {small_batches.to_dict()}. Consider merging these batches.")

    # Keep data as sparse if possible
    X_data = adata.X
    is_sparse = issparse(X_data)

    if is_sparse:
        # Validate and clean sparse matrix
        X_data = validate_sparse_data(X_data, verbose=verbose)
        
        # Compute variance for sparse matrix
        gene_variances = compute_gene_variance_sparse(X_data)
    else:
        # Handle dense matrix as before
        if np.isnan(X_data).any() or (X_data < 0).any() or np.isinf(X_data).any():
            if verbose:
                if np.isnan(X_data).any():
                    print("Warning: X_data contains NaN values.")
                if (X_data < 0).any():
                    print("Warning: X_data contains negative values.")
                if np.isinf(X_data).any():
                    print("Warning: X_data contains Inf values.")
                print("Warning: Found NaN or Inf values in expression data. Replacing with zeros.")
            X_data = np.nan_to_num(X_data, nan=0, posinf=0, neginf=0)
        else:
            if verbose:
                print("No NaN, negative, or Inf values found in X_data.")
        
        gene_variances = np.var(X_data, axis=0)

    nonzero_variance_mask = gene_variances > 0
    if not np.all(nonzero_variance_mask) and verbose:
        print("Warning: Found genes with zero variance. Excluding these genes from analysis.")

    gene_names = adata.var_names[nonzero_variance_mask]

    # Subset data efficiently
    if is_sparse:
        X_data = X_data[:, nonzero_variance_mask]
    else:
        X_data = X_data[:, nonzero_variance_mask]

    samples = adata.obs[sample_col].unique()
    cell_types = adata.obs[celltype_col].unique()

    cell_expression_df = pd.DataFrame(index=cell_types, columns=samples, dtype=object)
    cell_proportion_df = pd.DataFrame(index=cell_types, columns=samples, dtype=float)

    # Create boolean masks for efficient indexing
    sample_masks = {}
    for sample in samples:
        sample_masks[sample] = adata.obs[sample_col] == sample

    # First pass: compute raw expression values
    for sample in samples:
        sample_mask = sample_masks[sample]
        sample_indices = np.where(sample_mask)[0]
        total_cells = len(sample_indices)

        for ctype in cell_types:
            # Get indices for this cell type within the sample
            ctype_mask = sample_mask & (adata.obs[celltype_col] == ctype)
            ctype_indices = np.where(ctype_mask)[0]
            num_cells = len(ctype_indices)

            if num_cells > 0:
                # Compute mean expression efficiently
                if is_sparse:
                    expr_values = compute_sparse_mean(X_data, ctype_indices)
                else:
                    expr_values = X_data[ctype_indices, :].mean(axis=0)
            else:
                expr_values = np.zeros(len(gene_names))
            
            proportion = num_cells / total_cells if total_cells > 0 else 0.0

            # Create expression series (raw values for now)
            expr_series = pd.Series(expr_values, index=gene_names)

            cell_expression_df.at[ctype, sample] = expr_series
            cell_proportion_df.loc[ctype, sample] = proportion

    # Apply normalization
    if normalize:
        if atac:
            # For ATAC-seq: compute IDF weights and apply TF-IDF normalization
            if verbose:
                print("Computing TF-IDF normalization for ATAC-seq data...")
            
            # Compute IDF weights across all samples
            idf_weights = compute_idf_weights(cell_expression_df, verbose=verbose)
            
            # Apply TF-IDF normalization to each sample
            for sample in samples:
                for ctype in cell_types:
                    expr_series = cell_expression_df.at[ctype, sample]
                    if expr_series is not None and len(expr_series) > 0:
                        # Apply TF-IDF normalization
                        tfidf_values = _tfidf_normalize(expr_series.values, idf_weights.values)
                        # Apply log1p transformation after TF-IDF
                        tfidf_values = np.log1p(tfidf_values)
                        cell_expression_df.at[ctype, sample] = pd.Series(tfidf_values, index=expr_series.index)
            
            if verbose:
                print("TF-IDF normalization and log1p transformation completed.")
        else:
            # For RNA-seq: apply standard normalization and log1p
            if verbose:
                print(f"Applying standard normalization (target_sum={target_sum}) and log1p transformation...")
            
            for sample in samples:
                for ctype in cell_types:
                    expr_series = cell_expression_df.at[ctype, sample]
                    if expr_series is not None and len(expr_series) > 0:
                        # Apply standard normalization and log1p
                        norm_values = _normalize_and_log(expr_series.values, target_sum=target_sum)
                        cell_expression_df.at[ctype, sample] = pd.Series(norm_values, index=expr_series.index)
            
            if verbose:
                print("Standard normalization and log1p transformation completed.")

    if verbose:
        print("Successfully computed pseudobulk data.")

    # Save raw normalized expression data before batch correction
    if normalize:
        norm_type = "tfidf" if atac else "standard"
        save_dataframe_as_strings(cell_expression_df, pseudobulk_dir, f"{norm_type}_normalized_expression_raw.csv", verbose=verbose)

    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[pseudobulk] Total runtime: {elapsed_time:.2f} seconds")

    f = io.StringIO()
    if batch_correction:
        if verbose:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                cell_expression_corrected_df = combat_correct_cell_expressions(
                    adata, cell_expression_df, cell_proportion_df, pseudobulk_dir, 
                    batch_col=batch_col, sample_col=sample_col, verbose=verbose
                )
            print("ComBat correction completed successfully.")
        else:
            # Suppress warnings and redirect output to a StringIO object
            with warnings.catch_warnings(), contextlib.redirect_stdout(f):
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                cell_expression_corrected_df = combat_correct_cell_expressions(
                    adata, cell_expression_df, cell_proportion_df, pseudobulk_dir, 
                    batch_col=batch_col, sample_col=sample_col, verbose=verbose
                )
            print("ComBat correction completed successfully.")
    else:
        if verbose:
            print("Skipping ComBat correction as batch_col is None or not found.")
        cell_expression_corrected_df = cell_expression_df.copy(deep=True)

    cell_expression_corrected_df = highly_variable_gene_selection(cell_expression_corrected_df, n_features)

    cell_expression_corrected_df, pseudobulk_adata = select_hvf_loess(
        cell_expression_corrected_df, n_features=n_features, verbose=verbose
    )

    proportion_df = cell_proportion_df.T

    pseudobulk = {
        "cell_expression": cell_expression_df,
        "cell_proportion": proportion_df,
        "cell_expression_corrected": cell_expression_corrected_df,
        "normalized": normalize,
        "normalization_type": "tfidf" if (normalize and atac) else "standard" if normalize else None,
        "target_sum": target_sum if (normalize and not atac) else None
    }

    save_dataframe_as_strings(cell_expression_corrected_df, pseudobulk_dir, "expression.csv")
    save_dataframe_as_strings(cell_proportion_df, pseudobulk_dir, "proportion.csv")

    sc.write(os.path.join(pseudobulk_dir, "pseudobulk_adata.h5ad"), pseudobulk_adata)
    return pseudobulk, pseudobulk_adata