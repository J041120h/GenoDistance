import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Visualization import visualization_harmony
from combat.pycombat import pycombat

def contains_nan_in_lists(df: pd.DataFrame) -> bool:
    for row in df.itertuples(index=False):
        for array in row:
            if isinstance(array, np.ndarray) and np.isnan(array).any():
                return True
    return False

import numpy as np
import pandas as pd
from combat.pycombat import pycombat
import os

def save_dataframe_as_strings(df: pd.DataFrame, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    df.astype(str).to_csv(file_path, index=True)
    print(f"DataFrame saved as strings to {file_path}")

def combat_correct_cell_expressions(
    adata: sc.AnnData,
    cell_expression_df: pd.DataFrame,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    parametric: bool = True
) -> pd.DataFrame:
    sample_batch_map = (
        adata.obs[[sample_col, batch_col]]
        .drop_duplicates()
        .set_index(sample_col)[batch_col]
        .to_dict()
    )
    example_array = next(
        (arr for arr in cell_expression_df.iloc[0].dropna() if arr is not None and len(arr) > 0),
        None
    )
    if example_array is None:
        raise ValueError("No valid arrays found in cell_expression_df.")
    n_genes = len(example_array)
    corrected_df = cell_expression_df.copy(deep=True)
    for ctype in corrected_df.index:
        row_data = corrected_df.loc[ctype]
        batch_labels = []
        arrays_for_this_ctype = []
        for sample_id in row_data.index:
            expr_array = row_data[sample_id]
            if expr_array is None or len(expr_array) == 0:
                expr_array = np.zeros(n_genes)
            else:
                expr_array = np.array(expr_array, dtype=float)
            expr_array = np.nan_to_num(expr_array, nan=1e-308, posinf=1e-308, neginf=1e-308)
            arrays_for_this_ctype.append(expr_array)
            batch_labels.append(sample_batch_map.get(sample_id, "missing_batch"))
        batch_labels_array = np.array(batch_labels)
        unique_batches = pd.unique(batch_labels_array)
        batch_counts = pd.Series(batch_labels).value_counts()
        if len(unique_batches) < 2 or any(batch_counts < 2):
            print(f"Skipping ComBat for '{ctype}' due to insufficient batch diversity or small batch sizes: {batch_counts.to_dict()}.")
            continue
        expr_matrix = np.vstack(arrays_for_this_ctype).T
        if np.any(expr_matrix < 0):
            print(f"Warning: Negative values detected in '{ctype}' expression matrix.")
        var_per_gene = np.var(expr_matrix, axis=1)
        zero_var_idx = np.where(var_per_gene == 0)[0]
        if len(zero_var_idx) == expr_matrix.shape[0]:
            print(f"Skipping ComBat for '{ctype}' because all genes have zero variance.")
            continue
        expr_matrix_sub = np.delete(expr_matrix, zero_var_idx, axis=0)
        expr_df_t = pd.DataFrame(expr_matrix_sub, columns=row_data.index)
        batch_series = pd.Series(batch_labels, index=row_data.index, name='batch')
        if expr_df_t.isnull().values.any():
            print(f"Warning: NaN values detected in expression data for '{ctype}' before ComBat.")
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
        for i, sample_id in enumerate(row_data.index):
            corrected_df.loc[ctype, sample_id] = corrected_expr_matrix_t[i]
        print(f"ComBat correction applied for '{ctype}'.")
    if contains_nan_in_lists(corrected_df):
        print("\n\n\n\nWarning: NaN values detected even after correction. Returning uncorrected data.\n\n\n\n")
        return cell_expression_df
    else:
        print("\n\n\n\nNo NaN values detected, Good to continue.\n\n\n\n")
    print("ComBat correction completed.")
    return corrected_df

def compute_pseudobulk_dataframes(
    adata: sc.AnnData,
    batch_col: str = 'batch',
    sample_col: str = 'sample',
    celltype_col: str = 'cell_type',
    output_dir: str = './'
):
    pseudobulk_dir = os.path.join(output_dir, "pseudobulk")
    os.makedirs(pseudobulk_dir, exist_ok=True)

    if adata.obs[batch_col].isnull().any():
        print("\n\n\n\nWarning: Missing batch labels found. Filling missing values with 'Unknown'.\n\n\n\n")
        adata.obs[batch_col].fillna("Unknown", inplace=True)
    batch_counts = adata.obs[batch_col].value_counts()
    small_batches = batch_counts[batch_counts < 5]
    if not small_batches.empty:
        print(f"\n\n\n\nWarning: The following batches have fewer than 5 samples: {small_batches.to_dict()}. Consider merging these batches.\n\n\n\n")

    X_data = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
    if np.any(np.isnan(X_data)) or np.any(np.isinf(X_data)):
        print("\n\n\n\nWarning: Found NaN or Inf values in expression data. Replacing with zeros.\n\n\n\n")
        X_data = np.nan_to_num(X_data, nan=0.0, posinf=0.0, neginf=0.0)
    gene_variances = np.var(X_data, axis=0)
    nonzero_variance_mask = gene_variances > 0
    if not np.all(nonzero_variance_mask):
        print("\n\n\n\nWarning: Found genes with zero variance. Excluding these genes from analysis.\n\n\n\n")
    gene_names = adata.var_names[nonzero_variance_mask]
    X_data = X_data[:, nonzero_variance_mask]
    samples = adata.obs[sample_col].unique()
    cell_types = adata.obs[celltype_col].unique()
    cell_expression_df = pd.DataFrame(index=cell_types, columns=samples, dtype=object)
    cell_proportion_df = pd.DataFrame(index=cell_types, columns=samples, dtype=float)
    for sample in samples:
        sample_mask = adata.obs[sample_col] == sample
        total_cells = np.sum(sample_mask)
        for ctype in cell_types:
            ctype_mask = sample_mask & (adata.obs[celltype_col] == ctype)
            num_cells = np.sum(ctype_mask)
            expr_values = X_data[ctype_mask, :].mean(axis=0) if num_cells > 0 else np.zeros(len(gene_names))
            proportion = num_cells / total_cells if total_cells > 0 else 0.0
            cell_expression_df.loc[ctype, sample] = expr_values
            cell_proportion_df.loc[ctype, sample] = proportion
    print("\n\n\n\nSuccessfully computed pseudobulk data.\n\n\n\n")
    
    cell_expression_corrected_df = combat_correct_cell_expressions(adata, cell_expression_df)
    save_dataframe_as_strings(cell_expression_df, pseudobulk_dir, "expression.csv")
    save_dataframe_as_strings(cell_proportion_df, pseudobulk_dir, "proportion.csv")
    save_dataframe_as_strings(cell_expression_corrected_df, pseudobulk_dir, "corrected_expression.csv")
    pseudobulk = {
        "cell_expression": cell_expression_df,
        "cell_proportion": cell_proportion_df,
        "cell_expression_corrected": cell_expression_corrected_df
    }
    return pseudobulk
