import anndata as ad
import pandas as pd
from sklearn.cross_decomposition import CCA
import numpy as np
import os
import matplotlib.pyplot as plt
from anndata import AnnData
import time
from DR import process_anndata_with_pca
from ATAC_cell_type import cell_types_atac, cell_type_assign_atac
from pseudo_adata import compute_pseudobulk_adata
from CCA import *
from CCA_test import *
import warnings
from sklearn.preprocessing import StandardScaler

def generate_null_distribution_atac(pseudobulk_adata, column, sev_col,
                                   n_permutations=1000, n_pcs=None,
                                   save_path=None, verbose=True):
    """
    Generate null distribution for ATAC data using permutation testing.
    
    Parameters:
    -----------
    pseudobulk_adata : AnnData
        Pseudobulk AnnData object
    column : str
        Column name for dimension reduction coordinates
    sev_col : str
        Column name for severity levels
    n_permutations : int
        Number of permutations
    n_pcs : int, optional
        Number of PC dimensions to use. If None, uses all available PCs.
    save_path : str, optional
        Path to save null distribution
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    np.array
        Null distribution of CCA scores
    """
    # Check if column exists
    if column not in pseudobulk_adata.uns:
        raise ValueError(f"Column '{column}' not found in pseudobulk_adata.uns")
    
    # Get DR coordinates and severity levels
    dr_coords_full = pseudobulk_adata.uns[column].copy()
    sev_levels = pseudobulk_adata.obs[sev_col].values
    
    # Use specified number of DR components
    if n_pcs is None:
        dr_coords = dr_coords_full
        n_dims_used = dr_coords_full.shape[1]
    else:
        n_pcs = min(n_pcs, dr_coords_full.shape[1])
        dr_coords = dr_coords_full.iloc[:, :n_pcs]
        n_dims_used = n_pcs
    
    if len(dr_coords) < 3:
        raise ValueError(f"Insufficient samples: {len(dr_coords)}")
    if len(np.unique(sev_levels)) < 2:
        raise ValueError("Insufficient severity level variance")
    
    # Prepare data for CCA
    X = dr_coords.values
    y_original = sev_levels.copy()
    
    # Run permutations
    null_scores = []
    failed_permutations = 0
    
    for perm in range(n_permutations):
        try:
            # Randomly shuffle severity labels
            permuted_sev = np.random.permutation(y_original)
            
            # Run CCA
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_permuted_scaled = scaler_y.fit_transform(permuted_sev.reshape(-1, 1))
            
            # Fit CCA with 1 component
            cca_perm = CCA(n_components=1, max_iter=1000, tol=1e-6)
            cca_perm.fit(X_scaled, y_permuted_scaled)
            
            # Transform and compute correlation
            X_c_perm, y_c_perm = cca_perm.transform(X_scaled, y_permuted_scaled)
            perm_correlation = np.corrcoef(X_c_perm[:, 0], y_c_perm[:, 0])[0, 1]
            
            # Record the CCA score
            if np.isnan(perm_correlation) or np.isinf(perm_correlation):
                null_scores.append(0.0)
                failed_permutations += 1
            else:
                null_scores.append(abs(perm_correlation))
                
        except Exception:
            null_scores.append(0.0)
            failed_permutations += 1
    
    null_distribution = np.array(null_scores)
    
    if verbose:
        success_rate = (n_permutations - failed_permutations) / n_permutations * 100
        print(f"Null distribution generated using {n_dims_used} PC dimensions: {success_rate:.1f}% success rate")
    
    if save_path:
        np.save(save_path, null_distribution)
    
    return null_distribution

def generate_corrected_null_distribution_atac(all_resolution_results, n_permutations=1000):
    """
    Generate corrected null distribution accounting for resolution selection bias.
    
    Parameters:
    -----------
    all_resolution_results : list
        List of dictionaries containing results from all resolutions
    n_permutations : int
        Number of permutations
        
    Returns:
    --------
    np.array
        Corrected null distribution
    """
    corrected_null_scores = []
    
    for perm_idx in range(n_permutations):
        # Collect the CCA score from this permutation across all resolutions
        perm_scores_across_resolutions = []
        
        for resolution_result in all_resolution_results:
            if 'null_scores' in resolution_result and resolution_result['null_scores'] is not None:
                if len(resolution_result['null_scores']) > perm_idx:
                    perm_scores_across_resolutions.append(resolution_result['null_scores'][perm_idx])
        
        # Select the maximum score (mimicking optimal resolution selection)
        if perm_scores_across_resolutions:
            max_score_for_this_perm = max(perm_scores_across_resolutions)
            corrected_null_scores.append(max_score_for_this_perm)
    
    return np.array(corrected_null_scores)

def compute_corrected_pvalues_atac(df_results, corrected_null_distribution, output_dir, column):
    """
    Compute corrected p-values for all CCA scores and create visualization plots.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Results dataframe with CCA scores
    corrected_null_distribution : np.array
        Corrected null distribution
    output_dir : str
        Output directory for plots
    column : str
        Column name (e.g., 'X_DR_expression' or 'X_DR_proportion')
    """
    # Create p-value directory
    pvalue_dir = os.path.join(output_dir, "corrected_p_values")
    os.makedirs(pvalue_dir, exist_ok=True)
    
    # Add corrected p-value column
    df_results['corrected_pvalue'] = np.nan
    
    # Compute corrected p-values for each resolution
    for idx, row in df_results.iterrows():
        resolution = row['resolution']
        cca_score = row['cca_score']
        
        if not np.isnan(cca_score):
            # Compute corrected p-value
            corrected_p_value = np.mean(corrected_null_distribution >= cca_score)
            df_results.loc[idx, 'corrected_pvalue'] = corrected_p_value
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            
            # Plot histogram of null distribution
            plt.hist(corrected_null_distribution, bins=50, alpha=0.7, color='lightblue', 
                    density=True, label='Corrected Null Distribution')
            
            # Plot vertical line for observed CCA score
            plt.axvline(cca_score, color='red', linestyle='--', linewidth=2, 
                       label=f'Observed CCA Score: {cca_score:.4f}')
            
            # Add p-value text
            plt.text(0.05, 0.95, f'Corrected p-value: {corrected_p_value:.4f}', 
                    transform=plt.gca().transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Formatting
            plt.xlabel('CCA Score')
            plt.ylabel('Density')
            plt.title(f'Corrected P-value Analysis\nResolution: {resolution:.3f}, {column}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_filename = f'corrected_pvalue_res_{resolution:.3f}.png'
            plot_path = os.path.join(pvalue_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    return df_results

def create_comprehensive_summary_atac(df_results, best_resolution, column, output_dir, 
                                     has_corrected_pvalues=False):
    """
    Create comprehensive summary visualizations and reports for ATAC resolution optimization.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Results dataframe
    best_resolution : float
        Optimal resolution found
    column : str
        Column name for DR type
    output_dir : str
        Output directory
    has_corrected_pvalues : bool
        Whether corrected p-values are available
    """
    # Create summary directory
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Sort results by resolution
    df_sorted = df_results.sort_values('resolution').copy()
    
    # Create main visualization
    if has_corrected_pvalues:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot CCA scores
    valid_df = df_sorted[~df_sorted['cca_score'].isna()]
    
    # Color by pass type
    coarse_df = valid_df[valid_df['pass'] == 'coarse']
    fine_df = valid_df[valid_df['pass'] == 'fine']
    
    # Plot coarse and fine results
    ax1.scatter(coarse_df['resolution'], coarse_df['cca_score'], 
                color='blue', s=80, alpha=0.6, label='Coarse Search', zorder=2)
    ax1.scatter(fine_df['resolution'], fine_df['cca_score'], 
                color='green', s=60, alpha=0.8, label='Fine Search', zorder=3)
    
    # Connect points with lines
    ax1.plot(valid_df['resolution'], valid_df['cca_score'], 
             'k-', linewidth=1, alpha=0.4, zorder=1)
    
    # Highlight best resolution
    ax1.axvline(x=best_resolution, color='red', linestyle='--', linewidth=2,
                label=f'Best Resolution: {best_resolution:.3f}', zorder=4)
    
    # Add best score annotation
    best_score = valid_df.loc[valid_df['resolution'] == best_resolution, 'cca_score'].iloc[0]
    ax1.annotate(f'Best Score: {best_score:.4f}', 
                 xy=(best_resolution, best_score),
                 xytext=(best_resolution + 0.05, best_score + 0.01),
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                 fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    ax1.set_xlabel('Resolution', fontsize=12)
    ax1.set_ylabel('CCA Score', fontsize=12)
    ax1.set_title(f'ATAC Resolution Optimization: {column}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot corrected p-values if available
    if has_corrected_pvalues:
        valid_pval_df = valid_df[~valid_df['corrected_pvalue'].isna()]
        
        coarse_pval = valid_pval_df[valid_pval_df['pass'] == 'coarse']
        fine_pval = valid_pval_df[valid_pval_df['pass'] == 'fine']
        
        ax2.scatter(coarse_pval['resolution'], coarse_pval['corrected_pvalue'], 
                    color='blue', s=80, alpha=0.6, label='Coarse Search', zorder=2)
        ax2.scatter(fine_pval['resolution'], fine_pval['corrected_pvalue'], 
                    color='green', s=60, alpha=0.8, label='Fine Search', zorder=3)
        
        ax2.plot(valid_pval_df['resolution'], valid_pval_df['corrected_pvalue'], 
                 'k-', linewidth=1, alpha=0.4, zorder=1)
        
        ax2.axvline(x=best_resolution, color='red', linestyle='--', linewidth=2, zorder=4)
        ax2.axhline(y=0.05, color='orange', linestyle=':', linewidth=2,
                    label='p=0.05 threshold', zorder=4)
        
        # Add best p-value annotation
        best_pval = valid_pval_df.loc[valid_pval_df['resolution'] == best_resolution, 'corrected_pvalue'].iloc[0]
        ax2.annotate(f'p={best_pval:.4f}', 
                     xy=(best_resolution, best_pval),
                     xytext=(best_resolution + 0.05, best_pval + 0.05),
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                     fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        ax2.set_xlabel('Resolution', fontsize=12)
        ax2.set_ylabel('Corrected P-value', fontsize=12)
        ax2.set_title('Corrected P-values (Accounting for Resolution Selection)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    
    # Save main plot
    plot_path = os.path.join(summary_dir, f'resolution_optimization_summary_{column}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved to: {plot_path}")
    
    # Create text summary
    summary_path = os.path.join(summary_dir, f'optimization_results_{column}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"ATAC Resolution Optimization Results: {column}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best Resolution: {best_resolution:.3f}\n")
        f.write(f"Best CCA Score: {best_score:.4f}\n")
        
        if has_corrected_pvalues:
            f.write(f"Corrected P-value at Best Resolution: {best_pval:.4f}\n")
        
        f.write(f"\nTotal Resolutions Tested: {len(valid_df)}\n")
        f.write(f"  - Coarse Search: {len(coarse_df)} resolutions\n")
        f.write(f"  - Fine Search: {len(fine_df)} resolutions\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("All Results (sorted by resolution):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Resolution':<12} {'CCA Score':<12} {'Pass Type':<12}")
        if has_corrected_pvalues:
            f.write(f" {'Corrected P-value':<18}")
        f.write("\n")
        
        for _, row in valid_df.iterrows():
            f.write(f"{row['resolution']:<12.3f} {row['cca_score']:<12.4f} {row['pass']:<12}")
            if has_corrected_pvalues and 'corrected_pvalue' in row:
                pval = row['corrected_pvalue']
                pval_str = f"{pval:.4f}" if not np.isnan(pval) else "N/A"
                f.write(f" {pval_str:<18}")
            f.write("\n")
        
        # Add summary statistics
        f.write("\n" + "-" * 80 + "\n")
        f.write("Summary Statistics:\n")
        f.write(f"CCA Score Range: [{valid_df['cca_score'].min():.4f}, {valid_df['cca_score'].max():.4f}]\n")
        f.write(f"Mean CCA Score: {valid_df['cca_score'].mean():.4f} ± {valid_df['cca_score'].std():.4f}\n")
        
        if has_corrected_pvalues:
            valid_pvals = valid_df['corrected_pvalue'].dropna()
            if len(valid_pvals) > 0:
                f.write(f"\nCorrected P-value Statistics:\n")
                f.write(f"Min P-value: {valid_pvals.min():.4f}\n")
                f.write(f"Resolutions with p < 0.05: {(valid_pvals < 0.05).sum()}\n")
                f.write(f"Resolutions with p < 0.01: {(valid_pvals < 0.01).sum()}\n")
    
    print(f"Summary report saved to: {summary_path}")
    
    # Save detailed results CSV
    detailed_csv_path = os.path.join(summary_dir, f'detailed_results_{column}.csv')
    df_sorted.to_csv(detailed_csv_path, index=False)
    print(f"Detailed results saved to: {detailed_csv_path}")

def cca_analysis(pseudobulk_adata, column, sev_col, n_components=2, n_pcs=None):
    """
    CCA analysis for ATAC data with case-insensitive index matching.
    
    Parameters:
    -----------
    pseudobulk_adata : sc.AnnData
        AnnData object containing dimension reduction results
    column : str
        Column name for dimension reduction coordinates
    sev_col : str
        Column name for severity levels
    n_components : int, default 2
        Number of CCA components
    n_pcs : int, optional
        Number of PCs to use for CCA analysis. If None, uses all available PCs.
        
    Returns:
    --------
    dict with CCA results including weight vectors
    """
    
    result = {
        'cca_score': np.nan,
        'n_samples': 0,
        'n_features': 0,
        'column': column,
        'valid': False,
        'error_message': None,
        'X_weights': None,
        'Y_weights': None,
        'X_loadings': None,
        'Y_loadings': None,
        'n_pcs_used': 0
    }
    
    try:
        # Check if required columns exist
        if column not in pseudobulk_adata.uns:
            result['error_message'] = f"Column '{column}' not found in uns"
            return result
        if sev_col not in pseudobulk_adata.obs.columns:
            result['error_message'] = f"Column '{sev_col}' not found in obs"
            return result
        
        # Extract dimension reduction coordinates
        dr_coords_full = pseudobulk_adata.uns[column].copy()
        
        # Convert indices to lowercase for case-insensitive matching
        dr_coords_full.index = dr_coords_full.index.str.lower()
        
        # Create a mapping from lowercase to original obs indices
        obs_index_map = dict(zip(pseudobulk_adata.obs.index.str.lower(), pseudobulk_adata.obs.index))
        
        # Find common indices using lowercase
        dr_indices_lower = dr_coords_full.index
        obs_indices_lower = pseudobulk_adata.obs.index.str.lower()
        common_indices_lower = dr_indices_lower.intersection(obs_indices_lower)
        
        if len(common_indices_lower) == 0:
            result['error_message'] = f"No matching samples between DR results and observations. DR has {len(dr_indices_lower)} samples, obs has {len(obs_indices_lower)} samples."
            return result
        
        # Filter DR coordinates to common indices
        dr_coords_full = dr_coords_full.loc[common_indices_lower]
        
        # Get severity levels using the original obs indices
        original_indices = [obs_index_map[idx] for idx in common_indices_lower]
        sev_levels = pseudobulk_adata.obs.loc[original_indices, sev_col].values
        
        # Determine number of PCs to use
        max_pcs = dr_coords_full.shape[1]
        if n_pcs is None:
            n_pcs_to_use = max_pcs
        else:
            n_pcs_to_use = min(n_pcs, max_pcs)
        
        # Use only the specified number of PCs
        dr_coords = dr_coords_full.iloc[:, :n_pcs_to_use]
        
        # Basic validation
        if len(dr_coords) < 3:
            result['error_message'] = f"Insufficient samples: {len(dr_coords)}"
            return result
        if len(np.unique(sev_levels)) < 2:
            result['error_message'] = "Insufficient severity level variance"
            return result
        
        # Prepare data for CCA
        X = dr_coords.values
        y = sev_levels.reshape(-1, 1)
        
        result['n_samples'] = len(X)
        result['n_features'] = X.shape[1]
        result['n_pcs_used'] = n_pcs_to_use
        
        # Limit components
        max_components = min(X.shape[1], y.shape[1], X.shape[0] - 1)
        n_components_actual = min(n_components, max_components)
        
        if n_components_actual < 1:
            result['error_message'] = "Cannot compute CCA components"
            return result
        
        # Standardize data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Check for constant variance
        if np.var(X_scaled) < 1e-10 or np.var(y_scaled) < 1e-10:
            result['error_message'] = "Near-zero variance in standardized data"
            return result
        
        # Fit CCA
        try:
            cca = CCA(n_components=n_components_actual, max_iter=1000, tol=1e-6)
            cca.fit(X_scaled, y_scaled)
            
            # Transform and compute correlation
            X_c, y_c = cca.transform(X_scaled, y_scaled)
            
            if X_c.shape[0] > 1:
                correlation = np.corrcoef(X_c[:, 0], y_c[:, 0])[0, 1]
                cca_score = abs(correlation)
                
                if np.isnan(cca_score):
                    result['error_message'] = "CCA produced NaN correlation"
                    return result
            else:
                result['error_message'] = "Insufficient samples for correlation"
                return result
            
            # Store results
            result['cca_score'] = cca_score
            result['valid'] = True
            
            # Store weight vectors
            result['X_weights'] = cca.x_weights_.flatten()
            result['Y_weights'] = cca.y_weights_.flatten()
            
            # Store loadings if available
            result['X_loadings'] = cca.x_loadings_ if hasattr(cca, 'x_loadings_') else None
            result['Y_loadings'] = cca.y_loadings_ if hasattr(cca, 'y_loadings_') else None
            
        except Exception as e:
            result['error_message'] = f"CCA computation failed: {str(e)}"
            return result
        
    except Exception as e:
        result['error_message'] = f"CCA analysis failed: {str(e)}"
    
    return result

def find_optimal_cell_resolution_atac(
    AnnData_cell: AnnData,
    AnnData_sample: AnnData,
    output_dir: str,
    column: str,
    n_features: int = 40000,
    sev_col: str = "sev.level",
    batch_col: str = None,
    sample_col: str = "sample",
    use_rep: str = 'X_DM_harmony',
    num_DR_components: int = 30,
    num_DMs: int = 20,
    num_pvalue_simulations: int = 1000,
    n_pcs: int = 10,  # Renamed from n_pcs_for_null to match integration version
    compute_corrected_pvalues: bool = True,
    verbose: bool = True
) -> tuple:
    """
    Find optimal clustering resolution by maximizing CCA correlation between 
    dimension reduction and severity levels for ATAC data, with corrected p-value tracking.
    
    Parameters:
    -----------
    AnnData_cell : AnnData
        Cell-level AnnData object (ATAC)
    AnnData_sample : AnnData  
        Sample-level AnnData object (ATAC)
    output_dir : str
        Output directory for results
    column : str
        Column name in adata.uns for dimension reduction results (e.g., 'X_DR_expression' or 'X_DR_proportion')
    n_features : int
        Number of features for pseudobulk
    sev_col : str
        Column name for severity levels
    batch_col : str
        Column name for batch information
    sample_col : str
        Column name for sample identifiers
    use_rep : str
        Representation to use for neighborhood graph
    num_DR_components : int
        Number of dimension reduction components
    num_DMs : int
        Number of diffusion map components
    num_pvalue_simulations : int
        Number of simulations for null distribution (default: 1000)
    n_pcs : int
        Number of PCs to use for CCA analysis (default: 10)
    compute_corrected_pvalues : bool
        Whether to compute corrected p-values (default: True)
    verbose : bool
        Whether to print verbose output (default: True)
        
    Returns:
    --------
    tuple: (optimal_resolution, results_dataframe)
    """
    start_time = time.time()
    
    # Extract DR type from column name for clearer output
    dr_type = column.replace('X_DR_', '') if column.startswith('X_DR_') else column
    
    # Create main output directory structure
    main_output_dir = os.path.join(output_dir, f"ATAC_resolution_optimization_{dr_type}")
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Create subdirectories
    resolutions_dir = os.path.join(main_output_dir, "resolutions")
    os.makedirs(resolutions_dir, exist_ok=True)

    print(f"Starting ATAC resolution optimization for {column}...")
    print(f"Using representation: {use_rep} with {num_DMs} components")
    print(f"Using {n_pcs} PCs for CCA analysis")
    print(f"Testing resolutions from 0.01 to 1.00...")
    if compute_corrected_pvalues:
        print(f"Will compute corrected p-values with {num_pvalue_simulations} simulations per resolution")

    # Storage for all results
    all_results = []
    all_resolution_null_results = []

    # First pass: coarse search
    print("\n=== FIRST PASS: Coarse Search ===")
    for resolution in np.arange(0.1, 1.01, 0.1):
        print(f"\n\nTesting resolution: {resolution:.2f}\n")
        
        # Create resolution-specific directory
        resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.2f}")
        os.makedirs(resolution_dir, exist_ok=True)
        
        result_dict = {
            'resolution': resolution,
            'cca_score': np.nan,
            'p_value': np.nan,
            'corrected_pvalue': np.nan,
            'pass': 'coarse',
            'n_clusters': 0,
            'n_samples': 0,
            'n_features': 0,
            'n_pcs_used': n_pcs
        }
        
        # Initialize null results for this resolution
        resolution_null_result = {
            'resolution': resolution,
            'null_scores': None
        }
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_cell.obs.columns:
                AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if 'cell_type' in AnnData_sample.obs.columns:
                AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Perform clustering using ATAC-specific function
            AnnData_sample = cell_types_atac(
                AnnData_sample,
                cell_column='cell_type',
                Save=False,
                output_dir=resolution_dir,
                cluster_resolution=resolution,
                use_rep=use_rep,
                peaks=None,
                method='average',
                metric='euclidean',
                distance_mode='centroid',
                num_DMs=num_DMs,
                verbose=False
            )
            
            n_clusters = AnnData_sample.obs['cell_type'].nunique()
            result_dict['n_clusters'] = n_clusters
            print(f"Number of clusters: {n_clusters}")
            
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_sample, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features, 
                output_dir=resolution_dir,
                Save=False,
                verbose=False
            )
            
            result_dict['n_samples'] = len(pseudobulk_adata)
            
            # Perform dimension reduction
            process_anndata_with_pca(
                adata=AnnData_sample,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=True,
                output_dir=resolution_dir,
                not_save=True,
                verbose=False
            )

            # Check if column exists in pseudobulk_adata.uns (not AnnData_sample.uns)
            if column not in pseudobulk_adata.uns:
                print(f"Warning: {column} not found in pseudobulk_adata.uns. Skipping resolution {resolution:.2f}")
                all_results.append(result_dict)
                all_resolution_null_results.append(resolution_null_result)
                continue
            
            # Run CCA analysis using the ATAC-specific cca_analysis function
            # This allows us to specify n_pcs for the actual CCA analysis
            cca_result = cca_analysis(
                pseudobulk_adata=pseudobulk_adata,
                column=column,
                sev_col=sev_col,
                n_components=1,  # Always use 1 component for univariate severity
                n_pcs=n_pcs  # Use specified number of PCs for CCA
            )
            
            if cca_result['valid']:
                result_dict['cca_score'] = cca_result['cca_score']
                result_dict['n_features'] = cca_result['n_features']
                result_dict['n_pcs_used'] = cca_result['n_pcs_used']
                print(f"Resolution {resolution:.2f}: CCA Score = {cca_result['cca_score']:.4f} (using {cca_result['n_pcs_used']} PCs)")
                
                # Create CCA visualization plot
                try:
                    # Extract 2D visualization data for plotting
                    (pca_coords_2d, 
                     sev_levels, 
                     cca_model, 
                     cca_score, 
                     samples) = run_cca_on_2d_pca_from_adata(
                        adata=pseudobulk_adata,
                        column=column,
                        sev_col=sev_col
                    )
                    
                    # Save CCA plot
                    plot_path = os.path.join(resolution_dir, f"cca_plot_res_{resolution:.2f}.png")
                    plot_cca_on_2d_pca(
                        pca_coords_2d=pca_coords_2d,
                        sev_levels=sev_levels,
                        cca=cca_model,
                        output_path=plot_path,
                        sample_labels=None,
                        title_suffix=f"Resolution {resolution:.2f}"
                    )
                    if verbose:
                        print(f"Created CCA visualization plot")
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to create CCA visualization: {str(e)}")
                
                # Generate null distribution if computing corrected p-values
                if compute_corrected_pvalues:
                    try:
                        null_distribution = generate_null_distribution_atac(
                            pseudobulk_adata=pseudobulk_adata,
                            column=column,
                            sev_col=sev_col,
                            n_pcs=n_pcs,  # Use same number of PCs for null distribution
                            n_permutations=num_pvalue_simulations,
                            save_path=os.path.join(resolution_dir, f'null_dist_{resolution:.2f}.npy'),
                            verbose=verbose
                        )
                        resolution_null_result['null_scores'] = null_distribution
                        
                        # Compute standard p-value for this resolution
                        p_value = np.mean(null_distribution >= cca_result['cca_score'])
                        result_dict['p_value'] = p_value
                        print(f"Resolution {resolution:.2f}: Standard p-value = {p_value:.4f}")
                        
                    except Exception as e:
                        print(f"Warning: Failed to generate null distribution: {str(e)}")
            else:
                print(f"Error in CCA analysis at resolution {resolution:.2f}: {cca_result['error_message']}")
                
        except Exception as e:
            print(f"Error at resolution {resolution:.2f}: {str(e)}")
        
        all_results.append(result_dict)
        all_resolution_null_results.append(resolution_null_result)

    # Find best resolution from first pass
    coarse_results = [r for r in all_results if not np.isnan(r['cca_score'])]
    if not coarse_results:
        raise ValueError("No valid CCA scores obtained in coarse search. Check your ATAC data and parameters.")
    
    best_coarse = max(coarse_results, key=lambda x: x['cca_score'])
    best_resolution = best_coarse['resolution']
    print(f"\nBest resolution from first pass: {best_resolution:.2f}")
    print(f"Best CCA score: {best_coarse['cca_score']:.4f}")

    # Second pass: fine-tuned search
    print("\n=== SECOND PASS: Fine-tuned Search ===")
    search_range_start = max(0.01, best_resolution - 0.05)
    search_range_end = min(1.00, best_resolution + 0.05)
    
    print(f"Fine-tuning search from {search_range_start:.2f} to {search_range_end:.2f}...")

    for resolution in np.arange(search_range_start, search_range_end + 0.001, 0.01):
        resolution = round(resolution, 3)
        
        # Skip if already tested in coarse search
        if any(abs(r['resolution'] - resolution) < 0.001 for r in all_results):
            continue
        
        print(f"\nTesting fine-tuned resolution: {resolution:.3f}")
        
        # Create resolution-specific directory
        resolution_dir = os.path.join(resolutions_dir, f"resolution_{resolution:.3f}")
        os.makedirs(resolution_dir, exist_ok=True)
        
        result_dict = {
            'resolution': resolution,
            'cca_score': np.nan,
            'p_value': np.nan,
            'corrected_pvalue': np.nan,
            'pass': 'fine',
            'n_clusters': 0,
            'n_samples': 0,
            'n_features': 0,
            'n_pcs_used': n_pcs
        }
        
        # Initialize null results for this resolution
        resolution_null_result = {
            'resolution': resolution,
            'null_scores': None
        }
        
        try:
            # Clean up previous cell type assignments
            if 'cell_type' in AnnData_cell.obs.columns:
                AnnData_cell.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            if 'cell_type' in AnnData_sample.obs.columns:
                AnnData_sample.obs.drop(columns=['cell_type'], inplace=True, errors='ignore')
            
            # Perform clustering
            AnnData_sample = cell_types_atac(
                AnnData_sample,
                cell_column='cell_type',
                Save=False,
                output_dir=resolution_dir,
                cluster_resolution=resolution,
                use_rep=use_rep,
                peaks=None,
                method='average',
                metric='euclidean',
                distance_mode='centroid',
                num_DMs=num_DMs,
                verbose=False
            )
            
            # Record number of clusters
            n_clusters = AnnData_sample.obs['cell_type'].nunique()
            result_dict['n_clusters'] = n_clusters
            
            # Compute pseudobulk data
            pseudobulk_dict, pseudobulk_adata = compute_pseudobulk_adata(
                adata=AnnData_sample, 
                batch_col=batch_col, 
                sample_col=sample_col, 
                celltype_col='cell_type', 
                n_features=n_features, 
                output_dir=resolution_dir,
                Save=False,
                verbose=False
            )
            
            result_dict['n_samples'] = len(pseudobulk_adata)
            
            # Perform dimension reduction
            process_anndata_with_pca(
                adata=AnnData_sample,
                pseudobulk=pseudobulk_dict,
                pseudobulk_anndata=pseudobulk_adata,
                sample_col=sample_col,
                n_expression_pcs=num_DR_components,
                n_proportion_pcs=num_DR_components,
                atac=True,
                output_dir=resolution_dir,
                not_save=True,
                verbose=False
            )

            # Check if column exists
            if column not in pseudobulk_adata.uns:
                all_results.append(result_dict)
                all_resolution_null_results.append(resolution_null_result)
                continue
            
            # Run CCA analysis with specified n_pcs
            cca_result = cca_analysis(
                pseudobulk_adata=pseudobulk_adata,
                column=column,
                sev_col=sev_col,
                n_components=1,
                n_pcs=n_pcs
            )
            
            if cca_result['valid']:
                result_dict['cca_score'] = cca_result['cca_score']
                result_dict['n_features'] = cca_result['n_features']
                result_dict['n_pcs_used'] = cca_result['n_pcs_used']
                print(f"Fine-tuned Resolution {resolution:.3f}: Score {cca_result['cca_score']:.4f}")
                
                # Create CCA visualization plot
                try:
                    # Extract 2D visualization data for plotting
                    (pca_coords_2d, 
                     sev_levels, 
                     cca_model, 
                     cca_score, 
                     samples) = run_cca_on_2d_pca_from_adata(
                        adata=pseudobulk_adata,
                        column=column,
                        sev_col=sev_col
                    )
                    
                    # Save CCA plot
                    plot_path = os.path.join(resolution_dir, f"cca_plot_res_{resolution:.3f}.png")
                    plot_cca_on_2d_pca(
                        pca_coords_2d=pca_coords_2d,
                        sev_levels=sev_levels,
                        cca=cca_model,
                        output_path=plot_path,
                        sample_labels=None,
                        title_suffix=f"Resolution {resolution:.3f}"
                    )
                    if verbose:
                        print(f"Created CCA visualization plot")
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to create CCA visualization: {str(e)}")
                
                # Generate null distribution if computing corrected p-values
                if compute_corrected_pvalues:
                    try:
                        null_distribution = generate_null_distribution_atac(
                            pseudobulk_adata=pseudobulk_adata,
                            column=column,
                            sev_col=sev_col,
                            n_pcs=n_pcs,
                            n_permutations=num_pvalue_simulations,
                            save_path=os.path.join(resolution_dir, f'null_dist_{resolution:.3f}.npy'),
                            verbose=verbose
                        )
                        resolution_null_result['null_scores'] = null_distribution
                        
                        # Compute standard p-value
                        p_value = np.mean(null_distribution >= cca_result['cca_score'])
                        result_dict['p_value'] = p_value
                        
                    except Exception as e:
                        print(f"Warning: Failed to generate null distribution: {str(e)}")
                    
        except Exception as e:
            print(f"Error at fine-tuned resolution {resolution:.3f}: {str(e)}")
        
        all_results.append(result_dict)
        all_resolution_null_results.append(resolution_null_result)

    # Create comprehensive results dataframe
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values("resolution")
    
    # Generate corrected null distribution if computing corrected p-values
    if compute_corrected_pvalues:
        print("\n=== GENERATING CORRECTED NULL DISTRIBUTION ===")
        print("Accounting for resolution selection bias...")
        
        # Filter out null results that failed to generate
        valid_null_results = [r for r in all_resolution_null_results if r['null_scores'] is not None]
        
        if valid_null_results:
            corrected_null_distribution = generate_corrected_null_distribution_atac(
                all_resolution_results=valid_null_results,
                n_permutations=num_pvalue_simulations
            )
            
            # Save corrected null distribution
            corrected_null_dir = os.path.join(main_output_dir, "corrected_null")
            os.makedirs(corrected_null_dir, exist_ok=True)
            corrected_null_path = os.path.join(corrected_null_dir, f'corrected_null_distribution_{dr_type}.npy')
            np.save(corrected_null_path, corrected_null_distribution)
            print(f"Corrected null distribution saved to: {corrected_null_path}")
            
            # Compute corrected p-values for all resolutions
            print("\n=== COMPUTING CORRECTED P-VALUES ===")
            df_results = compute_corrected_pvalues_atac(
                df_results=df_results,
                corrected_null_distribution=corrected_null_distribution,
                output_dir=main_output_dir,
                column=column
            )
            
            # Create visualization of corrected null distribution
            plt.figure(figsize=(10, 6))
            plt.hist(corrected_null_distribution, bins=50, alpha=0.7, color='lightblue', 
                    density=True, edgecolor='black')
            plt.xlabel('Maximum CCA Score (across resolutions)', fontsize=12)
            plt.ylabel('Density', fontsize=12)
            plt.title(f'Corrected Null Distribution\n{column} - Accounts for Resolution Selection', 
                     fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            plt.text(0.02, 0.98, 
                    f'Mean: {np.mean(corrected_null_distribution):.4f}\n'
                    f'Std: {np.std(corrected_null_distribution):.4f}\n'
                    f'95th percentile: {np.percentile(corrected_null_distribution, 95):.4f}',
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            corrected_null_plot_path = os.path.join(corrected_null_dir, 'corrected_null_distribution.png')
            plt.savefig(corrected_null_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Corrected null distribution plot saved to: {corrected_null_plot_path}")
        else:
            print("Warning: No valid null distributions generated, cannot compute corrected p-values")
            compute_corrected_pvalues = False
    
    # Find final best resolution
    valid_results = df_results[~df_results['cca_score'].isna()]
    if valid_results.empty:
        raise ValueError("No valid results obtained.")
    
    final_best_idx = valid_results['cca_score'].idxmax()
    final_best_resolution = valid_results.loc[final_best_idx, 'resolution']
    final_best_score = valid_results.loc[final_best_idx, 'cca_score']
    final_best_pvalue = valid_results.loc[final_best_idx, 'p_value'] if 'p_value' in valid_results.columns else np.nan
    final_best_corrected_pvalue = valid_results.loc[final_best_idx, 'corrected_pvalue'] if compute_corrected_pvalues else np.nan

    print(f"\n=== FINAL RESULTS ===")
    print(f"Best resolution: {final_best_resolution:.3f}")
    print(f"Best CCA score: {final_best_score:.4f}")
    print(f"Number of clusters at best resolution: {valid_results.loc[final_best_idx, 'n_clusters']}")
    print(f"Number of PCs used: {valid_results.loc[final_best_idx, 'n_pcs_used']}")
    if not np.isnan(final_best_pvalue):
        print(f"Standard p-value: {final_best_pvalue:.4f}")
    if compute_corrected_pvalues and not np.isnan(final_best_corrected_pvalue):
        print(f"Corrected p-value: {final_best_corrected_pvalue:.4f}")

    # Create comprehensive summary
    create_comprehensive_summary_atac(
        df_results=df_results,
        best_resolution=final_best_resolution,
        column=column,
        output_dir=main_output_dir,
        has_corrected_pvalues=compute_corrected_pvalues
    )

    # Save complete results
    results_csv_path = os.path.join(main_output_dir, f"all_resolution_results_{dr_type}.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nAll results saved to: {results_csv_path}")

    # Create a final summary report
    final_summary_path = os.path.join(main_output_dir, "FINAL_SUMMARY.txt")
    with open(final_summary_path, 'w') as f:
        f.write("ATAC RESOLUTION OPTIMIZATION FINAL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runtime: {time.time() - start_time:.2f} seconds\n\n")
        
        f.write("OPTIMIZATION PARAMETERS:\n")
        f.write(f"  - Column analyzed: {column}\n")
        f.write(f"  - Representation used: {use_rep}\n")
        f.write(f"  - Number of DM components: {num_DMs}\n")
        f.write(f"  - Number of DR components: {num_DR_components}\n")
        f.write(f"  - Number of PCs used for CCA: {n_pcs}\n")
        f.write(f"  - Number of features: {n_features}\n")
        f.write(f"  - Number of simulations: {num_pvalue_simulations}\n\n")
        
        f.write("RESULTS:\n")
        f.write(f"  - Optimal resolution: {final_best_resolution:.3f}\n")
        f.write(f"  - Best CCA score: {final_best_score:.4f}\n")
        f.write(f"  - Number of clusters: {valid_results.loc[final_best_idx, 'n_clusters']}\n")
        if not np.isnan(final_best_pvalue):
            f.write(f"  - Standard p-value: {final_best_pvalue:.4f}\n")
        if compute_corrected_pvalues and not np.isnan(final_best_corrected_pvalue):
            f.write(f"  - Corrected p-value: {final_best_corrected_pvalue:.4f}\n")
        
        f.write(f"\nTotal resolutions tested: {len(valid_results)}\n")
        f.write(f"  - Coarse search: {len(valid_results[valid_results['pass'] == 'coarse'])} resolutions\n")
        f.write(f"  - Fine search: {len(valid_results[valid_results['pass'] == 'fine'])} resolutions\n")
        
        f.write("\nOUTPUT FILES:\n")
        f.write(f"  - Main directory: {main_output_dir}\n")
        f.write(f"  - Summary plots: {os.path.join(main_output_dir, 'summary')}\n")
        f.write(f"  - Resolution results: {os.path.join(main_output_dir, 'resolutions')}\n")
        if compute_corrected_pvalues:
            f.write(f"  - Corrected p-values: {os.path.join(main_output_dir, 'corrected_p_values')}\n")
            f.write(f"  - Corrected null distribution: {os.path.join(main_output_dir, 'corrected_null')}\n")
    
    print(f"\nFinal summary saved to: {final_summary_path}")
    print(f"\n[Find Optimal Resolution ATAC] Total runtime: {time.time() - start_time:.2f} seconds\n")

    return final_best_resolution, df_results