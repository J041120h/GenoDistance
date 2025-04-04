import anndata as ad
import pandas as pd
from sklearn.cross_decomposition import CCA
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from anndata import AnnData
import time
from pseudobulk import compute_pseudobulk_dataframes
from PCA import process_anndata_with_pca
from CCA import load_severity_levels
from CellType import cell_types, cell_type_assign

def find_optimal_cell_resolution(AnnData_cell, AnnData_sample, output_dir, summary_sample_csv_path, AnnData_sample_path, column):
    score_counter = dict()
    for resolution in np.arange(0.01, 1.01, 0.01):
        print(f"\n\nTesting resolution: {resolution}\n\n")
        cell_types(
            AnnData_cell, 
            cell_column='cell_type', 
            Save=False,
            output_dir=output_dir,
            cluster_resolution=resolution, 
            markers=None, 
            method='average', 
            metric='euclidean', 
            distance_mode='centroid', 
            num_PCs=20, 
            verbose=False
        )
        cell_type_assign(AnnData_cell, AnnData_sample, Save=False, output_dir=output_dir,verbose = False)
        pseudobulk = compute_pseudobulk_dataframes(AnnData_sample, 'batch', 'sample', 'cell_type', output_dir)
        process_anndata_with_pca(adata = AnnData_sample, pseudobulk = pseudobulk, output_dir = output_dir, adata_path=AnnData_sample_path, verbose = False)

        pca_coords = AnnData_sample.uns[column]
        pca_coords_2d = pca_coords[:, :2]  # shape: (n_samples, 2)

        samples = AnnData_sample.obs["sample"].values.unique()
        sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples)

        # CCA on the 2D PCA coordinates vs severity
        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca.transform(pca_coords_2d, sev_levels_2d)
        first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        print(first_component_score)
        score_counter[resolution] = first_component_score
        if 'cell_type' in AnnData_cell.obs.columns:
            del AnnData_cell.obs['cell_type']
        if 'cell_type' in AnnData_sample.obs.columns:
            del AnnData_sample.obs['cell_type']
    
    best_resolution = max(score_counter, key=score_counter.get)
    print(f"Best resolution from first pass: {best_resolution}")
    fine_score_counter = dict()
    for resolution in np.arange(max(0.1, best_resolution - 0.1), min(1.0, best_resolution + 0.1) + 0.01, 0.01):
        cell_types(
            AnnData_cell, 
            cell_column='cell_type', 
            Save=False,
            output_dir=output_dir,
            cluster_resolution=resolution, 
            markers=None, 
            method='average', 
            metric='euclidean', 
            distance_mode='centroid', 
            num_PCs=20, 
            verbose=False
        )
        cell_type_assign(AnnData_cell, AnnData_sample, Save=True, output_dir=output_dir, verbose=False)
        pseudobulk = compute_pseudobulk_dataframes(AnnData_sample, 'batch', 'sample', 'cell_type', output_dir)
        process_anndata_with_pca(adata=AnnData_sample, pseudobulk=pseudobulk, output_dir=output_dir, adata_path=AnnData_sample_path,verbose=False)

        pca_coords = AnnData_sample.uns[column]
        pca_coords_2d = pca_coords[:, :2]  # shape: (n_samples, 2)

        samples = AnnData_sample.obs["sample"].values.unique()
        sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples)

        # CCA on the 2D PCA coordinates vs severity
        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, sev_levels_2d)
        U, V = cca.transform(pca_coords_2d, sev_levels_2d)
        first_component_score = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

        print(f"Fine-tuned Resolution {resolution}: Score {first_component_score}")
        fine_score_counter[resolution] = first_component_score
        if 'cell_type' in AnnData_cell.obs.columns:
            del AnnData_cell.obs['cell_type']
        if 'cell_type' in AnnData_sample.obs.columns:
            del AnnData_sample.obs['cell_type']

    # Find the best resolution from the finer search
    final_best_resolution = max(fine_score_counter, key=fine_score_counter.get)
    print(f"Final best resolution: {final_best_resolution}")

    df_coarse = pd.DataFrame(list(score_counter.items()), columns=["resolution", "score"])
    df_fine = pd.DataFrame(list(fine_score_counter.items()), columns=["resolution", "score"])
    df_results = pd.concat([df_coarse, df_fine], ignore_index=True)
    output_dir = os.path.join(output_dir, "CCA test")
    os.makedirs(output_dir, exist_ok=True) 
    to_csv_path = os.path.join(output_dir, f"resolution_scores_{column}.csv")
    df_results.to_csv(to_csv_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(df_results["resolution"], df_results["score"], marker='o', linestyle='None', color='b', label="CCA Score")
    plt.xlabel("Resolution")
    plt.ylabel("CCA Score")
    plt.title("Resolution vs. CCA Score")
    plt.legend()
    plt.grid(True)

    # Save plot locally
    plot_path = os.path.join(output_dir, f"resolution_vs_cca_score_{column}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Resolution vs. CCA Score plot saved as '{plot_path}'.")
    print("Resolution scores saved as 'resolution_scores.csv'.")
    print("All data saved locally.")

def cca_pvalue_test(
    adata: AnnData,
    summary_sample_csv_path: str,
    column: str,
    input_correlation: float,
    output_directory: str,
    num_simulations: int = 1000,
    verbose: bool = True
):
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import CCA

    start_time = time.time() if verbose else None

    output_directory = os.path.join(output_directory, "CCA test")
    os.makedirs(output_directory, exist_ok=True)

    pca_coords = adata.uns[column]
    if pca_coords.shape[1] < 2:
        raise ValueError("X_pca must have at least 2 components for 2D plotting.")
    pca_coords = pca_coords.values
    pca_coords_2d = pca_coords[:, :2]

    if "sample" not in adata.obs.columns:
        raise KeyError("adata.obs must have a 'sample' column to match the CSV severity data.")
    samples = adata.obs["sample"].unique()
    if len(samples) != pca_coords_2d.shape[0]:
        raise ValueError("The number of PCA rows does not match the number of samples in adata.obs['sample'].")

    sev_levels_2d = load_severity_levels(summary_sample_csv_path, samples)
    sev_levels_1d = sev_levels_2d.flatten()

    simulated_scores = []
    for _ in range(num_simulations):
        permuted_sev_levels = np.random.permutation(sev_levels_1d).reshape(sev_levels_2d.shape)

        cca = CCA(n_components=1)
        cca.fit(pca_coords_2d, permuted_sev_levels)

        U, V = cca.transform(pca_coords_2d, permuted_sev_levels)

        corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        simulated_scores.append(corr)

    simulated_scores = np.array(simulated_scores)

    p_value = np.mean(simulated_scores >= input_correlation)

    plt.figure(figsize=(8, 5))
    plt.hist(simulated_scores, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(input_correlation, color='red', linestyle='dashed', linewidth=2,
                label=f'Observed correlation: {input_correlation}')
    plt.xlabel('Simulated Correlation Scores')
    plt.ylabel('Frequency')
    plt.title('Permutation Test: CCA Correlations')
    plt.legend()

    plot_path = os.path.join(output_directory, f"cca_pvalue_distribution_{column}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    p_value_path = os.path.join(output_directory, f"cca_pvalue_result_{column}.txt")
    with open(p_value_path, "w") as f:
        f.write(f"Observed correlation: {input_correlation}\n")
        f.write(f"P-value: {p_value}\n")

    print(f"P-value for observed correlation {input_correlation}: {p_value}")

    if verbose:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("CCA p-value test on cell proportions completed.")
        print(f"\n[CCA p-test] Total runtime processing: {elapsed_time:.2f} seconds\n")

    return p_value