import os
import pandas as pd
import numpy as np

def distanceCheck(df_path, row, method, summary_csv_path, adata=None):
    """
    Calculate in-group vs. between-group distances either:
    1) Using severity levels from 'adata.obs["sev.level"]' (if available), or
    2) Using the first two letters of each sample name (fallback).

    Then update a summary CSV file with the resulting score.

    Parameters
    ----------
    df_path : str
        Path to the CSV file containing the distance matrix.
    row : str
        Row name in the summary CSV to update (e.g., 'cell_proportion' or 'gene_expression').
    method : str
        Column name in the summary CSV to update.
    summary_csv_path : str
        Path to the summary CSV file.
    adata : anndata.AnnData or None
        An AnnData object where `adata.obs["sev.level"]` stores the sample severity levels 
        (optional). If not provided or 'sev.level' is absent, grouping will be done based on 
        the first two letters of the sample name.
    """

    # Read the distance matrix
    df = pd.read_csv(df_path, index_col=0)

    # Get the directory of the CSV file
    directory = os.path.dirname(os.path.abspath(df_path))

    # Get the sample names from the distance matrix
    samples = df.index.tolist()

    # Decide on the grouping approach
    use_sev_level = False
    if adata is not None:
        # Check if 'sev.level' exists in adata.obs
        if "sev.level" in adata.obs.columns:
            use_sev_level = True
            # Ensure 'sample' column exists if we are to map severity by sample
            if "sample" not in adata.obs.columns:
                raise KeyError("'sample' column is missing in adata.obs, but 'sev.level' is present.")
    
    # Build groups dictionary
    if use_sev_level:
        # Create a mapping from cell to sample
        cell_to_sample = adata.obs["sample"]
        # Get the unique samples in adata
        samples_in_data = set(cell_to_sample.unique())

        # Map each sample name to the mean 'sev.level' of cells belonging to that sample
        groups = {
            sample: adata.obs.loc[cell_to_sample[cell_to_sample == sample].index, "sev.level"].mean()
            for sample in samples if sample in samples_in_data
        }

        # Check for any sample in the distance matrix but missing in adata 
        missing_samples = [s for s in samples if s not in samples_in_data]
        if missing_samples:
            print(f"Warning: The following samples are in the distance matrix but not in adata.obs: {missing_samples}")
    else:
        # Fallback: group by the first two letters of the sample name
        groups = {sample: sample[:2] for sample in samples}

    # Initialize lists for in-group and between-group distances
    in_group_distances = []
    between_group_distances = []

    # Loop over all pairs of samples to compute distances
    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            if i >= j:
                # Avoid redundant pairs and self-distances
                continue
            distance = df.iloc[i, j]
            # Compare group values
            if groups[sample_i] == groups[sample_j]:
                in_group_distances.append(distance)
            else:
                between_group_distances.append(distance)

    # Calculate average distances
    average_in_group_distance = np.mean(in_group_distances) if in_group_distances else np.nan
    average_between_group_distance = np.mean(between_group_distances) if between_group_distances else np.nan

    # Calculate the score (handle potential division by zero if in_group_distances is empty)
    if np.isnan(average_in_group_distance) or average_in_group_distance == 0:
        score = np.nan
    else:
        score = average_between_group_distance / average_in_group_distance

    # Prepare the result string
    result = (
        f"Average in-group distance: {average_in_group_distance}\n"
        f"Average between-group distance: {average_between_group_distance}\n"
        f"Score: {score}"
    )

    # Define the output file path
    output_file = os.path.join(directory, 'distance_results.txt')

    # Write the results to a text file
    with open(output_file, 'w') as f:
        f.write(result)

    print(f"Results saved to {output_file}")

    # Update the summary CSV
    if os.path.isfile(summary_csv_path):
        # Read the existing summary CSV
        summary_df = pd.read_csv(summary_csv_path, index_col=0)
    else:
        # Create a new DataFrame with the required rows and columns
        rows = ['cell_proportion', 'gene_expression']
        summary_df = pd.DataFrame(index=rows)

    # Ensure the method column exists in summary_df
    if method not in summary_df.columns:
        summary_df[method] = np.nan

    # Update the cell with the computed score
    summary_df.loc[row, method] = score

    # Save the updated summary DataFrame
    summary_df.to_csv(summary_csv_path)
    print(f"Summary updated in {summary_csv_path}")
