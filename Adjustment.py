import pandas as pd
import scanpy as sc
import os

def convert_csv_to_h5ad(csv_path: str):
    """
    Converts a CSV file containing count data to an H5AD file and saves it with the same name.
    
    Parameters:
    csv_path (str): Path to the input CSV file.
    """
    # Read CSV file
    count = pd.read_csv(csv_path, index_col=0)

    count = count[count.sum(axis=1) > 0]
    
    adata = sc.AnnData(count.T)
    adata.var_names = count.index.astype(str)
    adata.obs_names = count.columns.astype(str)
    
    # Define H5AD output path
    h5ad_path = os.path.splitext(csv_path)[0] + ".h5ad"
    
    # Save as H5AD
    adata.write(h5ad_path)
    
    print(f"Successfully saved: {h5ad_path}")

def print_obs_columns(adata):
    """
    Prints all column names in adata.obs.

    Parameters
    ----------
    adata : AnnData
        An AnnData object containing single-cell data.

    Returns
    -------
    None
    """
    print("Columns in adata.obs:")
    for col in adata.obs.columns:
        print(col)

if __name__ == "__main__":
    AnnData_sample = sc.read_h5ad("/users/harry/desktop/GenoDistance/result/harmony/adata_sample.h5ad")
    convert_csv_to_h5ad(csv_file_path)


