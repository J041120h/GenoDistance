import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import harmonypy as hm
import matplotlib.pyplot as plt
from Harmony import treecor_harmony
from SampleSimilarity import calculate_sample_distances

def main():
    output_dir = "/users/harry/desktop/GenoDistance/result"
    count_path = "/users/harry/output_matrix.csv"
    sample_meta_path = "/users/harry/desktop/GenoDistance/Data/raw_data_sample_meta.csv"
    min_cells= 100
    min_features= 3
    pct_mito_cutoff=20
    exclude_genes=None
    vars_to_regress=['sample']
    resolution=0.5
    verbose=True
    cell_meta_path=None
    num_PCs=20
    num_harmony=20
    AnnData = sc.read_h5ad("/users/harry/desktop/GenoDistance/result/integrate.h5ad")

    # AnnData = treecor_harmony(count_path, sample_meta_path, output_dir)
    calculate_sample_distances(AnnData, output_dir)

if __name__ == '__main__':
    main()