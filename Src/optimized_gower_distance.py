# Gower distance + UMAP — nonlinear mixed‑datatype embedding
from joblib import Parallel, delayed
import os
import math
import psutil # Useful for real-time memory checks
import gower
import numpy as np
import pandas as pd


def _recommend_n_chunks_expert(n_rows, n_cols):
    n_cores = os.cpu_count() or 4
    available_gb = psutil.virtual_memory().available / (1024**3)
    
    # Heuristic: Each chunk calculation (chunk_size * n_rows) 
    # should fit comfortably in 10% of available RAM to allow for overhead.
    # float32 = 4 bytes
    max_chunk_size = int((available_gb * 0.1 * 1024**3) / (n_rows * 4))
    
    # Stay between 500 and 2000 for Gower stability
    target_rows = max(500, min(max_chunk_size, 2000))
    
    by_size = math.ceil(n_rows / target_rows)
    # Ensure we at least use all cores twice to handle "straggler" chunks
    by_cpu = n_cores * 2 
    
    n_chunks = max(by_size, by_cpu)
    
    # Expert touch: Round up to the nearest multiple of n_cores
    return int(math.ceil(n_chunks / n_cores) * n_cores)
    

def _gower_chunk(chunk_indices, df_full, cat_mask):
    """
    Worker function: compute Gower distances between a row subset
    and the full dataset.
    """
    df_chunk = df_full.iloc[chunk_indices]
    return gower.gower_matrix(df_chunk, df_full, cat_features=cat_mask)


def compute_gower(data, cat_cols, n_jobs=-1, verbose=True):
    """
    Parallel full Gower distance computation for mixed-type data.

    Parameters
    ----------
    df : pd.DataFrame
        Input mixed-type dataframe.
    cat_cols : list or set
        Names of categorical columns.
    n_chunks : int
        Number of row blocks to split into.
    n_jobs : int
        Number of parallel jobs (-1 = all cores).
    verbose : bool
        Print progress information.

    Returns
    -------
    gower_matrix : np.ndarray
        Full NxN Gower distance matrix.
    """
    # -----------------------------
    # Copy and optimize
    # -----------------------------
    df_opt = data.copy()
    cat_cols = set(cat_cols)
    n_rows, n_cols = df_opt.shape
    
    # helper function
    n_chunks = _recommend_n_chunks_expert(n_rows, n_cols)
    
    # Drop constant columns immediately
    constant_cols = [c for c in df_opt.columns if df_opt[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df_opt = df_opt.drop(columns=constant_cols)

    # Rebuild categorical mask
    cat_mask = [col in cat_cols for col in df_opt.columns]

    # Cast types
    for col, is_cat in zip(df_opt.columns, cat_mask):
        if is_cat:
            df_opt[col] = df_opt[col].astype("category")
        elif pd.api.types.is_numeric_dtype(df_opt[col]):
            df_opt[col] = pd.to_numeric(df_opt[col], errors="coerce").astype("float32")

    # n_rows = df_opt.shape[0]
    chunks = np.array_split(np.arange(n_rows), n_chunks)

    if verbose:
        print(f"Starting parallel Gower on {n_rows:,} rows × {df_opt.shape[1]} cols")
        print(f"Chunks: {n_chunks} | Jobs: {n_jobs}")

    # -----------------------------
    # Parallel chunk computation
    # -----------------------------
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_gower_chunk)(chunk, df_opt, cat_mask)
        for chunk in chunks
    )

    # -----------------------------
    # Stack into a full matrix
    # -----------------------------
    gower_matrix = np.vstack(results).astype("float32")

    if verbose:
        print("Calculation complete!")
        print(f"Gower shape: {gower_matrix.shape}")
        print(f"Approx memory: {gower_matrix.nbytes / 1024**2:.2f} MB")

    return gower_matrix