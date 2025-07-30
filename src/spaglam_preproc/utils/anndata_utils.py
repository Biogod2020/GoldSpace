# spaglam_preproc/utils/anndata_utils.py
import anndata
import numpy as np

def safe_get_spatial_coords(adata: anndata.AnnData) -> np.ndarray:
    """
    Safely retrieves spatial coordinates from an AnnData object.
    Checks for common keys and validates the shape.

    Args:
        adata: The AnnData object.

    Returns:
        A NumPy array of shape (n_obs, 2) with spatial coordinates.
    
    Raises:
        ValueError: If no valid spatial coordinates are found.
    """
    if 'spatial' in adata.obsm:
        coords = adata.obsm['spatial']
        if isinstance(coords, np.ndarray) and coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2]  # Return only the first two columns (x, y)
    
    raise ValueError(
        "Could not find valid spatial coordinates in `adata.obsm['spatial']`. "
        "Expected a NumPy array of shape (n_obs, 2)."
    )
