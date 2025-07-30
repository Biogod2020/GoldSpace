# spaglam_preproc/utils/validation.py

import logging
import anndata
import numpy as np
import os
from ..core.image_tiler import ImageHandler
from .anndata_utils import safe_get_spatial_coords

def pre_run_validation(adata: anndata.AnnData, image_handler: ImageHandler, config: dict):
    """
    Performs a series of checks on inputs before starting the main processing loop.
    Raises RuntimeError if a critical validation fails.
    """
    logging.info("🔬 Performing pre-run validation checks...")
    valid = True

    # 1. Check for required AnnData fields
    if 'spatial_connectivities' not in adata.obsp:
        logging.error("Validation failed: `adata.obsp['spatial_connectivities']` not found. A spatial graph is required.")
        valid = False
    
    try:
        coords = safe_get_spatial_coords(adata)
        if coords is None:
            raise ValueError("No valid spatial coordinates found.")
    except ValueError as e:
        logging.error(f"Validation failed: {e}")
        valid = False

    # 2. Check a sample coordinate against image boundaries
    if valid:
        img_w, img_h = image_handler.get_dimensions()
        sample_coord = coords[0]
        if not (0 <= sample_coord[0] < img_w and 0 <= sample_coord[1] < img_h):
            logging.warning(
                f"Validation warning: First spot coordinate ({sample_coord}) is outside image "
                f"dimensions (Width={img_w}, Height={img_h}). This may be acceptable for some datasets."
            )
        logging.info(f"Image dimensions (W x H): {img_w} x {img_h}. AnnData spots: {adata.n_obs}.")

    # 3. Check HVG list coverage
    hvg_path = config['paths'].get('hvg_list_path')
    if hvg_path:
        try:
            hvg_list = set(np.loadtxt(hvg_path, dtype=str))
            adata_genes = set(adata.var_names)
            overlap = len(hvg_list.intersection(adata_genes))
            if overlap == 0:
                logging.error("Validation failed: No overlap between provided HVG list and genes in AnnData object.")
                valid = False
            else:
                coverage = (overlap / len(hvg_list)) * 100
                logging.info(f"HVG list coverage: {overlap}/{len(hvg_list)} genes from the list found in AnnData ({coverage:.2f}%).")
        except FileNotFoundError:
            logging.error(f"Validation failed: HVG list file not found at '{hvg_path}'.")
            valid = False


    # 4. Check model config if precomputing embeddings
    if config['preprocessing']['precompute_embeddings']:
        if 'model' not in config or config['model'] is None:
            logging.error("Validation failed: `model` configuration is required when `precompute_embeddings` is true.")
            valid = False
        else:
            if not os.path.exists(config['model']['model_path']):
                logging.error(f"Validation failed: Model checkpoint not found at '{config['model']['model_path']}'.")
                valid = False


    if not valid:
        raise RuntimeError("Pre-run validation failed. Please check the logs for errors and correct your configuration or data.")
    
    logging.info("✅ Pre-run validation passed successfully.")
