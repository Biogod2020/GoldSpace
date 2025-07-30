#!/bin/bash

echo "🚀 Starting to build the 'spaglam_preproc' package structure inside 'src/'..."
echo ""

# --- Create Directory Structure ---
mkdir -p src/spaglam_preproc/core
mkdir -p src/spaglam_preproc/utils
echo "✅ Directory structure created."

# --- Create Core Logic Files ---

cat << 'EOF' > src/spaglam_preproc/core/graph_builder.py
# spaglam_preproc/core/graph_builder.py

from collections import deque
from scipy.sparse import csr_matrix

def get_k_hop_neighborhood(
    adjacency_matrix: csr_matrix, start_node_idx: int, k: int
) -> list[int]:
    """
    Finds all unique nodes within k hops of a starting node using Breadth-First Search (BFS).
    This is the most efficient method for this task.

    Args:
        adjacency_matrix: The sparse adjacency matrix (e.g., from adata.obsp['spatial_connectivities']).
        start_node_idx: The integer index of the starting node.
        k: The number of hops (e.g., 1 for immediate neighbors, 2 for neighbors of neighbors).

    Returns:
        A list of unique integer indices of all nodes in the k-hop neighborhood, including the start node.
    """
    if k == 0:
        return [start_node_idx]

    visited = {start_node_idx}
    # queue stores tuples of (node_index, current_hop_level)
    queue = deque([(start_node_idx, 0)])
    
    # We add the start node to the final list
    neighborhood = [start_node_idx]

    while queue:
        current_node, level = queue.popleft()

        if level >= k:
            continue

        # Get neighbors using the efficient .indices attribute of CSR matrices
        # adjacency_matrix.indices[start:end] slices the column indices for a given row
        start_ptr = adjacency_matrix.indptr[current_node]
        end_ptr = adjacency_matrix.indptr[current_node + 1]
        neighbors = adjacency_matrix.indices[start_ptr:end_ptr]

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                neighborhood.append(neighbor)
                queue.append((neighbor, level + 1))
                
    return neighborhood
EOF

cat << 'EOF' > src/spaglam_preproc/core/gene_encoder.py
# spaglam_preproc/core/gene_encoder.py

import numpy as np

def generate_gene_sentence(
    expression_vector: np.ndarray,
    gene_names: np.ndarray,
    n_top_genes: int
) -> str:
    """
    Generates a gene sentence string from a single spot's expression vector.
    This function operates entirely in memory.

    Args:
        expression_vector: A 1D numpy array of gene expression values.
        gene_names: A 1D numpy array of corresponding gene names for the expression vector.
        n_top_genes: The number of top genes to include in the sentence.

    Returns:
        A space-separated string of the top N gene names.
    """
    # np.argsort is highly optimized for this task. We reverse it to get descending order.
    sorted_indices = np.argsort(expression_vector)[-1::-1]
    
    # Slice the top N indices and corresponding names
    top_n_indices = sorted_indices[:n_top_genes]
    top_genes = gene_names[top_n_indices]
    
    return " ".join(top_genes)
EOF

cat << 'EOF' > src/spaglam_preproc/core/image_tiler.py
# spaglam_preproc/core/image_tiler.py

import logging
from pathlib import Path
from typing import Union, Optional

import numpy as np
from PIL import Image

# openslide-python is an optional dependency
try:
    import openslide
except ImportError:
    openslide = None

# squidpy is an optional dependency for reading from AnnData
try:
    from squidpy.im import ImageContainer
except ImportError:
    ImageContainer = None


class ImageHandler:
    """
    A unified interface to handle various image sources for tile extraction.
    It can be initialized with an AnnData object, a file path (WSI or standard image),
    or a pre-loaded image object (PIL Image or NumPy array).
    """
    def __init__(self, source: Optional[Union[str, Path, object]], adata: Optional[object] = None):
        self.image_obj = None
        self.width, self.height = 0, 0
        self._load_image(source, adata)

    def _load_image(self, source, adata):
        """Internal method to load the image from the specified source."""
        # 1. Try to load from AnnData object if it's the primary source
        if source is None and adata is not None and ImageContainer is not None:
            spatial_key = list(adata.uns.get('spatial', {}).keys())
            if spatial_key:
                # Assuming standard squidpy storage format
                img_container = adata.uns['spatial'][spatial_key[0]]['images'].get('hires')
                if isinstance(img_container, ImageContainer):
                    self.image_obj = img_container
                    self.width, self.height = self.image_obj.shape[1], self.image_obj.shape[0]
                    logging.info(f"Loaded image '{spatial_key[0]}' from AnnData object.")
                    return
            else:
                 raise ValueError("Image source is None and no spatial image found in adata.uns['spatial'].")
        
        if source is None:
            raise ValueError("No image source provided (path or AnnData).")

        # 2. Handle file paths (WSI or standard formats)
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found at: {path}")

            wsi_extensions = {".svs", ".tiff", ".tif", ".vms", ".vmu", ".ndpi", ".scn", ".mrxs", ".svslide"}
            if openslide and path.suffix.lower() in wsi_extensions:
                try:
                    self.image_obj = openslide.OpenSlide(str(path))
                    self.width, self.height = self.image_obj.dimensions
                    logging.info(f"Loaded WSI image: {path}")
                    return
                except openslide.OpenSlideError:
                    logging.warning(f"Could not open {path} with openslide, trying Pillow.")

            try:
                img = Image.open(path)
                self.image_obj = img.convert("RGB")
                self.width, self.height = self.image_obj.size
                logging.info(f"Loaded standard image with Pillow: {path}")
                return
            except Exception as e:
                raise IOError(f"Failed to load image file {path} with both OpenSlide and Pillow.") from e

        # 3. Handle pre-loaded image objects
        elif isinstance(source, Image.Image):
            self.image_obj = source.convert("RGB")
            self.width, self.height = self.image_obj.size
            logging.info("Loaded image from pre-loaded PIL.Image object.")
            return
        elif isinstance(source, np.ndarray):
            self.image_obj = Image.fromarray(source).convert("RGB")
            self.width, self.height = self.image_obj.size
            logging.info("Loaded image from pre-loaded NumPy array.")
            return
        
        raise TypeError(f"Unsupported image source type: {type(source)}")

    def get_dimensions(self) -> tuple[int, int]:
        return self.width, self.height

    def get_tile(self, coordinates: np.ndarray, tile_size: int) -> Image.Image:
        """
        Extracts a single image tile in memory. Handles boundary conditions.
        """
        col, row = int(round(coordinates[0])), int(round(coordinates[1]))
        half_tile = tile_size // 2

        top_left_x = col - half_tile
        top_left_y = row - half_tile

        read_left = max(top_left_x, 0)
        read_top = max(top_left_y, 0)
        read_right = min(top_left_x + tile_size, self.width)
        read_bottom = min(top_left_y + tile_size, self.height)
        
        read_width = read_right - read_left
        read_height = read_bottom - read_top

        if read_width <= 0 or read_height <= 0:
            return Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        
        if openslide and isinstance(self.image_obj, openslide.OpenSlide):
            region = self.image_obj.read_region((read_left, read_top), 0, (read_width, read_height)).convert("RGB")
        elif ImageContainer and isinstance(self.image_obj, ImageContainer):
            region_np = self.image_obj.crop_corner(read_left, read_top, size_x=read_width, size_y=read_height).data
            region = Image.fromarray(region_np).convert("RGB")
        elif isinstance(self.image_obj, Image.Image):
            region = self.image_obj.crop((read_left, read_top, read_right, read_bottom))
        else:
            raise TypeError(f"Cannot extract tile from unsupported image object type: {type(self.image_obj)}")

        tile_img = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        paste_x = read_left - top_left_x
        paste_y = read_top - top_left_y
        tile_img.paste(region, (paste_x, paste_y))
        
        return tile_img
EOF

cat << 'EOF' > src/spaglam_preproc/core/dataset_writer.py
# spaglam_preproc/core/dataset_writer.py

import os
import io
import json
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Dict, Any, Optional

import torch
import pandas as pd
import anndata
import webdataset as wds
import numpy as np
from scipy.sparse import issparse, csr_matrix

from ..utils.validation import pre_run_validation
from ..utils.qc_tools import generate_summary_report, generate_visual_artifact, display_visual_artifact_notebook
from ..utils.anndata_utils import safe_get_spatial_coords
from .graph_builder import get_k_hop_neighborhood
from .gene_encoder import generate_gene_sentence
from .image_tiler import ImageHandler

# Import open_clip conditionally for pre-computing embeddings
try:
    import open_clip
except ImportError:
    open_clip = None

# Detect if running in a notebook for TQDM
def _is_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except:
        return False

TQDM_BAR = None
if _is_notebook():
    from tqdm.notebook import tqdm
    TQDM_BAR = tqdm
else:
    from tqdm import tqdm
    TQDM_BAR = tqdm


def _process_subgraph_to_sample(
    center_spot_info: pd.Series,
    *, # Enforce keyword-only arguments
    adata: anndata.AnnData,
    adata_hvg: anndata.AnnData,
    adjacency_matrix: csr_matrix,
    gene_names_hvg: np.ndarray,
    image_handler: ImageHandler,
    config: Dict[str, Any],
    model_resources: Dict[str, Any],
    collect_qc_sample: bool = False
) -> tuple[Optional[Dict], Optional[Dict], Optional[str]]:
    """
    Worker function for a single center spot. It performs the entire pipeline in memory.
    Returns the sample for the shard, an optional sample for QC, and an error message.
    """
    center_spot_id = center_spot_info.name
    qc_sample = None
    try:
        # a. Get k-hop neighborhood using BFS
        center_node_idx = adata.obs_names.get_loc(center_spot_id)
        k = config['preprocessing']['neighborhood_hops']
        global_indices = get_k_hop_neighborhood(adjacency_matrix, center_node_idx, k)
        all_spot_ids = adata.obs_names[global_indices].tolist()
        num_nodes = len(all_spot_ids)

        # b. Build local graph structure
        global_id_to_local_idx = {sid: i for i, sid in enumerate(all_spot_ids)}
        local_edge_index = []
        for local_u_idx, u_id in enumerate(all_spot_ids):
            u_global_idx = adata.obs_names.get_loc(u_id)
            start, end = adjacency_matrix.indptr[u_global_idx], adjacency_matrix.indptr[u_global_idx + 1]
            for v_global_idx in adjacency_matrix.indices[start:end]:
                v_id = adata.obs_names[v_global_idx]
                if v_id in global_id_to_local_idx:
                    local_v_idx = global_id_to_local_idx[v_id]
                    if local_u_idx < local_v_idx:
                        local_edge_index.append([local_u_idx, local_v_idx])
        
        # c. In-memory generation of raw data
        images_to_process = []
        texts_to_process = []
        spatial_coords = safe_get_spatial_coords(adata)
        for spot_id in all_spot_ids:
            spot_idx = adata.obs_names.get_loc(spot_id)
            coords = spatial_coords[spot_idx]
            tile = image_handler.get_tile(coords, config['preprocessing']['tile_size'])
            images_to_process.append(tile)
            
            expression = adata_hvg.X[spot_idx]
            expression_vector = expression.toarray().flatten() if issparse(expression) else np.array(expression).flatten()
            sentence = generate_gene_sentence(
                expression_vector, gene_names_hvg, config['preprocessing']['n_top_genes_in_sentence'])
            texts_to_process.append(sentence)
        
        # Collect a QC sample if requested (only for the center node)
        if collect_qc_sample:
            qc_sample = {
                'id': center_spot_id,
                'tile': images_to_process[0],
                'sentence': texts_to_process[0]
            }

        # d. Construct the final sample
        output_sample = {
            "__key__": center_spot_id,
            "json": json.dumps({"num_nodes": num_nodes, "edge_index": local_edge_index}).encode('utf-8')
        }

        if config['preprocessing']['precompute_embeddings']:
            model = model_resources['model']
            image_preprocessor = model_resources['image_preprocessor']
            tokenizer = model_resources['tokenizer']
            device = model_resources['device']

            image_input = torch.stack([image_preprocessor(img) for img in images_to_process])
            text_input = tokenizer(texts_to_process)
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_input, text_input = image_input.to(device), text_input.to(device)
                image_embeddings = model.encode_image(image_input).cpu()
                text_embeddings = model.encode_text(text_input).cpu()

            for i in range(num_nodes):
                img_buf, txt_buf = io.BytesIO(), io.BytesIO()
                torch.save(image_embeddings[i], img_buf)
                torch.save(text_embeddings[i], txt_buf)
                output_sample[f"{i}.image.pth"] = img_buf.getvalue()
                output_sample[f"{i}.text.pth"] = txt_buf.getvalue()
        else:
            for i in range(num_nodes):
                img_buf = io.BytesIO()
                images_to_process[i].save(img_buf, format="PNG")
                output_sample[f"{i}.png"] = img_buf.getvalue()
                output_sample[f"{i}.txt"] = texts_to_process[i].encode('utf-8')
        
        return output_sample, qc_sample, None

    except Exception as e:
        logging.error(f"Error processing {center_spot_id}", exc_info=True)
        return None, None, f"Skipping {center_spot_id}: {type(e).__name__} - {e}"

class SpaglamPipeline:
    """
    An object-oriented wrapper for the SpaGLaM preprocessing pipeline.
    Designed for easy use in both scripts and interactive notebook environments.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adata = None
        self.adata_hvg = None
        self.adjacency_matrix = None
        self.gene_names_hvg = None
        self.image_handler = None
        self.model_resources = {}
        self.qc_samples_collected = []
        self.metrics = {
            'config': self.config,
            'timing': {},
            'counts': {},
            'graph_stats': {'num_nodes': [], 'num_edges': []},
        }

        self._load_resources()
        pre_run_validation(self.adata, self.image_handler, self.config)

    def _load_resources(self):
        """Loads all heavy resources like data and models into memory."""
        logging.info("--- Loading and Preparing Resources ---")
        
        logging.info(f"💾 Loading AnnData from: {self.config['paths']['adata_path']}")
        self.adata = anndata.read_h5ad(self.config['paths']['adata_path'])
        
        logging.info("🖼️ Initializing ImageHandler...")
        self.image_handler = ImageHandler(
            source=self.config['paths'].get('image_path'), 
            adata=self.adata
        )

        logging.info("🧬 Preparing gene lists...")
        hvg_list_path = self.config['paths'].get('hvg_list_path')
        if hvg_list_path and os.path.exists(hvg_list_path):
            hvg_list = np.loadtxt(hvg_list_path, dtype=str)
        else:
            hvg_list = self.adata.var_names
            logging.warning("No HVG list provided or found. Using all genes.")
        
        self.adata_hvg = self.adata[:, self.adata.var_names.isin(hvg_list)].copy()
        self.gene_names_hvg = self.adata_hvg.var_names.to_numpy()
        
        self.adjacency_matrix = self.adata.obsp['spatial_connectivities'].tocsr()

        if self.config['preprocessing']['precompute_embeddings']:
            if open_clip is None:
                raise ImportError("`open-clip-torch` is required to pre-compute embeddings. Install with `pip install open-clip-torch`.")
            logging.info("🔧 Loading OmiCLIP model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, image_preprocessor = open_clip.create_model_and_transforms(
                self.config['model']['model_name'], pretrained=self.config['model']['model_path'], device=device)
            model.eval()
            self.model_resources = {
                "model": model, 
                "image_preprocessor": image_preprocessor, 
                "tokenizer": open_clip.get_tokenizer(self.config['model']['model_name']),
                "device": device
            }
            logging.info(f"🔌 Using device: {device}")
        logging.info("--- Resource Loading Complete ---")

    def run(self):
        """Executes the main parallel processing pipeline."""
        start_time = time.time()
        output_dir = self.config['paths']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        shards_pattern = os.path.join(output_dir, "shard-%06d.tar")
        
        num_to_process = self.config['performance'].get('num_spots_to_process', -1)
        if num_to_process != -1:
            spots_to_process = self.adata.obs.head(num_to_process)
            logging.info(f"🔬 Processing a subset of {len(spots_to_process)} spots.")
        else:
            spots_to_process = self.adata.obs
            logging.info(f"✅ Processing all {len(spots_to_process)} spots.")
        
        qc_config = self.config.get('qc', {})
        qc_enabled = qc_config.get('enabled', True)
        num_visual_samples = qc_config.get('num_visual_samples', 0)
        self.qc_samples_collected.clear()
        qc_indices = set(random.sample(range(len(spots_to_process)), k=min(num_visual_samples, len(spots_to_process)))) if qc_enabled else set()

        worker_fn = partial(
            _process_subgraph_to_sample,
            adata=self.adata, adata_hvg=self.adata_hvg, adjacency_matrix=self.adjacency_matrix,
            gene_names_hvg=self.gene_names_hvg, image_handler=self.image_handler,
            config=self.config, model_resources=self.model_resources
        )
        
        success_count, error_count = 0, 0
        with wds.ShardWriter(shards_pattern, maxcount=self.config['performance']['max_samples_per_shard']) as sink:
            with ThreadPoolExecutor(max_workers=self.config['performance']['max_workers']) as executor:
                futures = {
                    executor.submit(worker_fn, spots_to_process.iloc[i], collect_qc_sample=(i in qc_indices)): i 
                    for i in range(len(spots_to_process))
                }
                
                pbar = TQDM_BAR(as_completed(futures), total=len(spots_to_process), desc="Generating Shards", unit="spot")
                for future in pbar:
                    sample, qc_sample, error_msg = future.result()
                    if sample:
                        sink.write(sample)
                        success_count += 1
                        if qc_sample:
                            self.qc_samples_collected.append(qc_sample)
                        
                        graph_info = json.loads(sample['json'].decode('utf-8'))
                        self.metrics['graph_stats']['num_nodes'].append(graph_info['num_nodes'])
                        self.metrics['graph_stats']['num_edges'].append(len(graph_info['edge_index']))
                    else:
                        error_count += 1
                        if error_count < 20:
                            logging.warning(error_msg)
        
        self._finalize_run(start_time, success_count, error_count)
        return self

    def _finalize_run(self, start_time, success_count, error_count):
        """Logs metrics and generates QC artifacts after the run."""
        elapsed_time = time.time() - start_time
        self.metrics['timing']['total_runtime_minutes'] = round(elapsed_time / 60, 2)
        self.metrics['timing']['spots_per_second'] = round(success_count / elapsed_time if elapsed_time > 0 else 0, 2)
        self.metrics['counts']['spots_processed'] = success_count
        self.metrics['counts']['spots_failed'] = error_count
        
        if self.metrics['graph_stats']['num_nodes']:
            self.metrics['graph_stats']['avg_nodes_per_subgraph'] = round(np.mean(self.metrics['graph_stats']['num_nodes']), 2)
            self.metrics['graph_stats']['avg_edges_per_subgraph'] = round(np.mean(self.metrics['graph_stats']['num_edges']), 2)
            self.metrics['graph_stats']['max_nodes_per_subgraph'] = int(np.max(self.metrics['graph_stats']['num_nodes']))
        
        logging.info("\n" + "="*80)
        logging.info("🏁 Preprocessing Pipeline Finished!")
        logging.info(f"  - Successfully processed: {success_count} spots")
        logging.info(f"  - Skipped due to errors:  {error_count} spots")
        logging.info(f"  - Total time:             {self.metrics['timing']['total_runtime_minutes']:.2f} minutes")
        logging.info(f"  - Avg. throughput:        {self.metrics['timing']['spots_per_second']:.2f} spots/sec")
        logging.info(f"  - Output saved to:        {self.config['paths']['output_dir']}")
        
        if self.config.get('qc', {}).get('enabled', True):
            generate_summary_report(self.metrics, self.config['paths']['output_dir'])
            generate_visual_artifact(self.qc_samples_collected, self.config['paths']['output_dir'], self.config['qc']['num_visual_samples'])
        
        logging.info("="*80)
        
    def display_samples(self):
        """Displays the QC visual artifact directly in a notebook environment."""
        if not _is_notebook():
            logging.warning("Sample display is only available in a notebook environment.")
            return
        
        artifact_path = os.path.join(self.config['paths']['output_dir'], "qc_sample_grid.png")
        if os.path.exists(artifact_path):
            display_visual_artifact_notebook(artifact_path)
        else:
            logging.error("QC visual artifact not found. Please run the pipeline first with QC enabled.")


def create_dataset_shards(config: Dict[str, Any]):
    """
    High-level function to instantiate and run the SpaglamPipeline.
    This can be called from the CLI or a script.
    
    Args:
        config: A dictionary containing the pipeline configuration.
    """
    pipeline = SpaglamPipeline(config)
    pipeline.run()
EOF

echo "✅ Core logic files created."

# --- Create Utils, Config, and CLI ---

cat << 'EOF' > src/spaglam_preproc/utils/anndata_utils.py
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
EOF

cat << 'EOF' > src/spaglam_preproc/utils/logging_setup.py
# spaglam_preproc/utils/logging_setup.py

import logging
from rich.logging import RichHandler

def setup_logging(log_path: str):
    """
    Configures a rich logger to print to the console and a file.

    Args:
        log_path: Path to the output log file.
    """
    # Configure the rich handler for beautiful console output
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
        tracebacks_suppress=[__import__("typer")], # Suppress typer's internal traceback frames
    )
    
    # Configure the file handler for persistent logging
    file_handler = logging.FileHandler(log_path, mode='w') # Overwrite log on each run
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s - %(message)s")
    )

    # Get the root logger and add handlers
    # We set the level on the handlers individually to control verbosity
    rich_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[rich_handler, file_handler]
    )
EOF

cat << 'EOF' > src/spaglam_preproc/utils/validation.py
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
EOF

cat << 'EOF' > src/spaglam_preproc/utils/qc_tools.py
# spaglam_preproc/utils/qc_tools.py

import json
import logging
import math
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np

# For notebook display
try:
    from IPython.display import display
except ImportError:
    display = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def generate_summary_report(metrics: dict, output_dir: str):
    """Saves a JSON summary of the preprocessing run."""
    report_path = Path(output_dir) / "qc_summary.json"
    try:
        # Convert NumPy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=convert_numpy)
        logging.info(f"📊 QC summary report saved to: {report_path}")
    except Exception as e:
        logging.error(f"Failed to save QC summary report: {e}")

def generate_visual_artifact(samples: list, output_dir: str, num_samples: int):
    """Creates and saves a grid image of sample tiles and their gene sentences."""
    if not samples:
        logging.warning("No samples were collected, skipping visual QC artifact generation.")
        return

    num_to_display = min(num_samples, len(samples))
    if num_to_display == 0:
        return
        
    grid_size = math.ceil(math.sqrt(num_to_display))
    
    first_tile = samples[0]['tile']
    tile_w, tile_h = first_tile.size
    
    cell_w, cell_h = tile_w + 20, tile_h + 90  # Extra space for padding and text
    grid_img = Image.new("RGB", (grid_size * cell_w, grid_size * cell_h), "#F0F0F0")
    
    try:
        # Try a common system font, fallback to default
        font = ImageFont.truetype("DejaVuSans.ttf", 10)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 11)
    except IOError:
        font = ImageFont.load_default()
        font_bold = font
    
    draw = ImageDraw.Draw(grid_img)

    for i, sample in enumerate(samples[:num_to_display]):
        row, col = divmod(i, grid_size)
        x_offset, y_offset = col * cell_w, row * cell_h

        # Paste tile with a small border
        grid_img.paste(sample['tile'], (x_offset + 10, y_offset + 10))
        
        # Draw spot ID
        draw.text(
            (x_offset + 10, y_offset + tile_h + 20),
            f"Spot ID: {sample['id']}", fill="black", font=font_bold)
        
        # Draw wrapped gene sentence
        wrapped_text = textwrap.fill(f"Top Genes: {sample['sentence']}", width=45)
        draw.multiline_text(
            (x_offset + 10, y_offset + tile_h + 35),
            wrapped_text, fill="#555555", font=font)

    artifact_path = Path(output_dir) / "qc_sample_grid.png"
    try:
        grid_img.save(artifact_path)
        logging.info(f"🖼️ QC visual artifact with {num_to_display} samples saved to: {artifact_path}")
    except Exception as e:
        logging.error(f"Failed to save QC visual artifact: {e}")

def display_visual_artifact_notebook(artifact_path: str):
    """Displays the visual artifact image directly in a Jupyter notebook."""
    if not (plt and display):
        logging.warning("Matplotlib or IPython not found. Cannot display image in this environment.")
        return
        
    try:
        img = Image.open(artifact_path)
        plt.figure(figsize=(12, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Visual QC Samples from {Path(artifact_path).name}")
        plt.show()
    except FileNotFoundError:
        logging.error(f"Artifact file not found at {artifact_path}. Cannot display.")
    except Exception as e:
        logging.error(f"Error displaying visual artifact: {e}")
EOF

cat << 'EOF' > src/spaglam_preproc/config.py
# spaglam_preproc/config.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class PathConfig:
    """Configuration for all input and output paths."""
    adata_path: str
    output_dir: str
    # Image can be a WSI file, a standard image file, or loaded from adata.
    # If None, the pipeline will attempt to load the image from adata.uns['spatial'].
    image_path: Optional[str] = None
    # Optional path to a pre-computed list of highly variable genes (one gene per line or CSV column).
    hvg_list_path: Optional[str] = None

@dataclass
class PreprocessingConfig:
    """Parameters for data transformation and graph construction."""
    neighborhood_hops: int = 2
    n_top_genes_in_sentence: int = 50
    tile_size: int = 224
    precompute_embeddings: bool = True

@dataclass
class ModelConfig:
    """Model parameters, only required if precompute_embeddings is True."""
    model_path: str = "path/to/your/omiclip_model.pt"
    model_name: str = "ViT-B-32"

@dataclass
class QualityControlConfig:
    """Configuration for quality control, logging, and visualization."""
    enabled: bool = True
    num_visual_samples: int = 16  # Number of samples to include in the visual grid
    log_file_name: str = "preprocessing.log" # Name for the detailed log file

@dataclass
class PerformanceConfig:
    """Parameters to control performance and parallelization."""
    max_workers: int = 32
    max_samples_per_shard: int = 10000
    # Process a subset for quick testing. Set to -1 to process all spots.
    num_spots_to_process: int = -1 

@dataclass
class MainConfig:
    """Root configuration object that nests all other configurations."""
    paths: PathConfig
    preprocessing: PreprocessingConfig
    performance: PerformanceConfig
    qc: QualityControlConfig = field(default_factory=QualityControlConfig)
    # The model config is optional and only needed for one mode.
    model: Optional[ModelConfig] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MainConfig':
        """Creates a MainConfig object from a dictionary, handling nested structures."""
        # This allows for easy loading from a parsed YAML file.
        return cls(
            paths=PathConfig(**config_dict['paths']),
            preprocessing=PreprocessingConfig(**config_dict['preprocessing']),
            performance=PerformanceConfig(**config_dict['performance']),
            qc=QualityControlConfig(**config_dict.get('qc', {})),
            model=ModelConfig(**config_dict['model']) if 'model' in config_dict else None
        )
EOF

cat << 'EOF' > src/spaglam_preproc/cli.py
# spaglam_preproc/cli.py

import json
import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console

from .core.dataset_writer import create_dataset_shards
from .utils.logging_setup import setup_logging

app = typer.Typer(
    name="spaglam-preproc",
    help="A high-performance, single-pass preprocessing pipeline for SpaGLaM.",
    add_completion=False,
)
console = Console()

@app.command()
def run(
    config_path: Path = typer.Option(
        ..., 
        "--config", 
        "-c",
        help="Path to the YAML configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    )
):
    """
    Run the full SpaGLaM data preprocessing pipeline using a configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[bold red]Error parsing config file '{config_path}':[/bold red] {e}")
        raise typer.Exit(code=1)
    
    # Ensure output directory exists before setting up logging
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file and console
    log_file_name = config.get('qc', {}).get('log_file_name', 'preprocessing.log')
    log_path = output_dir / log_file_name
    setup_logging(str(log_path))
    
    # Log the configuration for reproducibility
    logging.info("🚀 Starting SpaGLaM preprocessing pipeline with configuration:")
    # Pretty print the config to the log file
    logging.info("\n" + json.dumps(config, indent=2))
    
    try:
        create_dataset_shards(config)
        console.print(f"\n[bold green]✅ Pipeline finished successfully! Check outputs in '{output_dir}'.[/bold green]")
    except Exception:
        # The rich handler will automatically log the traceback
        logging.error("Pipeline failed with an unhandled exception.")
        console.print(f"\n[bold red]❌ Pipeline failed. See full traceback in the log file: {log_path}[/bold red]")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
EOF

echo "✅ Utils, Config, and CLI files created."

# --- Create Package __init__ files ---

cat << 'EOF' > src/spaglam_preproc/__init__.py
# spaglam_preproc/__init__.py

__version__ = "0.1.0"

# Expose the main user-facing classes and functions for easy import
# This allows users to write `from spaglam_preproc import SpaglamPipeline`
from .core.dataset_writer import SpaglamPipeline, create_dataset_shards
from .core.image_tiler import ImageHandler

__all__ = [
    "SpaglamPipeline",
    "create_dataset_shards",
    "ImageHandler",
    "__version__",
]
EOF

touch src/spaglam_preproc/core/__init__.py
touch src/spaglam_preproc/utils/__init__.py
echo "✅ Package __init__ files created."

# --- Create/Overwrite Project Root Files ---

cat << 'EOF' > pyproject.toml
# pyproject.toml

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "GoldSpace"
version = "0.1.0"
authors = [
  { name="jijh", email="your.email@example.com" },
]
description = "A research project for SpaGLaM, including open_clip_training and a dedicated preprocessing pipeline."
readme = "README.md"
requires-python = ">=3.9"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[tool.setuptools.packages.find]
where = ["src"]

[project.optional-dependencies]
# Dependencies for the SpaGLaM preprocessing package
preproc_models = ["open-clip-torch"]
preproc_wsi = ["openslide-python"]
preproc_squidpy = ["squidpy"]
preproc_notebook = ["matplotlib", "ipython", "jupyter"]
preproc_all = [
    "spaglam-preproc[preproc_models]",
    "spaglam-preproc[preproc_wsi]",
    "spaglam-preproc[preproc_squidpy]",
    "spaglam-preproc[preproc_notebook]",
]

[project.scripts]
spaglam-preproc = "spaglam_preproc.cli:app"
EOF

cat << 'EOF' > README.md
# GoldSpace Project

This repository contains the source code for the SpaGLaM (Spatial Graph Large Model) project, including the training framework based on `open_clip` and a new, high-performance preprocessing pipeline.

## `spaglam-preproc`: The Preprocessing Pipeline

A high-performance, single-pass data preprocessing pipeline designed for SpaGLaM. This tool efficiently converts spatial transcriptomics data (AnnData and histology images) into graph-based `webdataset` shards suitable for large-scale model training.

### Features

-   **High-Performance Single Pass**: Extracts image tiles and generates gene sentences on-the-fly, eliminating the I/O bottleneck of writing and reading millions of intermediate files.
-   **Flexible Output**: Generate `webdataset` shards containing either raw data (`.png`, `.txt`) or pre-computed OmiCLIP embeddings (`.pth`), controlled by a simple config flag.
-   **Versatile Image Support**: Natively handles Whole-Slide Images (e.g., `.svs`, `.tif`), standard images (`.png`, `.jpeg`), and images embedded in `AnnData` objects.
-   **Robust Quality Control**: Includes pre-run validation checks, live progress monitoring, and automatically generates a final QC report and a visual sample grid for easy verification.
-   **User-Friendly Interface**: A simple Command-Line Interface (CLI) driven by a clean YAML configuration file.
-   **Notebook-Ready**: The core pipeline is encapsulated in a class, allowing for easy, interactive use and visualization within Jupyter notebooks.

### Installation

It is recommended to install the project in editable mode. From the `GoldSpace` root directory:

**1. Basic Installation (for training with existing data):**

Install the base dependencies for training.
```bash
pip install -e .
# You may also need to install from your requirements files
pip install -r requirements.txt
