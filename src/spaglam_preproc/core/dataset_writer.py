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

            # ===== SOTA MODIFICATION START =====
            # Instead of saving 2*N files per sample, we save ONE file containing all embeddings.
            # This is a massive I/O performance improvement for the data loader.
            
            embeddings_dict = {
                'image': image_embeddings, # Shape: [num_nodes, embed_dim]
                'text': text_embeddings,   # Shape: [num_nodes, embed_dim]
            }
            emb_buf = io.BytesIO()
            torch.save(embeddings_dict, emb_buf)
            output_sample["embeddings.pth"] = emb_buf.getvalue()
            # ===== SOTA MODIFICATION END =====

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
