
# ===== File: preprocess_src/embedd_graph_dataset_construct.py =====

# SOTA_generate_embeddings.py
#
# PURPOSE:
# A high-performance, one-pass script to generate OmiCLIP embedding shards directly
# from raw data files and an AnnData spatial graph.
#
# FEATURES:
# - Single-pass efficiency: Raw Files -> Embedding Shards (No intermediate files).
# - Configurable K-Hop Neighborhoods: Specify 1-hop, 2-hop, etc.
# - Debug-friendly Sampling: Easily switch to process a small subset of data.
# - SOTA Engineering: Modular, robust, highly parallelized, and clear.

import os
import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from collections import deque
import glob

import torch
import pandas as pd
import scanpy as sc
import webdataset as wds
from PIL import Image
from tqdm import tqdm
from scipy.sparse import csr_matrix

try:
    import open_clip
except ImportError:
    print("FATAL: open_clip library not found. Please ensure it's installed and in your PYTHONPATH.")
    exit(1)

# ==============================================================================
# STEP 1: SCRIPT CONFIGURATION
# ==============================================================================
CONFIG = {
    # --- Input Data ---
    "adata_path": "/cwStorage/nodecw_group/jijh/spaglam_sota_data/master_adata_with_graph_and_paths.h5ad",
    "tiles_base_dir": "/cwStorage/nodecw_group/jijh/hest_output",
    "sentences_base_dir": "/cwStorage/nodecw_group/jijh/hest_sentences_human_all",

    # --- Output ---
    "output_dir": "/cwStorage/nodecw_group/jijh/spaglam_embedding_shards_direct_k_hop",

    # --- Graph & Sampling Control ---
    "neighborhood_hops": 2,  # <<< Set the number of hops (e.g., 1, 2, 3)
    "sampling_config": {
        "enabled": False,     # <<< Set to True to enable sampling for quick tests
        "num_samples": 1000, # Number of spots to process if sampling is enabled
    },

    # --- Model ---
    "model_path": "/cwStorage/nodecw_group/jijh/training_log/cfff_train/openclip_train/train_log/finetune_20250623-092915/2025_06_23-09_30_48-model_ViT-B-32-lr_2e-05-b_1800-j_16-p_amp_bfloat16/checkpoints/epoch_25.pt",
    "model_name": "ViT-B-32",

    # --- Performance ---
    "max_samples_per_shard": 10000,
    "max_workers": 32, # CPU workers for parallel I/O and data prep
}

# ==============================================================================
# STEP 2: CORE LOGIC & HELPER FUNCTIONS
# ==============================================================================

def get_k_hop_neighborhood(
    adjacency_matrix: csr_matrix, start_node_idx: int, k: int
) -> list[int]:
    """
    Finds all unique nodes within k hops of a starting node using Breadth-First Search (BFS).
    This is the most efficient method for this task.

    Args:
        adjacency_matrix: The sparse adjacency matrix (from adata.obsp).
        start_node_idx: The integer index of the starting node.
        k: The number of hops.

    Returns:
        A list of unique integer indices of all nodes in the k-hop neighborhood.
    """
    if k == 0:
        return [start_node_idx]

    visited = {start_node_idx}
    queue = deque([start_node_idx])
    
    for _ in range(k):
        # Process all nodes at the current level
        level_size = len(queue)
        if level_size == 0:
            break
        for _ in range(level_size):
            current_node = queue.popleft()
            # Get neighbors using the efficient .indices attribute of CSR matrices
            neighbors = adjacency_matrix[current_node].indices
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return list(visited)


def process_subgraph_to_embeddings(
    center_spot_id: str,
    *, # Enforce keyword-only arguments for clarity
    adata: sc.AnnData,
    adjacency_matrix: csr_matrix,
    model: torch.nn.Module,
    image_preprocessor,
    tokenizer,
    device: str,
    config: dict
):
    """
    Worker function for a single center spot. It performs the entire pipeline in memory:
    1. Finds the k-hop neighborhood.
    2. Builds the local subgraph and its edge index.
    3. Loads all required image and text files from disk.
    4. Encodes data using the GPU model.
    5. Serializes and returns the final sample dictionary for the ShardWriter.
    """
    try:
        # a. Get k-hop neighborhood using BFS
        center_node_idx = adata.obs_names.get_loc(center_spot_id)
        k = config["neighborhood_hops"]
        global_indices = get_k_hop_neighborhood(adjacency_matrix, center_node_idx, k)
        all_spot_ids = adata.obs_names[global_indices].tolist()
        num_nodes = len(all_spot_ids)

        # b. Build the local graph structure with re-indexed nodes
        global_id_to_local_idx = {spot_id: i for i, spot_id in enumerate(all_spot_ids)}
        local_edge_index = []
        for local_u_idx, u_spot_id in enumerate(all_spot_ids):
            u_global_idx = adata.obs_names.get_loc(u_spot_id)
            for v_global_idx in adjacency_matrix[u_global_idx].indices:
                v_spot_id = adata.obs_names[v_global_idx]
                if v_spot_id in global_id_to_local_idx:
                    local_v_idx = global_id_to_local_idx[v_spot_id]
                    # Add edge only once to avoid duplicates
                    if local_u_idx < local_v_idx:
                        local_edge_index.append([local_u_idx, local_v_idx])

        # c. Load all raw data for the subgraph from disk
        images_to_process = []
        texts_to_process = []
        for spot_id in all_spot_ids:
            meta = adata.obs.loc[spot_id]
            sid, row, col = meta['sample_id'], meta['pxl_row_in_fullres'], meta['pxl_col_in_fullres']
            suffix = f"{int(round(row))}_{int(round(col))}" if pd.notna(row) and pd.notna(col) else f"{adata.obs_names.get_loc(spot_id)}_0_no_coord"
            
            tile_path = os.path.join(config["tiles_base_dir"], f"{sid}_tiles", f"{sid}_{suffix}.png")
            sentence_path = os.path.join(config["sentences_base_dir"], f"{sid}_sentences_hvg", f"{sid}_{suffix}.txt")
            
            images_to_process.append(Image.open(tile_path).convert("RGB"))
            with open(sentence_path, 'r', encoding='utf-8') as f:
                texts_to_process.append(f.read())
        
        # d. Preprocess, encode on GPU, and move results to CPU
        image_input = torch.stack([image_preprocessor(img) for img in images_to_process])
        text_input = tokenizer(texts_to_process)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_input, text_input = image_input.to(device), text_input.to(device)
            image_embeddings = model.encode_image(image_input).cpu()
            text_embeddings = model.encode_text(text_input).cpu()

        # e. Construct the final sample for sharding
        output_sample = {
            "__key__": center_spot_id,
            "json": json.dumps({"num_nodes": num_nodes, "edge_index": local_edge_index}).encode('utf-8')
        }
        for i in range(num_nodes):
            img_buf, txt_buf = io.BytesIO(), io.BytesIO()
            torch.save(image_embeddings[i], img_buf)
            torch.save(text_embeddings[i], txt_buf)
            output_sample[f"{i}.image.pth"] = img_buf.getvalue()
            output_sample[f"{i}.text.pth"] = txt_buf.getvalue()
        
        return output_sample, None

    except Exception as e:
        return None, f"Skipping {center_spot_id}: {type(e).__name__} - {e}"

# ==============================================================================
# STEP 3: MAIN EXECUTION ORCHESTRATOR
# ==============================================================================

def main(config: dict):
    """Orchestrates the entire high-performance embedding generation pipeline."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    start_time = time.time()
    
    # --- Log Configuration for Reproducibility ---
    logging.info("=" * 80)
    logging.info("🚀 Starting Direct Embedding Generation Pipeline (SOTA Version)")
    logging.info(f"   - Neighborhood Hops: {config['neighborhood_hops']}")
    logging.info(f"   - Sampling Enabled: {config['sampling_config']['enabled']}")
    if config['sampling_config']['enabled']:
        logging.info(f"   - Samples to Process: {config['sampling_config']['num_samples']}")
    logging.info(f"   - Output Directory: {config['output_dir']}")
    logging.info("=" * 80)
    
    # --- Setup ---
    os.makedirs(config["output_dir"], exist_ok=True)
    shards_pattern = os.path.join(config["output_dir"], "shard-%06d.tar")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Load Global Resources ---
    logging.info(f"🔌 Using device: {device}")
    logging.info("🔧 Loading OmiCLIP model...")
    model, _, image_preprocessor = open_clip.create_model_and_transforms(
        config['model_name'], pretrained=config['model_path'], device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(config['model_name'])
    
    logging.info("💾 Loading AnnData and spatial graph...")
    adata = sc.read_h5ad(config['adata_path'])
    # Ensure graph is in CSR format for fast row slicing
    adjacency_matrix = adata.obsp['spatial_connectivities'].tocsr()
    
    # --- Determine Spots to Process (with sampling logic) ---
    all_spot_ids = adata.obs.index.tolist()
    if config["sampling_config"]["enabled"]:
        num_to_sample = config["sampling_config"]["num_samples"]
        spots_to_process = all_spot_ids[:num_to_sample]
        logging.info(f"✅ Sampling enabled. Processing the first {len(spots_to_process)} spots.")
    else:
        spots_to_process = all_spot_ids
        logging.info(f"✅ Processing all {len(spots_to_process)} spots.")

    # --- Parallel Processing ---
    worker_fn = partial(
        process_subgraph_to_embeddings,
        adata=adata,
        adjacency_matrix=adjacency_matrix,
        model=model,
        image_preprocessor=image_preprocessor,
        tokenizer=tokenizer,
        device=device,
        config=config,
    )
    
    success_count, error_count = 0, 0
    with wds.ShardWriter(shards_pattern, maxcount=config['max_samples_per_shard']) as sink:
        with ThreadPoolExecutor(max_workers=config['max_workers']) as executor:
            futures = {executor.submit(worker_fn, spot_id): spot_id for spot_id in spots_to_process}
            
            pbar = tqdm(as_completed(futures), total=len(spots_to_process), desc="Generating Embeddings", unit="spot")
            for future in pbar:
                sample, error_msg = future.result()
                if sample:
                    sink.write(sample)
                    success_count += 1
                else:
                    error_count += 1
                    if error_count < 20: # Log first few errors for debugging
                        logging.warning(error_msg)
    
    # --- Final Summary ---
    elapsed_time = time.time() - start_time
    logging.info("\n" + "="*80)
    logging.info("🏁 Pipeline Finished!")
    logging.info(f"  - Successfully processed: {success_count} spots")
    logging.info(f"  - Skipped due to errors:  {error_count} spots")
    logging.info(f"  - Total time:             {elapsed_time / 60:.2f} minutes")
    logging.info(f"  - Avg. throughput:        {success_count / elapsed_time:.2f} spots/sec")
    logging.info(f"  - Output saved to:        {config['output_dir']}")
    logging.info("="*80)

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================
if __name__ == '__main__':
    main(CONFIG)
# ===== File: preprocess_src/gene_to_sentence.py =====

import os
import pandas as pd
import numpy as np
from scipy.sparse import issparse

def anndata_to_sentence_files_hvg(
    sample,
    hvg_list: pd.Index,
    output_basedir: str,
    n_top_genes_in_sentence: int = 50
):
    """
    为单个样本对象的所有spots生成独立的基因句子文件。
    该函数只考虑在高变基因（HVG）列表中的基因。

    参数:
        sample (object): 一个包含 .sample_id 和 .adata 属性的对象。
                         .sample_id 是样本的唯一标识符 (str)。
                         .adata 是该样本的 AnnData 对象。
        hvg_list (pd.Index): 全局高变基因列表。
        output_basedir (str): 保存所有句子文件夹的根目录。
        n_top_genes_in_sentence (int): 每个句子（文件）包含的基因数量。
    """
    # 1. 为当前样本创建一个独立的输出目录
    sample_sentence_dir = os.path.join(output_basedir, f"{sample.sample_id}_sentences_hvg")
    os.makedirs(sample_sentence_dir, exist_ok=True)
    
    if sample.adata is None:
        print(f"警告: 样本 {sample.sample_id} 的 AnnData 为空，已跳过。")
        return
        
    # 2. 筛选AnnData，使其只包含高变基因，这能极大加速后续处理
    adata = sample.adata
    adata_hvg = adata[:, adata.var_names.isin(hvg_list)].copy()
    
    if adata_hvg.n_vars == 0:
        print(f"警告: 样本 {sample.sample_id} 中没有找到任何全局HVGs，已跳过。")
        return

    # 3. 准备所需数据：坐标、基因名
    obs_df = adata_hvg.obs.copy()
    gene_names_hvg = adata_hvg.var_names.to_numpy()
    
    # 确保坐标存在，如果不存在，则使用索引作为后备
    if 'pxl_col_in_fullres' not in obs_df.columns or 'pxl_row_in_fullres' not in obs_df.columns:
        if 'spatial' in adata_hvg.obsm:
            obs_df['pxl_col_in_fullres'] = adata_hvg.obsm['spatial'][:, 0]
            obs_df['pxl_row_in_fullres'] = adata_hvg.obsm['spatial'][:, 1]
        else:
            # 如果连 'spatial' 都没有，就用 spot 的索引作为文件名
            print(f"警告: 样本 {sample.sample_id} 缺失坐标信息，将使用索引作为文件名。")
            obs_df['pxl_col_in_fullres'] = np.arange(adata_hvg.n_obs)
            obs_df['pxl_row_in_fullres'] = 0

    # 4. 【【【核心逻辑】】】遍历每个spot并生成文件
    # for i in tqdm(range(adata_hvg.n_obs), desc=f"生成句子: {sample.sample_id}", leave=False):
    for i in range(adata_hvg.n_obs):
        spot_info = obs_df.iloc[i]
        
        # --- 文件名生成 ---
        row_val = spot_info['pxl_row_in_fullres']
        col_val = spot_info['pxl_col_in_fullres']

        # 处理坐标可能为NaN的情况
        if pd.isna(row_val) or pd.isna(col_val):
            filename_suffix = f"{i}_0_no_coord"
        else:
            row_coord = int(round(row_val))
            col_coord = int(round(col_val))
            filename_suffix = f"{row_coord}_{col_coord}"
            
        sentence_filename = os.path.join(sample_sentence_dir, f"{sample.sample_id}_{filename_suffix}.txt")
        
        # --- 基因句子生成 ---
        # a. 获取当前spot在HVG子集上的表达向量
        expression_vector = adata_hvg.X[i].toarray().flatten() if issparse(adata_hvg.X) else np.array(adata_hvg.X[i]).flatten()
            
        # b. 对表达量进行排序，得到基因的索引（从高到低）
        #    np.argsort返回的是从小到大的索引，所以用 `[-1::-1]` 将其反转
        sorted_indices = np.argsort(expression_vector)[-1::-1]
        
        # c. 从排序后的索引中选择前N个，并获取对应的基因名
        top_n_indices = sorted_indices[:n_top_genes_in_sentence]
        top_genes = gene_names_hvg[top_n_indices]
        
        # d. 将基因名列表用空格连接成一个字符串，并写入文件
        with open(sentence_filename, 'w') as f:
            f.write(" ".join(top_genes))
# ===== File: preprocess_src/tiling_code.py =====

import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from scipy.spatial import cKDTree
import concurrent.futures
import matplotlib.pyplot as plt
import random
import logging

# 配置 logging 模块
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def extract_tiles_from_image(image,
                             coordinates,
                             output_dir,
                             base_filename="tile",
                             tile_size=None, 
                             tile_size_strategy="median",
                             max_workers=8):
    """
    从一个大图像中根据给定的坐标列表提取图像瓦片。

    该函数是通用的，不依赖于特定的数据结构。它只需要一个图像对象
    （需支持 .dimensions 和 .read_region 接口）和 N x 2 的坐标数组。

    瓦片大小的自动计算逻辑:
      - 若 tile_size 为 None，则根据坐标构建 KD 树，并计算每个点到其最近邻的距离。
      - tile_size_strategy ("min", "mean", "median", "max") 决定使用哪个统计量。

    瓦片边界处理策略:
      - 以每个坐标为中心，构造正方形区域。如果区域超出图像边界，
        仅提取有效部分并将其粘贴到与瓦片大小相同的白色背景上。

    参数:
      image (object): 图像对象。需要具备以下接口：
                      - .dimensions: 一个返回 (width, height) 元组的属性。
                      - .read_region((x, y), level, (w, h)): 一个方法，用于读取图像区域。
                        (level 通常为 0，表示最高分辨率)。
      coordinates (np.ndarray): 一个 shape 为 (N, 2) 的 NumPy 数组，
                                包含 N 个点的 (x, y) 坐标。
      output_dir (str): 保存瓦片图像的目录。
      base_filename (str): 输出瓦片文件名的前缀。
      tile_size (int or None): 瓦片边长（像素）。为 None 时自动计算。
      tile_size_strategy (str): 自动计算瓦片大小时的策略。
      max_workers (int): 并发处理的线程数。

    返回:
      list[str]: 一个包含所有成功生成的瓦片文件路径的列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证坐标数据
    if not isinstance(coordinates, np.ndarray) or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        logging.error("坐标必须是 (N, 2) 的 NumPy 数组。")
        return []
    if coordinates.shape[0] == 0:
        logging.warning("坐标列表为空，无需处理。")
        return []

    # 1. 如果 tile_size 为 None，则自动计算
    if tile_size is None:
        if len(coordinates) < 2:
            logging.warning("坐标点少于2个，无法自动计算瓦片大小。请手动指定 `tile_size`。")
            return []
            
        tree = cKDTree(coordinates)
        dists, _ = tree.query(coordinates, k=2)  # k=2 找到自身和最近的邻居
        nn_dists = dists[:, 1]  # 提取最近邻距离
        
        if tile_size_strategy == "min":
            tile_size = int(round(np.min(nn_dists)))
        elif tile_size_strategy == "mean":
            tile_size = int(round(np.mean(nn_dists)))
        elif tile_size_strategy == "max":
            tile_size = int(round(np.max(nn_dists)))
        else:  # 默认 "median"
            tile_size = int(round(np.median(nn_dists)))
        logging.info(f"自动计算瓦片大小: {tile_size} 像素 (策略: {tile_size_strategy})")
    
    full_width, full_height = image.dimensions

    # 2. 筛选有效坐标点
    valid_spots_args = []
    for i, (col_coord_float, row_coord_float) in enumerate(coordinates):
        col_coord = int(round(col_coord_float))
        row_coord = int(round(row_coord_float))
        half_tile = tile_size // 2
        top_left_x = col_coord - half_tile
        top_left_y = row_coord - half_tile
        
        # 判断是否完全出界
        if (top_left_x + tile_size) > 0 and (top_left_y + tile_size) > 0 and \
           top_left_x < full_width and top_left_y < full_height:
            valid_spots_args.append((i, col_coord, row_coord, top_left_x, top_left_y))
    
    logging.info(f"总坐标点数: {len(coordinates)}, 有效点数: {len(valid_spots_args)}")

    # 3. 定义并发处理函数
    def process_tile(args):
        idx, col, row, top_left_x, top_left_y = args
        
        # 构造文件名，使用坐标作为唯一标识
        tile_filename = os.path.join(output_dir, f"{base_filename}_{row}_{col}.png")
        
        if os.path.exists(tile_filename):
            return tile_filename
        
        # 计算在 WSI 内的有效读取区域
        read_left = max(top_left_x, 0)
        read_top = max(top_left_y, 0)
        read_right = min(top_left_x + tile_size, full_width)
        read_bottom = min(top_left_y + tile_size, full_height)
        read_width = read_right - read_left
        read_height = read_bottom - read_top
        
        # 如果计算出的区域无效，则跳过
        if read_width <= 0 or read_height <= 0:
            return None

        try:
            # 从大图像中读取指定区域 (level=0 表示最高分辨率)
            region = image.read_region((read_left, read_top), 0, (read_width, read_height))
            region = region.convert("RGB")
        except Exception as e:
            logging.error(f"读取图像区域失败，坐标 ({col}, {row})。错误: {e}", exc_info=True)
            return None
        
        # 创建白色背景瓦片
        tile_img = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        
        # 计算粘贴位置
        paste_x = read_left - top_left_x
        paste_y = read_top - top_left_y
        tile_img.paste(region, (paste_x, paste_y))
        
        try:
            tile_img.save(tile_filename)
        except Exception as e:
            logging.error(f"保存瓦片失败 {tile_filename}。错误: {e}", exc_info=True)
            return None
            
        return tile_filename

    # 4. 使用线程池并发执行
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(process_tile, args): args for args in valid_spots_args}
        for future in tqdm(concurrent.futures.as_completed(future_map), total=len(valid_spots_args), desc=f"提取瓦片"):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # 5. 随机展示部分结果
    if results:
        sample_files = random.sample(results, min(8, len(results)))
        n_samples = len(sample_files)
        fig, axes = plt.subplots(1, n_samples, figsize=(n_samples * 2.5, 2.5))
        if n_samples == 1:
            axes = [axes]
            
        for ax, file_path in zip(axes, sample_files):
            try:
                img = Image.open(file_path)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(os.path.basename(file_path).replace('.png', ''), fontsize=8)
            except Exception as e:
                logging.error(f"展示瓦片失败: {file_path}, 错误: {e}")
        plt.tight_layout()
        plt.show()
    
    return results


# --- 使用示例 ---
# 为了让代码可以独立运行，我们创建一个模拟的图像对象和坐标数据
if __name__ == '__main__':

    # 1. 创建一个模拟的 WSI 对象，它满足 `image` 参数的要求
    class MockWSI:
        def __init__(self, width, height):
            self.dimensions = (width, height)
            # 创建一个渐变色的模拟 WSI 背景以便观察提取的位置
            r, g = np.mgrid[0:255:height*1j, 0:255:width*1j]
            b = np.zeros_like(r)
            self._image_array = np.stack([r, g, b], axis=-1).astype(np.uint8)
            self._image = Image.fromarray(self._image_array)
            print(f"创建了一个 {width}x{height} 的模拟图像。")

        def read_region(self, location, level, size):
            # `level` 在此模拟中被忽略
            left, top = location
            width, height = size
            return self._image.crop((left, top, left + width, top + height))

    # 2. 定义参数
    IMG_WIDTH, IMG_HEIGHT = 4000, 3000
    NUM_SPOTS = 500
    OUTPUT_DIR = "generic_tiles_output"
    
    # 3. 创建模拟数据
    mock_image_obj = MockWSI(IMG_WIDTH, IMG_HEIGHT)
    
    # 随机生成在图像范围内的坐标点
    mock_coordinates = np.random.rand(NUM_SPOTS, 2) * np.array([IMG_WIDTH, IMG_HEIGHT])

    print(f"\n开始使用 {NUM_SPOTS} 个随机坐标进行瓦片提取...")
    
    # 4. 调用通用的瓦片提取函数
    generated_files = extract_tiles_from_image(
        image=mock_image_obj,
        coordinates=mock_coordinates,
        output_dir=OUTPUT_DIR,
        base_filename="sampleA",
        tile_size=None,  # 让函数自动计算最佳瓦片大小
        tile_size_strategy="median",
        max_workers=8
    )

    print(f"\n处理完成！共生成 {len(generated_files)} 个瓦片图像。")
    if generated_files:
        print("文件保存在目录:", os.path.abspath(OUTPUT_DIR))
        print("部分生成的文件名示例:")
        for f in generated_files[:5]:
            print(f" - {os.path.basename(f)}")
# ===== File: src/open_clip_train/__init__.py =====


# ===== File: src/open_clip_train/data_defs.py =====

# ===== 文件: src/open_clip_train/data_defs.py (新创建) =====

from dataclasses import dataclass
from torch.utils.data import DataLoader, DistributedSampler
from multiprocessing import Value

@dataclass
class DataInfo:
    """
    一个用于封装 Dataloader 及其相关信息的简单数据类。
    """
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: 'SharedEpoch' = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

class SharedEpoch:
    """
    一个使用多进程安全值的类，用于在 Dataloader 的工作进程之间同步 epoch 计数。
    """
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value
# ===== File: src/open_clip_train/data.py =====

# ===== 文件: src/open_clip_train/data.py (已修正) =====

import ast
import json
import logging
import math
import os
import random
import sys
import braceexpand

import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

# 【核心修复】: 从新的 data_defs.py 导入公共类，打破循环
from .data_defs import DataInfo, SharedEpoch
from .spaglam_data import get_spaglam_dataset

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts

# --- DataInfo 和 SharedEpoch 的定义已移至 data_defs.py ---

def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split('::')
        assert len(weights) == len(urllist),\
            f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, preprocess_fns, split):
    assert split in ["train", "val", "v2"]
    is_train = split == "train"
    preprocess_train, preprocess_val = preprocess_fns

    if split == "v2":
        from imagenetv2_pytorch import ImageNetV2Dataset
        dataset = ImageNetV2Dataset(location=args.imagenet_v2, transform=preprocess_val)
    else:
        data_path = args.imagenet_train if is_train else args.imagenet_val
        preprocess_fn = preprocess_train if is_train else preprocess_val
        assert data_path
        dataset = datasets.ImageFolder(data_path, transform=preprocess_fn)

    if is_train:
        # ... (sampler logic remains the same)
        idxs = np.zeros(len(dataset.targets))
        target_array = np.array(dataset.targets)
        k = 50
        for c in range(1000):
            m = target_array == c
            n = len(idxs[m])
            arr = np.zeros(n)
            arr[:k] = 1
            np.random.shuffle(arr)
            idxs[m] = arr
        idxs = idxs.astype('int')
        sampler = SubsetRandomSampler(np.where(idxs)[0])
    else:
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, sampler=sampler)

    return DataInfo(dataloader=dataloader, sampler=sampler)


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed(increment=0):
    worker_info = get_worker_info()
    if worker_info is not None:
        seed = worker_info.seed
        if increment:
            seed += increment * max(1, worker_info.num_workers)
        return seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(self, bufsize=1000, initial=100, seed=0, epoch=-1):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed(epoch)
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    def __init__(self, urls, weights=None, nshards=sys.maxsize, worker_seed=None, deterministic=False, epoch=-1):
        super().__init__()
        urls, weights = expand_urls(urls, weights)
        self.urls = urls
        self.weights = weights
        if self.weights is not None:
            assert len(self.urls) == len(self.weights)
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            if self.worker_seed is None:
                seed = pytorch_worker_seed(epoch)
            else:
                seed = self.worker_seed() + epoch
            self.rng.seed(seed)
        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    # (This function remains unchanged)
    # ...
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled

    if resampled:
        pipeline = [ResampledShards2(
            input_shards, weights=args.train_data_upsampling_factors, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(bufsize=_SHARD_SHUFFLE_SIZE, initial=_SHARD_SHUFFLE_INITIAL, seed=args.seed, epoch=shared_epoch),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            tarfile_to_samples_nothrow,
            wds.shuffle(bufsize=_SAMPLE_SHUFFLE_SIZE, initial=_SAMPLE_SHUFFLE_INITIAL),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
        ])

    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)
    else:
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset, batch_size=None, shuffle=False, num_workers=args.workers, persistent_workers=args.workers > 0)

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    # (This function remains unchanged)
    # ...
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename, preprocess_fn, img_key=args.csv_img_key,
        caption_key=args.csv_caption_key, sep=args.csv_separator, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=True, sampler=sampler, drop_last=is_train)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SyntheticDataset(Dataset):
    # (This class remains unchanged)
    # ...
    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, tokenizer=None):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size
        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    # (This function remains unchanged)
    # ...
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
        pin_memory=True, sampler=sampler, drop_last=is_train)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "spaglam":
        return get_spaglam_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data
# ===== File: src/open_clip_train/distributed.py =====

import os
import warnings
from typing import Optional

import torch
import torch.distributed as dist

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def is_global_master(args):
    return args.rank == 0


def is_local_master(args):
    return args.local_rank == 0


def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)


def is_device_available(device):
    device_type = torch.device(device).type
    is_avail = False
    is_known = False
    if device_type == 'cuda':
        is_avail = torch.cuda.is_available()
        is_known = True
    elif device_type == 'npu':
        # NOTE autoload device extension needed for this not to error out on this check
        is_avail = torch.npu.is_available()
        is_known = True
    elif device_type == 'mps':
        is_avail = torch.backends.mps.is_available()
        is_known = True
    elif device_type == 'cpu':
        is_avail = True
        is_known = True

    return is_avail, is_known


def set_device(device):
    if device.startswith('cuda:'):
        torch.cuda.set_device(device)
    elif device.startswith('npu:'):
        torch.npu.set_device(device)


def is_using_horovod():
    # NOTE w/ horovod run, OMPI vars should be set, but w/ SLURM PMI vars will be set
    # Differentiating between horovod and DDP use via SLURM may not be possible, so horovod arg still required...
    ompi_vars = ["OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"]
    pmi_vars = ["PMI_RANK", "PMI_SIZE"]
    if all([var in os.environ for var in ompi_vars]) or all([var in os.environ for var in pmi_vars]):
        return True
    else:
        return False


def is_using_distributed():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    result = init_distributed_device_so(
        device=getattr(args, 'device', 'cuda'),
        dist_backend=getattr(args, 'dist_backend', None),
        dist_url=getattr(args, 'dist_url', None),
        horovod=getattr(args, 'horovod', False),
        no_set_device_rank=getattr(args, 'no_set_device_rank', False),
    )
    args.device = result['device']
    args.world_size = result['world_size']
    args.rank = result['global_rank']
    args.local_rank = result['local_rank']
    args.distributed = result['distributed']
    device = torch.device(args.device)
    return device


def init_distributed_device_so(
        device: str = 'cuda',
        dist_backend: Optional[str] = None,
        dist_url: Optional[str] = None,
        horovod: bool = False,
        no_set_device_rank: bool = False,
):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    global_rank = 0
    local_rank = 0
    device_type, *device_idx = device.split(':', maxsplit=1)
    is_avail, is_known = is_device_available(device_type)
    if not is_known:
        warnings.warn(f"Device {device} was not known and checked for availability, trying anyways.")
    elif not is_avail:
        warnings.warn(f"Device {device} was not available, falling back to CPU.")
        device_type = device = 'cpu'

    if horovod:
        import horovod.torch as hvd
        assert hvd is not None, "Horovod is not installed"
        hvd.init()
        local_rank = int(hvd.local_rank())
        global_rank = hvd.rank()
        world_size = hvd.size()
        distributed = True
    elif is_using_distributed():
        if dist_backend is None:
            dist_backends = {
                "cuda": "nccl",
                "hpu": "hccl",
                "npu": "hccl",
                "xpu": "ccl",
            }
            dist_backend = dist_backends.get(device_type, 'gloo')

        dist_url = dist_url or 'env://'

        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            local_rank, global_rank, world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(global_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=world_size,
                rank=global_rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
            )
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
        distributed = True

    if distributed and not no_set_device_rank and device_type not in ('cpu', 'mps'):
        # Ignore manually specified device index in distributed mode and
        # override with resolved local rank, fewer headaches in most setups.
        if device_idx:
            warnings.warn(f'device index {device_idx[0]} removed from specified ({device}).')
        device = f'{device_type}:{local_rank}'
        set_device(device)

    return dict(
        device=device,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
    )


def broadcast_object(args, obj, src=0):
    # broadcast a pickle-able python object from rank-0 to all ranks
    if args.horovod:
        return hvd.broadcast_object(obj, root_rank=src)
    else:
        if args.rank == src:
            objects = [obj]
        else:
            objects = [None]
        dist.broadcast_object_list(objects, src=src)
        return objects[0]


def all_gather_object(args, obj, dst=0):
    # gather a pickle-able python object across all ranks
    if args.horovod:
        return hvd.allgather_object(obj)
    else:
        objects = [None for _ in range(args.world_size)]
        dist.all_gather_object(objects, obj)
        return objects

# ===== File: src/open_clip_train/file_utils.py =====

import logging
import os
import multiprocessing
import subprocess
import time
import fsspec
import torch
from tqdm import tqdm

def remote_sync_s3(local_dir, remote_dir):
    # skip epoch_latest which can change during sync.
    result = subprocess.run(["aws", "s3", "sync", local_dir, remote_dir, '--exclude', '*epoch_latest.pt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(f"Error: Failed to sync with S3 bucket {result.stderr.decode('utf-8')}")
        return False
        
    logging.info(f"Successfully synced with S3 bucket")
    return True

def remote_sync_fsspec(local_dir, remote_dir):
    # FIXME currently this is slow and not recommended. Look into speeding up.
    a = fsspec.get_mapper(local_dir)
    b = fsspec.get_mapper(remote_dir)

    for k in a:
        # skip epoch_latest which can change during sync.
        if 'epoch_latest.pt' in k:
            continue

        logging.info(f'Attempting to sync {k}')
        if k in b and len(a[k]) == len(b[k]):
            logging.debug(f'Skipping remote sync for {k}.')
            continue

        try:
            logging.info(f'Successful sync for {k}.')
            b[k] = a[k]
        except Exception as e:
            logging.info(f'Error during remote sync for {k}: {e}')
            return False

    return True

def remote_sync(local_dir, remote_dir, protocol):
    logging.info('Starting remote sync.')
    if protocol == 's3':
        return remote_sync_s3(local_dir, remote_dir)
    elif protocol == 'fsspec':
        return remote_sync_fsspec(local_dir, remote_dir)
    else:
        logging.error('Remote protocol not known')
        return False

def keep_running_remote_sync(sync_every, local_dir, remote_dir, protocol):
    while True:
        time.sleep(sync_every)
        remote_sync(local_dir, remote_dir, protocol)

def start_sync_process(sync_every, local_dir, remote_dir, protocol):
    p = multiprocessing.Process(target=keep_running_remote_sync, args=(sync_every, local_dir, remote_dir, protocol))
    return p

# Note: we are not currently using this save function.
def pt_save(pt_obj, file_path):
    of = fsspec.open(file_path, "wb")
    with of as f:
        torch.save(pt_obj, file_path)

def pt_load(file_path, map_location=None):
    if file_path.startswith('s3'):
        logging.info('Loading remote checkpoint, which may take a bit.')
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out

def check_exists(file_path):
    try:
        with fsspec.open(file_path):
            pass
    except FileNotFoundError:
        return False
    return True

# ===== File: src/open_clip_train/logger.py =====

import logging


def setup_logging(log_file, level, include_host=False):
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


# ===== File: src/open_clip_train/main.py =====

import copy
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss, SpaGLaM
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync



LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    # ————做了修改的部分————
    omiclip_model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )
    # --- 关键逻辑：根据配置决定是否包装成SpaGLaM模型 ---
    if args.use_spaglam_model:
        if is_master(args):
            logging.info("Wrapping the OmiCLIP model with the SOTA SpaGLaM architecture.")
        
        # 将基础模型和配置参数传递给SpaGLaM包装器
        model = SpaGLaM(open_clip_model=omiclip_model, config=args)
        model = model.to(device)
    else:
        # 如果不使用SpaGLaM，则行为和原来一样
        model = omiclip_model
# ——————————————————————————————————————

    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        # 注意: tracing SpaGLaM 可能会很复杂，因为输入是图。暂时假设trace只用于标准CLIP模型。
        if not args.use_spaglam_model:
            model = trace_model(model, batch_size=args.batch_size, device=device)
        else:
            logging.warning("Tracing is not supported for SpaGLaM model at the moment. Skipping.")

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else:
            # If some params are not passed, we use the default values based on model name.
            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            include = lambda n, p: not exclude(n, p)

            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            if opt == 'adamw':
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            else:
                assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            if is_master(args):
                defaults = copy.deepcopy(optimizer.defaults)
                defaults['weight_decay'] = args.wd
                defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
                logging.info(
                    f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
                )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir, context_length=args.force_context_length)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])

# ===== File: src/open_clip_train/params.py =====

import argparse
import ast


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto", "spaglam"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument("--momentum", type=float, default=None, help="Momentum (for timm optimizers).")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--opt", type=str, default='adamw',
        help="Which optimizer to use. Choices are ['adamw', or any timm optimizer 'timm/{opt_name}']."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-context-length', type=int, default=None,
        help='Override default context length'
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help="distributed backend. \"nccl\" for GPU, \"hccl\" for Ascend NPU"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )
    parser.add_argument(
        "--loss-dist-impl",
        default=None,
        type=str,
        help='A string to specify a specific distributed loss implementation.'
    )

#     # 添加一个新的参数来指定 AnnData 文件的路径
#     parser.add_argument(
#         "--anndata-path",
#         type=str,
#         default=None,
#         help="Path to the master AnnData file (.h5ad) containing graph and metadata for SpaGLaM."
#     )
#     # 添加GNN相关的参数，为模型配置做准备
    

        # --- SpaGLaM SOTA 模型专属参数 ---
    parser.add_argument(
        "--use-spaglam-model",
        action="store_true",
        default=False,
        help="Activate the SpaGLaM model architecture wrapper instead of standard CLIP."
    )
        # ==================== NEW ARGUMENT ====================
    parser.add_argument(
        "--use-precomputed-embeddings",
        action="store_true",
        default=False,
        help="If activated, the dataloader is expected to provide pre-computed embeddings. The SpaGLaM model will bypass its internal encoder."
    )
    parser.add_argument(
        "--freeze-omiclip",
        action="store_true",
        default=False,
        help="Freeze the pretrained OmiCLIP encoders during SpaGLaM training."
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gat",
        choices=["gat", "graphtransformer"],
        help="Type of GNN backbone to use."
    )
    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=2,
        help="Number of layers in the GNN towers."
    )
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension of the GNN layers."
    )
    parser.add_argument(
        "--gnn-heads",
        type=int,
        default=8,
        help="Number of attention heads for GAT or Graph Transformer."
    )
    parser.add_argument(
        "--use-deep-fusion",
        action="store_true",
        default=False,
        help="Enable deep modality interaction blocks between GNN layers."
    )

    args = parser.parse_args(args)

    if 'timm' not in args.opt:
        # set default opt params based on model name (only if timm optimizer not used)
        default_params = get_default_params(args.model)
        for name, val in default_params.items():
            if getattr(args, name) is None:
                setattr(args, name, val)

    return args

# ===== File: src/open_clip_train/precision.py =====

import torch
from contextlib import suppress
from functools import partial


def get_autocast(precision, device_type='cuda'):
    if precision =='amp':
        amp_dtype = torch.float16
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        amp_dtype = torch.bfloat16
    else:
        return suppress

    return partial(torch.amp.autocast, device_type=device_type, dtype=amp_dtype)
# ===== File: src/open_clip_train/profiler.py =====

import argparse

import torch
import open_clip
import pandas as pd
from torch.utils.flop_counter import FlopCounterMode
try:
    import fvcore
    import fvcore.nn
except:
    fvcore = None

parser = argparse.ArgumentParser(description='OpenCLIP Profiler')

# benchmark specific args
parser.add_argument('--model', metavar='NAME', default='',
                    help='model(s) to profile')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for results')
parser.add_argument('--profiler', default='torch', type=str, choices=['torch', 'fvcore'])
parser.add_argument('--batch-size', default=1, type=int, help='Batch size for profiling')
parser.add_argument(
    "--device", default="cuda", type=str, help="Accelerator to use."
)

def profile_fvcore(
        model,
        image_input_size=(3, 224, 224),
        text_input_size=(77,),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_image_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    example_text_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    fca = fvcore.nn.FlopCountAnalysis(model, (example_image_input, example_text_input))
    aca = fvcore.nn.ActivationCountAnalysis(model, (example_image_input, example_text_input))
    if detailed:
        fcs = fvcore.nn.flop_count_str(fca)
        print(fcs)
    return fca.total() / batch_size, aca.total() / batch_size


def profile_fvcore_text(
        model,
        text_input_size=(77,),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device = next(model.parameters()).device
    example_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    fca = fvcore.nn.FlopCountAnalysis(model, example_input)
    aca = fvcore.nn.ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = fvcore.nn.flop_count_str(fca)
        print(fcs)
    return fca.total() / batch_size, aca.total() / batch_size


def profile_fvcore_image(
        model,
        image_input_size=(3, 224, 224),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    fca = fvcore.nn.FlopCountAnalysis(model, example_input)
    aca = fvcore.nn.ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = fvcore.nn.flop_count_str(fca)
        print(fcs)
    return fca.total() / batch_size, aca.total() / batch_size


def profile_torch_image(model, image_input_size, batch_size=1, force_cpu=False):
    """Profile the image encoder using torch.utils.flop_counter"""
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)

    flop_counter = FlopCounterMode()
    with flop_counter:
        model(example_input)
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    return total_flops / batch_size


def profile_torch_text(model, text_input_size, batch_size=1, force_cpu=False):
    """Profile the text encoder using torch.utils.flop_counter"""
    if force_cpu:
        model = model.to('cpu')
    device = next(model.parameters()).device
    example_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)

    flop_counter = FlopCounterMode()
    with flop_counter:
        model(example_input)
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    return total_flops / batch_size


def profile_torch(model, text_input_size, image_input_size, batch_size=1, force_cpu=False):
    """Profile the full model using torch.utils.flop_counter"""
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    image_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    text_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)

    flop_counter = FlopCounterMode()
    with flop_counter:
        model(image_input, text_input)
    total_flops = sum(flop_counter.get_flop_counts()['Global'].values())
    return total_flops / batch_size


def count_params(model):
    return sum(m.numel() for m in model.parameters())

def profile_model(model_name, batch_size=1, profiler='torch', device="cuda"):
    assert profiler in ['torch', 'fvcore'], 'Only torch and fvcore profilers are supported'
    if profiler == 'fvcore':
        assert fvcore is not None, 'Please install fvcore.'
    model = open_clip.create_model(model_name, force_custom_text=True, pretrained_hf=False)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
    elif device == "npu" and torch.npu.is_available():
        model = model.npu()

    if isinstance(model.visual.image_size, (tuple, list)):
        image_input_size = (3,) + tuple(model.visual.image_size[-2:])
    else:
        image_input_size = (3, model.visual.image_size, model.visual.image_size)

    text_input_size = (77,)
    if hasattr(model, 'context_length') and model.context_length:
        text_input_size = (model.context_length,)

    results = {}
    results['model'] = model_name
    results['image_size'] = image_input_size[1]

    model_cfg = open_clip.get_model_config(model_name)
    if model_cfg:
        vision_cfg = open_clip.CLIPVisionCfg(**model_cfg['vision_cfg'])
        text_cfg = open_clip.CLIPTextCfg(**model_cfg['text_cfg'])
        results['image_width'] = int(vision_cfg.width)
        results['text_width'] = int(text_cfg.width)
        results['embed_dim'] = int(model_cfg['embed_dim'])
    else:
        results['image_width'] = 0
        results['text_width'] = 0
        results['embed_dim'] = 0

    retries = 2
    while retries:
        retries -= 1
        try:
            results['mparams'] = round(count_params(model) / 1e6, 2)
            results['image_mparams'] = round(count_params(model.visual) / 1e6, 2)
            results['text_mparams'] = round(count_params(model.text) / 1e6, 2)

            if profiler == 'fvcore':
                macs, acts = profile_fvcore(
                    model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)

                image_macs, image_acts = profile_fvcore_image(
                    model.visual, image_input_size=image_input_size, force_cpu=not retries, batch_size=batch_size)

                text_macs, text_acts = profile_fvcore_text(
                    model.text, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)

                results['gmacs'] = round(macs / 1e9, 2)
                results['macts'] = round(acts / 1e6, 2)
                
                results['image_gmacs'] = round(image_macs / 1e9, 2)
                results['image_macts'] = round(image_acts / 1e6, 2)
                
                results['text_gmacs'] = round(text_macs / 1e9, 2)
                results['text_macts'] = round(text_acts / 1e6, 2)
            elif profiler == 'torch':
                image_flops = profile_torch_image(
                    model.visual, image_input_size=image_input_size, force_cpu=not retries, batch_size=batch_size)
                text_flops = profile_torch_text(
                    model.text, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)
                total_flops = profile_torch(
                    model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries, batch_size=batch_size)

                results['gflops'] = round(total_flops / 1e9, 2)
                results['image_gflops'] = round(image_flops / 1e9, 2)
                results['text_gflops'] = round(text_flops / 1e9, 2)

        except RuntimeError as e:
            pass
    return results


def main():
    args = parser.parse_args()

    # FIXME accept a text file name to allow lists of models in txt/csv
    if args.model == 'all':
        parsed_model = open_clip.list_models()
    else:
        parsed_model = args.model.split(',')

    results = []
    models_with_errors = []
    for m in parsed_model:
        print('='*100)
        print(f'Profiling {m}')
        try:
            row = profile_model(m, batch_size=args.batch_size, profiler=args.profiler, device=args.device)
            results.append(row)
        except Exception as e:
            print(f'Error profiling {m}: {e}')
            import traceback
            traceback.print_exc()
            models_with_errors.append(m)

    df = pd.DataFrame(results, columns=results[0].keys())

    if 'gmacs' in df.columns:
        df = df.sort_values(by=['gmacs', 'mparams', 'model'])
    else:
        df = df.sort_values(by=['gflops', 'mparams', 'model'])

    print('='*100)
    print('Done.')
    print(df)
    if args.results_file:
        df.to_csv(args.results_file, index=False)

    if models_with_errors:
        print('Models with errors:', models_with_errors)


if __name__ == '__main__':
    main()

# ===== File: src/open_clip_train/scheduler.py =====

import math


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def const_lr_cooldown(optimizer, base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e / es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


# ===== File: src/open_clip_train/spaglam_data.py =====

# ===== 文件: src/open_clip_train/spaglam_data.py (已修正) =====

import logging
import io
import torch
import torch.utils.data
import webdataset as wds
import json
from PIL import Image
import io
from torch_geometric.data import Data as PyGData, Batch as PyGBatch

# 【核心修复】: 从新的 data_defs.py 导入公共类，打破循环
from .data_defs import SharedEpoch, DataInfo


def process_prepackaged_sample(sample: dict, image_processor: callable, tokenizer: callable) -> PyGData:
    """
    此函数被映射到从 WebDataset 流中读取的每个“预打包”样本上。
    它的工作是解码原始字节数据，应用变换，并构建一个 PyTorch Geometric 的 Data 对象。
    """
    try:
        # 1. 从 .json 文件中加载图结构元数据
        graph_metadata = json.loads(sample['json'])
        num_nodes = graph_metadata['num_nodes']
        
        # 2. 从边列表中重建无向图的 edge_index
        edge_list_undirected = []
        for u, v in graph_metadata['edge_index']:
            edge_list_undirected.append([u, v])
            edge_list_undirected.append([v, u])
        
        if edge_list_undirected:
            edge_index = torch.tensor(edge_list_undirected, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 3. 为子图中的所有节点加载、解码和处理图像与文本
        images_to_process = []
        texts_to_tokenize = []
        
        for i in range(num_nodes):
            image_bytes = sample[f"{i}.png"]
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images_to_process.append(pil_image)
            
            text_bytes = sample[f"{i}.txt"]
            texts_to_tokenize.append(text_bytes.decode('utf-8'))

        # 4. 批量进行数据变换
        processed_images = torch.stack([image_processor(img) for img in images_to_process])
        tokenized_texts = tokenizer(texts_to_tokenize)

        # 5. 创建最终的 PyG Data 对象
        pyg_data = PyGData(
            x_image=processed_images,
            x_text=tokenized_texts,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        return pyg_data

    except Exception as e:
        logging.warning(f"跳过损坏或不完整的样本 {sample.get('__key__', 'UNKNOWN')}. 错误: {e}")
        return None


def get_spaglam_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    Factory function for the pre-computed SpaGLaM dataset.
    NOTE: `preprocess_fn` and `tokenizer` are ignored as data is already processed.
    """
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None, "A path to the pre-computed WebDataset shards must be provided."
    
    pipeline = [
        wds.ResampledShards(input_shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        wds.map(process_embedding_sample, handler=wds.warn_and_continue),
        wds.select(lambda x: x is not None),
        wds.batched(args.batch_size, partial=not is_train, collation_fn=PyGBatch.from_data_list),
    ]

    dataset = wds.DataPipeline(*pipeline)
    
    num_samples = args.train_num_samples if is_train else args.val_num_samples
    if is_train and num_samples:
        # Correctly set the epoch length
        world_size = args.world_size if args.world_size > 0 else 1
        num_workers = args.workers if args.workers > 0 else 1
        global_batch_size = args.batch_size * world_size
        
        if global_batch_size > 0:
            num_batches = num_samples // global_batch_size
            if num_workers > 0:
                num_worker_batches = num_batches // num_workers
                if num_worker_batches > 0:
                    dataset = dataset.with_epoch(num_worker_batches)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None, # Batching is done inside the pipeline
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    
    if num_samples is not None:
        world_size = args.world_size if args.world_size > 0 else 1
        global_batch_size = args.batch_size * world_size
        dataloader.num_samples = num_samples
        dataloader.num_batches = num_samples // global_batch_size if global_batch_size > 0 else 0
    else:
        dataloader.num_samples = 0
        dataloader.num_batches = 0

    return DataInfo(dataloader=dataloader, shared_epoch=SharedEpoch(epoch))




# Import from the new definitions file
from .data_defs import DataInfo, SharedEpoch

def process_embedding_sample(sample: dict) -> PyGData:
    """
    Processes a sample from a pre-computed embedding shard.
    It loads serialized PyTorch tensors instead of raw images/text.
    """
    try:
        # 1. Load graph structure from JSON
        metadata = json.loads(sample['json'])
        num_nodes = metadata['num_nodes']
        
        # 2. Reconstruct edge_index
        edge_list_undirected = []
        for u, v in metadata['edge_index']:
            edge_list_undirected.append([u, v])
            edge_list_undirected.append([v, u])
        
        edge_index = torch.tensor(
            edge_list_undirected, dtype=torch.long
        ).t().contiguous() if edge_list_undirected else torch.empty((2, 0), dtype=torch.long)
            
        # 3. Load all node embeddings from .pth files
        image_embeddings = []
        text_embeddings = []
        for i in range(num_nodes):
            img_emb = torch.load(io.BytesIO(sample[f"{i}.image.pth"]))
            txt_emb = torch.load(io.BytesIO(sample[f"{i}.text.pth"]))
            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)

        # 4. Create the final PyG Data object
        pyg_data = PyGData(
            x_image=torch.stack(image_embeddings),
            x_text=torch.stack(text_embeddings),
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        return pyg_data

    except Exception as e:
        logging.warning(f"Skipping corrupted embedding sample {sample.get('__key__', 'UNKNOWN_KEY')}. Error: {e}")
        return None

# ===== File: src/open_clip_train/train.py =====

import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype, CLIP, CustomTextCLIP, SpaGLaM
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if args.use_spaglam_model and args.accum_freq > 1:
        raise NotImplementedError(
            "Gradient accumulation is not yet supported for SpaGLaM in this script. Please use --accum-freq 1."
        )

    model.train()
    if args.distill:
        dist_model.eval()

    data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data['train'].dataloader
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_features = [], [], {}

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if not args.skip_scheduler:
            scheduler(step)

        # --- 关键修改：数据处理分支 ---
        if args.use_spaglam_model:
            # 如果是SpaGLaM模式，batch是一个PyG对象，直接移动到GPU
            batch = batch.to(device, non_blocking=True)
            images, texts = None, None # 显式地将旧变量设为None
            batch_size = batch.num_graphs
        else:
            images, texts = batch
            images = images.to(device=device, dtype=input_dtype, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)
            batch_size = images.shape[0]  # 使用 .shape[0] 比 len() 更稳健

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                # 如果是SpaGLaM模型，直接调用模型
                if args.use_spaglam_model:
                    model_out = model(batch)
                else:
                    model_out = model(images, texts)
                # 修改结束
                logit_scale = model_out["logit_scale"]
                if args.distill:
                    with torch.no_grad():
                        dist_model_out = dist_model(images, texts)
                    model_out.update({f'dist_{k}': v for k, v in dist_model_out.items()})
                losses = loss(**model_out, output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # --- 累积梯度部分的修改 (如果需要的话) ---
            # 注意：累积梯度对于图模型来说逻辑更复杂，因为它不能简单地拼接特征。
            # SOTA策略：建议在初期先将 --accum-freq 设为 1，以简化问题。
            # 如果必须使用累积梯度，你需要累积PyG的Data对象，并在最后一步将它们
            # collate成一个超大的Batch对象再进行一次前向和反向传播。
            # 这里我们暂时假设 accum_freq == 1。
            if args.accum_freq > 1:
                # WARNING: Gradient accumulation with graph-based models is non-trivial.
                # The logic below is for standard CLIP and WILL NOT WORK for SpaGLaM.
                # Please set --accum-freq=1 for SpaGLaM training for now.
                if args.use_spaglam_model:
                    raise NotImplementedError(
                        "Gradient accumulation is not yet supported for SpaGLaM in this script. Please use --accum-freq 1."
                    )
                # ... (原始的累积梯度代码) ...
            pass # 确保这里的逻辑在accum_freq > 1时能被正确处理或跳过

            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]

                accum_images.append(images)
                accum_texts.append(texts)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = loss(**inputs, **inputs_no_accum, output_dict=True)
                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_features = [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            # batch_size = len(images)
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"]
            }            
            log_data.update({name:val.val for name,val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    if not args.use_spaglam_model:
        zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
        metrics.update(zero_shot_metrics)
    else:
        metrics["zero_shot_skipped"] = True

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        # --- 评估循环的修改 ---
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                # --- 数据处理分支 ---
                if args.use_spaglam_model:
                    batch = batch.to(device, non_blocking=True)
                    images, texts = None, None
                    batch_size = batch.num_graphs  # <-- 从 PyG Batch 对象获取 batch_size
                else:
                    images, texts = batch
                    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                    texts = texts.to(device=device, non_blocking=True)
                    batch_size = len(images)  # <-- 从 image tensor 获取 batch_size
                # --- 修改结束 ---

                with autocast():
                    # --- 模型调用分支 ---
                    if args.use_spaglam_model:
                        model_out = model(batch)
                    else:
                        model_out = model(images, texts)
                    # --- 修改结束 ---
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    if args.use_spaglam_model:
                        batch_size = batch.num_graphs
                    else:
                        batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

# ===== File: src/open_clip_train/zero_shot.py =====

import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip_train.precision import get_autocast


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results

# ===== File: src/open_clip/__init__.py =====

from .version import __version__

# --- 在这里或其他模型导入附近添加新行 ---
from .spaglam_model import SpaGLaM 
from .coca_model import CoCa
# ------------------------------------

from .coca_model import CoCa
from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .factory import create_model, create_model_and_transforms, create_model_from_pretrained, get_tokenizer, create_loss
from .factory import list_models, add_model_config, get_model_config, load_checkpoint
from .loss import ClipLoss, DistillClipLoss, CoCaLoss
from .model import CLIP, CustomTextCLIP, CLIPTextCfg, CLIPVisionCfg, \
    convert_weights_to_lp, convert_weights_to_fp16, trace_model, get_cast_dtype, get_input_dtype, \
    get_model_tokenize_cfg, get_model_preprocess_cfg, set_model_preprocess_cfg
from .openai import load_openai_model, list_openai_models
from .pretrained import list_pretrained, list_pretrained_models_by_tag, list_pretrained_tags_by_model, \
    get_pretrained_url, download_pretrained_from_url, is_pretrained_cfg, get_pretrained_cfg, download_pretrained
from .push_to_hf_hub import push_pretrained_to_hf_hub, push_to_hf_hub
from .tokenizer import SimpleTokenizer, tokenize, decode
from .transform import image_transform, AugmentationCfg
from .zero_shot_classifier import build_zero_shot_classifier, build_zero_shot_classifier_legacy
from .zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES

# ===== File: src/open_clip/coca_model.py =====

from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower

try:
    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StopStringCriteria,
        EosTokenCriteria,
        StoppingCriteriaList
    )

    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search"
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search"
    }
    _has_transformers = False


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = (
        LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    )

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return decoder


def _token_to_tensor(token_id, device: str = "cpu") -> torch.Tensor:
    if not isinstance(token_id, torch.Tensor):
        if isinstance(token_id, int):
            token_id = [token_id]
        token_id = torch.tensor(token_id, device=device)
    return token_id


class CoCa(nn.Module):
    def __init__(
            self,
            embed_dim,
            multimodal_cfg: MultimodalCfg,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            pad_id: int = 0,
    ):
        super().__init__()
        multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        vocab_size = (
            text_cfg.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text_decoder = _build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None
        self.pad_id = pad_id

        self.context_length = multimodal_cfg.context_length

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images, normalize: bool = True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, dim=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize: bool = True):
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, dim=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize: bool = True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize: bool = True):
        text_latent, _ = self._encode_text(text, normalize=normalize)
        return text_latent

    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            image_indices: Optional[Union[int, List[int]]] = None,
            text_indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize: bool = True,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            image_output_fmt: str = 'NCHW',
            image_output_extra_tokens: bool = False,
            text_output_fmt: str = 'NLC',
            text_output_extra_tokens: bool = False,
            output_logits: bool = False,
            output_logit_scale_bias: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            image: Input image tensor
            text: Input text tensor
            image_indices: For image tower, Take last n blocks if int, all if None, select matching indices if sequence
            text_indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize: L2 Normalize final image and text features (if present)
            normalize_intermediates: Apply final encoder norm layer to all intermediates (if possible)
            intermediates_only: Only return intermediate features, do not return final features
            image_output_fmt: Shape of intermediate image feature outputs
            image_output_extra_tokens: Return both prefix and spatial intermediate tokens
            text_output_fmt: Shape of intermediate text feature outputs
            text_output_extra_tokens: Return both prefix and spatial intermediate tokens
            output_logits: Include logits in output
            output_logit_scale_bias: Include the logit scale bias in the output
        Returns:

        """
        output = {}
        if intermediates_only:
            # intermediates only disables final feature normalization, and include logits
            normalize = False
            output_logits = False
        if output_logits:
            assert False, 'FIXME, needs implementing'

        if image is not None:
            image_output = self.visual.forward_intermediates(
                image,
                indices=image_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
                output_extra_tokens=image_output_extra_tokens,
            )
            if normalize and "image_features" in image_output:
                image_output["image_features"] = F.normalize(image_output["image_features"], dim=-1)
            output.update(image_output)

        if text is not None:
            text_output = self.text.forward_intermediates(
                text,
                indices=text_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=text_output_fmt,
                output_extra_tokens=text_output_extra_tokens,
            )
            if normalize and "text_features" in text_output:
                text_output["text_features"] = F.normalize(text_output["text_features"], dim=-1)
            output.update(text_output)

        # FIXME text decoder
        logit_scale_exp = self.logit_scale.exp() if output_logits or output_logit_scale_bias else None
        if output_logit_scale_bias:
            output["logit_scale"] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias

        return output

    def forward(
            self,
            image,
            text: Optional[torch.Tensor] = None,
            image_latent: Optional[torch.Tensor] = None,
            image_embs: Optional[torch.Tensor] = None,
            output_labels: bool = True,
    ):
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        if text is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        text_latent, token_embs = self._encode_text(text)

        # FIXME this isn't an ideal solution, would like to improve -RW
        labels: Optional[torch.Tensor] = text[:, 1:] if output_labels else None
        if output_labels:
            # align text_embs and thus logits with labels for teacher-forcing caption loss
            token_embs = token_embs[:, :-1]

        logits = self.text_decoder(image_embs, token_embs)
        out_dict = {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "logit_scale": self.logit_scale.exp()
        }
        if labels is not None:
            out_dict["labels"] = labels
        if self.logit_bias is not None:
            out_dict["logit_bias"] = self.logit_bias
        return out_dict

    def generate(
        self,
        image,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="beam_search",
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False # if True output.shape == (batch_size, seq_len)
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"
        device = image.device

        with torch.no_grad():
            sot_token_id = _token_to_tensor(49406 if sot_token_id is None else sot_token_id, device=device)
            eos_token_id = _token_to_tensor(49407 if eos_token_id is None else eos_token_id, device=device)
            pad_token_id = self.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]
            stopping_criteria = StoppingCriteriaList(stopping_criteria)

            if generation_type == "beam_search":
                output = self._generate_beamsearch(
                    image_inputs=image,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    pad_len = seq_len - output.shape[1]
                    return torch.cat((
                            output,
                            torch.ones(output.shape[0], pad_len, device=device, dtype=output.dtype) * pad_token_id
                        ),
                        dim=1
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            image_latent, image_embs = self._encode_image(image)

            if text is None:
                text = torch.ones((image.shape[0], 1), device=device, dtype=torch.long) * sot_token_id

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            self.eval()
            out = text

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]
                logits = self(
                    image,
                    x,
                    image_latent=image_latent,
                    image_embs=image_embs,
                    output_labels=False,
                )["logits"][:, -1]
                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if (cur_len + 1 == seq_len):
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if all(stopping_criteria(out, None)):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out

    def _generate_beamsearch(
            self,
            image_inputs,
            pad_token_id=None,
            eos_token_id=None,
            sot_token_id=None,
            num_beams=6,
            num_beam_groups=3,
            min_seq_len=5,
            stopping_criteria=None,
            logit_processor=None,
            logit_warper=None,
    ):
        device = image_inputs.device
        batch_size = image_inputs.shape[0]
        image_inputs = torch.repeat_interleave(image_inputs, num_beams, dim=0)
        image_latent, image_embs = self._encode_image(image_inputs)

        input_ids = torch.ones((batch_size * num_beams, 1), device=device, dtype=torch.long)
        input_ids = input_ids * sot_token_id
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            num_beam_groups=num_beam_groups,
        )
        # instantiate logits processors
        logits_processor = (
            LogitsProcessorList([MinLengthLogitsProcessor(min_seq_len, eos_token_id=eos_token_id)])
            if logit_processor is None
            else logit_processor
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        batch_beam_size, cur_len = input_ids.shape
        beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        # initialise score of first beam of each group with 0 and the rest with 1e-9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = prepare_inputs_for_generation(input_ids=input_ids, image_inputs=image_inputs)
            outputs = self(
                model_inputs['images'],
                model_inputs['text'],
                image_latent=image_latent,
                image_embs=image_embs,
                output_labels=False,
            )

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of currentg group only
                next_token_logits = outputs['logits'][batch_group_indices, -1, :]
                vocab_size = next_token_logits.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_logits, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            # increase cur_len
            cur_len = cur_len + 1
            if beam_scorer.is_done or all(stopping_criteria(input_ids, None)):
                break

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
        )
        return sequence_outputs['sequences']


def prepare_inputs_for_generation(input_ids, image_inputs, past=None, **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    else:
        position_ids = None
    return {
        "text": input_ids,
        "images": image_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }

# ===== File: src/open_clip/constants.py =====

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

# Default name for a weights file hosted on the Huggingface Hub.
HF_WEIGHTS_NAME = "open_clip_pytorch_model.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "open_clip_model.safetensors"  # safetensors version
HF_CONFIG_NAME = 'open_clip_config.json'

# ===== File: src/open_clip/convert.py =====

""" Conversion functions for 3rd part state-dicts and non-torch native checkpoint formats.
"""
from typing import Union

import torch
import numpy as np

from .model import CLIP, CustomTextCLIP
from .transformer import TextTransformer, Transformer


@torch.no_grad()
def load_big_vision_weights(model: CustomTextCLIP, checkpoint_path: str):
    """ Load weights from .npz checkpoints for official Google big_vision image-text models

    Currently, the SigLIP source models are supported and a CustomTextCLIP destination model
    w/ timm image encoder.
    """
    from timm.layers import resample_patch_embed, resample_abs_pos_embed

    def _n2p(w, t=True, idx=None):
        if idx is not None:
            w = w[idx]
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False

    def _convert_timm_img(module, prefix):
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
        if embed_conv_w.shape[-2:] != module.patch_embed.proj.weight.shape[-2:]:
            embed_conv_w = resample_patch_embed(
                embed_conv_w,
                module.patch_embed.proj.weight.shape[-2:],
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        module.patch_embed.proj.weight.copy_(embed_conv_w)
        module.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))

        if module.cls_token is not None:
            module.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))

        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
        if pos_embed_w.shape != module.pos_embed.shape:
            assert False, f'{pos_embed_w.shape}, {module.pos_embed.shape}'
            num_prefix_tokens = 0 if getattr(module, 'no_embed_class', False) else getattr(module, 'num_prefix_tokens', 1)
            pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
                pos_embed_w,
                new_size=module.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        module.pos_embed.copy_(pos_embed_w)

        mha_sub, b_sub, ln1_sub = (0, 0, 1)
        for i, block in enumerate(module.blocks.children()):
            if f'{prefix}Transformer/encoderblock/LayerNorm_0/scale' in w:
                block_prefix = f'{prefix}Transformer/encoderblock/'
                idx = i
            else:
                block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
                idx = None
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
            block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale'], idx=idx))
            block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias'], idx=idx))
            block.attn.qkv.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False, idx=idx).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.qkv.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False, idx=idx).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel'], idx=idx).flatten(1))
            block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias'], idx=idx))
            block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale'], idx=idx))
            block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias'], idx=idx))
            for r in range(2):
                getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                    _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel'], idx=idx))
                getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                    _n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias'], idx=idx))

        module.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
        module.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

        if module.attn_pool is not None:
            block_prefix = f'{prefix}MAPHead_0/'
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
            module.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
            module.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
            module.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
            module.attn_pool.kv.weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
            module.attn_pool.kv.bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
            module.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
            module.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
            module.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
            module.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
            for r in range(2):
                getattr(module.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
                getattr(module.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    def _convert_openclip_transformer(module: Transformer, prefix):
        for i, block in enumerate(module.resblocks.children()):
            if f'{prefix}encoderblock/LayerNorm_0/scale' in w:
                block_prefix = f'{prefix}encoderblock/'
                idx = i
            else:
                block_prefix = f'{prefix}encoderblock_{i}/'
                idx = None
            mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
            block.ln_1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale'], idx=idx))
            block.ln_1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias'], idx=idx))
            block.attn.in_proj_weight.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/kernel'], t=False, idx=idx).flatten(1).T for n in ('query', 'key', 'value')]))
            block.attn.in_proj_bias.copy_(torch.cat([
                _n2p(w[f'{mha_prefix}{n}/bias'], t=False, idx=idx).reshape(-1) for n in ('query', 'key', 'value')]))
            block.attn.out_proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel'], idx=idx).flatten(1))
            block.attn.out_proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias'], idx=idx))
            block.ln_2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_1/scale'], idx=idx))
            block.ln_2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_1/bias'], idx=idx))
            block.mlp.c_fc.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_0/kernel'], idx=idx))
            block.mlp.c_fc.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_0/bias'], idx=idx))
            block.mlp.c_proj.weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_1/kernel'], idx=idx))
            block.mlp.c_proj.bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_1/bias'], idx=idx))

    def _convert_openclip_txt(module: TextTransformer, prefix):
        module.token_embedding.weight.copy_(_n2p(w[f'{prefix}Embed_0/embedding'], t=False))
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False).squeeze(0)
        module.positional_embedding.copy_(pos_embed_w)
        _convert_openclip_transformer(module.transformer, prefix=prefix + 'Encoder_0/')
        module.ln_final.weight.copy_(_n2p(w[f'{prefix}Encoder_0/encoder_norm/scale']))
        module.ln_final.bias.copy_(_n2p(w[f'{prefix}Encoder_0/encoder_norm/bias']))
        if module.text_projection is not None:
            module.text_projection.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
            module.text_projection.bias.copy_(_n2p(w[f'{prefix}head/bias']))

    root_prefix = 'params/' if 'params/b' in w else ''
    _convert_timm_img(model.visual.trunk, f'{root_prefix}img/')
    _convert_openclip_txt(model.text, f'{root_prefix}txt/')
    model.logit_bias.copy_(_n2p(w[f'{root_prefix}b'])[0])
    model.logit_scale.copy_(_n2p(w[f'{root_prefix}t'])[0])


@torch.no_grad()
def convert_mobile_clip_state_dict(model: CustomTextCLIP, state_dict, fastvit = True):

    def _convert_timm_img(state_dict):
        if fastvit:
            from timm.models.fastvit import checkpoint_filter_fn
        else:
            from timm.models.vision_transformer_hybrid import checkpoint_filter_fn
        timm_state_dict = checkpoint_filter_fn(state_dict, model.visual.trunk)
        timm_state_dict = {'visual.trunk.' + k: v for k, v in timm_state_dict.items()}
        return timm_state_dict

    def _convert_openclip_txt(state_dict, prefix='text_encoder.'):
        text_dict = {}
        for k, v in state_dict.items():
            if not k.startswith(prefix):
                continue
            k = k.replace(prefix, '')
            k = k.replace('projection_layer', 'text_projection')
            k = k.replace('embedding_layer', 'token_embedding')
            if k.startswith('positional_embedding.pos_embed.pos_embed'):
                k = k.replace('positional_embedding.pos_embed.pos_embed', 'positional_embedding')
                v = v.squeeze()
            k = k.replace('final_layer_norm', 'ln_final')
            k = k.replace('pre_norm_mha.0', 'ln_1')
            k = k.replace('pre_norm_mha.1', 'attn')
            k = k.replace('pre_norm_ffn.0', 'ln_2')
            k = k.replace('pre_norm_ffn.1', 'mlp.c_fc')
            k = k.replace('pre_norm_ffn.4', 'mlp.c_proj')
            k = k.replace('qkv_proj.weight', 'in_proj_weight')
            k = k.replace('qkv_proj.bias', 'in_proj_bias')
            k = k.replace('transformer.', 'transformer.resblocks.')
            text_dict['text.' + k] = v
        return text_dict

    image_dict = _convert_timm_img(state_dict)
    text_dict = _convert_openclip_txt(state_dict)
    out_dict = {**image_dict, **text_dict}
    out_dict['logit_scale'] = state_dict['logit_scale']
    return out_dict


def convert_state_dict(model: Union[CustomTextCLIP, CLIP], state_dict):
    if 'image_encoder.model.patch_embed.0.rbr_conv.0.conv.weight' in state_dict:
        # Apple MobileCLIP s1 & s2 state_dicts (s0 and b not currently supported)
        state_dict = convert_mobile_clip_state_dict(model, state_dict)
    if 'image_encoder.model.patch_emb.0.block.conv.weight' in state_dict:
        # convert b model
        state_dict = convert_mobile_clip_state_dict(model, state_dict, fastvit=False)
    return state_dict

# ===== File: src/open_clip/factory.py =====

import json
import logging
import os
import re
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .convert import convert_state_dict
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg
from .coca_model import CoCa
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained,\
    list_pretrained_tags_by_model, download_pretrained_from_hf
from .transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs
from .tokenizer import HFTokenizer, SimpleTokenizer, SigLipTokenizer, DEFAULT_CONTEXT_LENGTH

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    """ Fetch model config from builtin (local library) configs.
    """
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

# Define Schema Prefixes as constants
HF_HUB_PREFIX = 'hf-hub:'
LOCAL_DIR_PREFIX = 'local-dir:'

def parse_model_name(model_name: str) -> Tuple[Optional[str], str]:
    """
    Parses a model name string to identify a schema and the remaining identifier.

    Args:
        model_name: The model name string (e.g., 'ViT-B-32',
                    'hf-hub:org/repo', 'local-dir:/path/to/dir',
                    'local-dir:./relative/path').

    Returns:
        A tuple (schema, identifier):
          - schema (Optional[str]): 'hf-hub', 'local-dir', or None if no schema detected.
          - identifier (str): The part after the schema prefix, or the original
                              string if no schema was present. For 'local-dir',
                              this is the raw path string provided.
    Raises:
        ValueError: If a schema prefix is present but the identifier part is empty.
    """
    # Check for local directory schema first
    if model_name.startswith(LOCAL_DIR_PREFIX):
        # Extract the identifier (path) after the prefix
        identifier = model_name[len(LOCAL_DIR_PREFIX):]
        # Validate that the identifier (path) is not empty
        if not identifier:
            raise ValueError("Empty path specified after 'local-dir:' schema.")
        # Return the schema and the raw path identifier
        # Note: We don't resolve or fully validate the path here,
        #       that's left to the calling function (e.g., using os.path.isdir)
        return 'local-dir', identifier

    # Check for Hugging Face Hub schema
    elif model_name.startswith(HF_HUB_PREFIX):
        # Extract the identifier (HF Hub ID) after the prefix
        identifier = model_name[len(HF_HUB_PREFIX):]
        # Validate that the identifier is not empty
        if not identifier:
            raise ValueError("Empty identifier specified after 'hf-hub:' schema.")
        # Return the schema and the HF Hub ID
        return 'hf-hub', identifier

    # If neither schema prefix is found
    else:
        # No schema detected, return None for schema and the original string as identifier
        return None, model_name


def _get_hf_config(
        model_id: str,
        cache_dir: Optional[str] = None,
):
    """ Fetch model config from HuggingFace Hub.
    """
    config_path = download_pretrained_from_hf(
        model_id,
        filename='open_clip_config.json',
        cache_dir=cache_dir,
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def load_state_dict(
        checkpoint_path: str,
        device='cpu',
        weights_only=True,
):
    # Check if safetensors or not and load weights accordingly
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(
        model: Union[CLIP, CustomTextCLIP],
        checkpoint_path: str,
        strict: bool = True,
        weights_only: bool = True,
        device='cpu',
):
    if Path(checkpoint_path).suffix in ('.npz', '.npy'):
        # Separate path loading numpy big_vision (SigLIP) weights
        from open_clip.convert import load_big_vision_weights
        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path, device=device, weights_only=weights_only)

    # Detect & convert 3rd party state_dicts -> open_clip
    state_dict = convert_state_dict(model, state_dict)

    # Detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)

    # correct if logit_scale differs in being scaler vs 1d param
    if 'logit_scale' in state_dict and model.logit_scale.ndim != state_dict['logit_scale'].ndim:
        state_dict['logit_scale'] = state_dict['logit_scale'].reshape(model.logit_scale.shape)

    # correct if logit_bias differs in being scaler vs 1d param
    if 'logit_bias' in state_dict and model.logit_bias.ndim != state_dict['logit_bias'].ndim:
        state_dict['logit_bias'] = state_dict['logit_bias'].reshape(model.logit_bias.shape)

    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    if 'logit_bias' not in state_dict and model.logit_bias is not None:
        state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])

    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]

    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)

    # Finally, load the massaged state_dict into model
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def _find_checkpoint_in_dir(dir_path: Path) -> Optional[str]:
    checkpoints = list(dir_path.glob('*.safetensors')) + list(dir_path.glob('*.bin')) + list(dir_path.glob('*.pth'))
    if not checkpoints:
        return None
    checkpoints.sort()
    checkpoints.sort(key=lambda x: x.suffix == '.safetensors', reverse=True)
    preferred_order = [
        "open_clip_model.safetensors", "open_clip_pytorch_model.safetensors",
        "open_clip_pytorch_model.bin",  "open_clip_pytorch_model.pth",
        "model.safetensors", "pytorch_model.bin", "pytorch_model.pth", "model.pth"
    ]
    preferred_checkpoints = [c for c in checkpoints if c.name in preferred_order]
    if preferred_checkpoints:
        preferred_checkpoints.sort(key=lambda x: preferred_order.index(x.name))
        chosen = preferred_checkpoints[0]
        logging.info(f"Found preferred checkpoint file: {chosen.name} in {dir_path}")
        return str(chosen)
    chosen = checkpoints[0]
    logging.warning(
        f"Multiple checkpoints found in {dir_path}: {[c.name for c in checkpoints]}. Using '{chosen.name}'.")
    return str(chosen)


def create_model(
        model_name: str, # Can contain schemas 'hf-hub:' or 'local-dir:'
        pretrained: Optional[str] = None, # Used ONLY if model_name has NO schema
        load_weights: bool = True,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        force_context_length: Optional[int] = None,
        pretrained_image: bool = False, # Load default base image weights (at creation, if no CLIP weights)
        pretrained_text: bool = True,  # Load default base text weights (at creation, if no CLIP weights) - NEW
        pretrained_image_path: Optional[str] = None, # Load specific image weights from file (after creation)
        pretrained_text_path: Optional[str] = None, # Load specific text weights from file (after creation)
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        weights_only: bool = True,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Creates and configures a contrastive vision-language model.

    `model_name` specifies architecture/config source:
      - 'ViT-B-32': Built-in model name. `pretrained` specifies CLIP weights source (tag or file path).
      - 'hf-hub:org/repo': Loads config/weights from HF Hub. `pretrained` is IGNORED.
      - 'local-dir:/path/to/folder': Loads config/weights from local dir. `pretrained` is IGNORED.

    Base tower weights loading controlled by `pretrained_image` and `pretrained_text` flags,
    only effective if no full CLIP checkpoint (`pretrained` or schema source) is loaded.

    Tower-specific weights can be loaded *after* creation via `pretrained_image_path`
    and `pretrained_text_path`.

    Args:
        model_name: Model identifier, potentially with schema ('hf-hub:', 'local-dir:').
        pretrained: Source for CLIP weights (tag or file path) ONLY if model_name has no schema.
        load_weights: Load the resolved pretrained weights if True, otherwise random init or tower overrides only.
        precision: Model precision ('fp32', 'fp16', 'bf16', ...).
        device: Device ('cpu', 'cuda', ...).
        jit: If True, JIT compile the model.
        force_quick_gelu: Force use of QuickGELU activation in model config.
        force_custom_text: Force use of custom text encoder architecture.
        force_patch_dropout: Override patch dropout value in model config.
        force_image_size: Override image size in model config.
        force_preprocess_cfg: Dict to override specific FINAL preprocessing parameters.
        force_context_length: Override context length in model config.
        pretrained_image: Load default base weights for image tower at creation if no CLIP weights loaded.
        pretrained_text: Load default base weights for text tower at creation if no CLIP weights loaded (default: True).
        pretrained_image_path: Path to load weights specifically into image tower after creation.
        pretrained_text_path: Path to load weights specifically into text tower after creation.
        cache_dir: Cache directory for downloads.
        output_dict: If True and model supports it, return dict output.
        require_pretrained: Raise error if no `pretrained` CLIP weights loaded when required.
        weights_only: Use weights_only=True for torch.load (safer).
        **model_kwargs: Additional keyword arguments for model constructor (highest override priority).

    Returns:
        The created model instance.
    """
    schema, identifier = parse_model_name(model_name)
    if 'pretrained_hf' in model_kwargs:
        # for backwards compat, override pretrained_text
        pretrained_text = model_kwargs.pop('pretrained_hf')
    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = None
    preprocess_cfg = asdict(PreprocessCfg())  # initialize with defaults
    checkpoint_path = None # Final path for *full CLIP* weights
    pretrained_cfg_for_tag = None # Store tag config if pretrained is a tag and schema is None

    config_source_description = f"Schema: {schema}, Identifier: {identifier}"
    logging.info(f"Parsing model identifier. {config_source_description}")

    if schema and pretrained:
        logging.warning(f"Ignoring `pretrained='{pretrained}'` because `model_name` has '{schema}' schema.")
        pretrained = None  # Nullify pretrained as it's ignored

    # Handle schemas first - these ignore the `pretrained` argument
    if schema == 'local-dir':
        # Handle local directory schema
        local_path = Path(identifier)
        if not local_path.is_dir():
            raise FileNotFoundError(f"Directory specified via 'local-dir:' schema not found: {local_path}")

        local_config_path = local_path / 'open_clip_config.json'
        logging.info(f"Attempting to load config from local dir: {local_config_path}")
        if local_config_path.is_file():
            try:
                # Try loading and parsing the JSON config
                with open(local_config_path, 'r', encoding='utf-8') as f:
                    local_json_config = json.load(f)
                # Check if the required 'model_cfg' key is present
                if 'model_cfg' in local_json_config:
                    # Load model config and merge preprocess config
                    model_cfg = local_json_config['model_cfg']
                    preprocess_cfg = merge_preprocess_dict(preprocess_cfg, local_json_config.get('preprocess_cfg', {}))
                    logging.info(f"Loaded model config and preprocess from: {local_config_path}")
                    # Look for weights checkpoint in the same directory
                    checkpoint_path = _find_checkpoint_in_dir(local_path)
                    if checkpoint_path:
                        logging.info(f"Found CLIP weights in local folder: {checkpoint_path}")
                    else:
                        logging.warning(f"Local config loaded, but no CLIP weights found in {local_path}")
                else:
                    # Config file exists but lacks the necessary key
                    raise ValueError(f"Local config {local_config_path} missing 'model_cfg'.")
            except Exception as e:
                # Handle JSON parsing errors or other exceptions during config load
                raise ValueError(f"Could not load valid config from specified 'local-dir:{identifier}': {e}") from e
        else:
            # Directory exists but the config file is missing
            raise FileNotFoundError(f"'local-dir:' specified, but config file missing: {local_config_path}")

    elif schema == 'hf-hub':
        # Handle Hugging Face Hub schema
        model_id = identifier
        logging.info(f"Attempting to load config from HF Hub: {model_id}")
        try:
            # Fetch configuration from Hugging Face Hub
            hf_config = _get_hf_config(model_id, cache_dir=cache_dir)
            if 'model_cfg' not in hf_config:
                raise RuntimeError(f"'model_cfg' not found in config from {model_id}")
            # Load model config and merge preprocess config
            model_cfg = hf_config['model_cfg']
            preprocess_cfg = merge_preprocess_dict(preprocess_cfg, hf_config.get('preprocess_cfg', {}))
            logging.info(f"Loaded model config from HF Hub: {model_id}")
            # Attempt find default weights file from the Hub repo
            try:
                checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=cache_dir)
                logging.info(f"Found default weights file on HF Hub: {checkpoint_path}")
            except Exception as e_weights:
                # Log warning if weights download fails, but proceed (might only need config)
                logging.warning(f"Could not find/download default weights on HF Hub for {model_id}: {e_weights}")
        except Exception as e_config:
            # Handle errors during config fetching from HF Hub
            raise RuntimeError(f"Failed initial config/weights load from HF Hub {model_id}: {e_config}") from e_config

    # No Schema Prefix - Use built-in name + pretrained arg (tag or file)
    elif schema is None:
        # Handle model names without schema prefix
        # Use identifier (original model_name) and clean it for lookup
        model_name_cleaned = identifier.replace('/', '-')

        # Get base config from built-in name using the cleaned identifier
        model_cfg = get_model_config(model_name_cleaned)
        if model_cfg is None:
            # Raise error if no matching built-in config found
            raise RuntimeError(
                f"Model config for '{model_name_cleaned}' not found in built-ins. Available: {list_models()}")
        logging.info(f"Loaded built-in {model_name_cleaned} model config.")

        # Determine checkpoint path and update preprocess_cfg based on `pretrained` arg (tag or file)
        if pretrained:
            # Check if `pretrained` is a known tag
            pretrained_cfg_for_tag = get_pretrained_cfg(model_name_cleaned, pretrained)
            if pretrained_cfg_for_tag:
                try:
                    # Download weights associated with the tag
                    checkpoint_path = download_pretrained(pretrained_cfg_for_tag, cache_dir=cache_dir)
                    preprocess_cfg = merge_preprocess_dict(preprocess_cfg, pretrained_cfg_for_tag)
                    # QuickGELU compatibility check will happen in after force overrides
                except Exception as e:
                    logging.error(f"Failed to download weights for tag '{pretrained}': {e}")
                    raise RuntimeError(f"Failed to download weights for tag '{pretrained}': {e}")
            elif os.path.isfile(pretrained):
                # Handle pretrained file path
                logging.info(f"`pretrained` specifies file path: {pretrained}")
                checkpoint_path = pretrained
            else:
                logging.error(
                    f"Pretrained tag or path ({pretrained}) for '{model_name_cleaned}' not found. "
                    f"Available tags: {list_pretrained_tags_by_model(model_name_cleaned)}"
                )
                raise RuntimeError(f"Pretrained value '{pretrained}' is not a known tag or valid file path")

    # Apply model config overrides
    if model_cfg is None:
        raise RuntimeError("Model configuration could not be determined after Stage 1.")
    text_cfg = model_cfg['text_cfg']
    vision_cfg = model_cfg['vision_cfg']
    if force_quick_gelu:
        model_cfg["quick_gelu"] = True
    if force_patch_dropout is not None:
        vision_cfg["patch_dropout"] = force_patch_dropout
    if force_image_size is not None:
        vision_cfg["image_size"] = force_image_size
    if force_context_length is not None:
        text_cfg["context_length"] = force_context_length

    # Check compatibility (e.g., QuickGELU warning for tags)
    if schema is None and pretrained_cfg_for_tag:
        # Only perform check if config came from built-in and weights from a tag
        model_quick_gelu = model_cfg.get('quick_gelu', False) # Check the potentially overridden value
        tag_quick_gelu = pretrained_cfg_for_tag.get('quick_gelu', False)
        if tag_quick_gelu != model_quick_gelu:
            # Warn if the final model config's GELU setting mismatches the tag's training setting
             warnings.warn(
                 f"QuickGELU mismatch between final model config (quick_gelu={model_quick_gelu}) "
                 f"and pretrained tag '{pretrained}' (quick_gelu={tag_quick_gelu}).",
                 UserWarning
             )

    # Decide whether to use the checkpoint path based on load_weights
    if checkpoint_path is not None:
        if not load_weights:
            logging.info(
                f"Potential checkpoint path '{checkpoint_path}' found, but skipping assignment due to load_weights=False.")
            checkpoint_path = None
    else:
        logging.info("No potential checkpoint path found from config source or pretrained arg.")

    # Set default base weight loading flags for image and text towers
    # Only load base pretrained weights if other weights will not be loaded into respective towers
    enable_default_image_weights = pretrained_image and pretrained_image_path is None and checkpoint_path is None
    enable_default_text_weights = pretrained_text and pretrained_text_path is None and checkpoint_path is None
    is_timm_model = 'timm_model_name' in model_cfg.get("vision_cfg", {})
    is_hf_text_model = 'hf_model_name' in model_cfg.get('text_cfg', {})
    if is_timm_model:
        vision_cfg['timm_model_pretrained'] = enable_default_image_weights
    if is_hf_text_model:
        text_cfg['hf_model_pretrained'] = enable_default_text_weights

    # Determine model class (CLIP, CustomTextCLIP, CoCa)
    custom_text = model_cfg.pop('custom_text', False) or force_custom_text or is_hf_text_model
    if custom_text:
        # Use CustomTextCLIP (or CoCa if multimodal_cfg is present)
        if "multimodal_cfg" in model_cfg:
            model_class = CoCa
        else:
            model_class = CustomTextCLIP
    else:
        # Default to standard CLIP
        model_class = CLIP

    # Apply final **kwargs overrides (highest priority) to a copy of model_cfg
    final_model_cfg = deepcopy(model_cfg)
    final_model_cfg.update(model_kwargs)

    # Get casting dtype based on precision argument
    cast_dtype = get_cast_dtype(precision)

    # Instantiate the model
    logging.info(f"Instantiating model architecture: {model_class.__name__}")
    model = model_class(**final_model_cfg, cast_dtype=cast_dtype)
    _set_model_device_and_precision(model, device, precision, is_timm_model)

    # Load Full Pretrained CLIP Weights (if path exists)
    pretrained_loaded = False
    if checkpoint_path:
        logging.info(f'Loading full pretrained weights from: {checkpoint_path}')
        # Use the load_checkpoint helper which handles state dict loading, conversions, etc.
        # Use strict=True by default for full model loading to catch mismatches.
        load_checkpoint(
            model,
            checkpoint_path,
            strict=True,
            weights_only=weights_only,
            device='cpu' # Load to CPU first
        )
        pretrained_loaded = True

    # Load tower-specific weights (image and text), after the full CLIP checkpoint, potentially overwriting parts.
    pretrained_image_loaded = False # Track if specific image weights loaded
    if pretrained_image_path:
        if os.path.isfile(pretrained_image_path):
            logging.info(f"Attempting to load image tower weights from: {pretrained_image_path}")
            try:
                # Load the state dict from the file
                image_state_dict = load_state_dict(
                    pretrained_image_path,
                    device='cpu',
                    weights_only=weights_only
                )
                # Check if model has the 'visual' attribute
                if hasattr(model, 'visual'):
                    # Load into the visual tower, use strict=False for flexibility
                    incompatible_keys = model.visual.load_state_dict(image_state_dict, strict=False)
                    logging.info(
                        f"Loaded image tower weights from {pretrained_image_path}. Incompatible keys: {incompatible_keys}")
                    pretrained_image_loaded = True # Mark specific image weights as loaded
                else:
                    # Model structure doesn't match expectation
                    logging.warning(
                        f"Model does not have a 'visual' attribute, cannot load image tower weights from {pretrained_image_path}")
            except Exception as e:
                # Handle errors during image tower weight loading
                logging.error(f"Error loading image tower weights from {pretrained_image_path}: {e}")
        else:
            # Path provided is not a valid file
            logging.warning(f"Invalid file path specified for pretrained_image_path: {pretrained_image_path}")

    pretrained_text_loaded = False # Track if specific text weights loaded
    if pretrained_text_path:
        if os.path.isfile(pretrained_text_path):
            logging.info(f"Attempting to load text tower weights from: {pretrained_text_path}")
            try:
                # Load the state dict from the file
                text_state_dict = load_state_dict(
                    pretrained_text_path,
                    device='cpu',
                    weights_only=weights_only
                )
                # Safely get the text attribute (usually 'text', but could be different)
                text_module = getattr(model, 'text', model)
                if text_module is not None:
                    # Load into the text tower, use strict=False for flexibility
                    incompatible_keys = text_module.load_state_dict(text_state_dict, strict=False)
                    logging.info(f"Loaded text tower weights from {pretrained_text_path}. Incompatible keys: {incompatible_keys}")
                    pretrained_text_loaded = True # Mark specific text weights as loaded
                else:
                    # Model structure doesn't match expectation
                    logging.warning(f"Model does not have a standard 'text' attribute, cannot load text tower weights from {pretrained_text_path}")
            except Exception as e:
                # Handle errors during text tower weight loading
                logging.error(f"Error loading text tower weights from {pretrained_text_path}: {e}")
        else:
            # Path provided is not a valid file
            logging.warning(f"Invalid file path specified for pretrained_text_path: {pretrained_text_path}")

    partially_loaded = enable_default_text_weights or enable_default_image_weights \
        or pretrained_image_loaded or pretrained_text_loaded
    if require_pretrained and not pretrained_loaded:
         # If CLIP weights were required but failed to load, raise an error.
         # Loading tower-specific weights does not satisfy `require_pretrained`.
         raise RuntimeError(
             f"Required pretrained weights (`model_name='{model_name}', pretrained='{pretrained}'`) could not be loaded. "
         )
    elif not pretrained_loaded and partially_loaded:
         # Some tower weights loaded
         logging.warning(f"Model {model_name} initialized partially.")
    elif not pretrained_loaded and not partially_loaded:
         # Absolutely no weights were loaded from any source
         logging.warning(f"No pretrained weights loaded for model '{model_name}'. Model initialized randomly.")

    if output_dict and hasattr(model, "output_dict"):
        # Enable dictionary output if model supports it
        model.output_dict = True

    if jit:
        logging.info("Attempting JIT scripting...")
        try:
            model = torch.jit.script(model)
            logging.info("JIT scripting successful.")
        except Exception as e:
            logging.warning(f"JIT scripting failed: {e}. Returning non-JIT model.")

    # Prepare and set final preprocessing configuration on the model
    final_preprocess_cfg = deepcopy(preprocess_cfg) # Start with config determined earlier
    # Ensure image_size in preprocess config matches the actual model's visual component size, if possible
    visual_module = getattr(model, 'visual', None)
    if visual_module is not None and hasattr(visual_module, 'image_size'):
        # Update preprocess size from the instantiated visual module
         final_preprocess_cfg['size'] = visual_module.image_size
    # Apply force_preprocess_cfg overrides (highest priority for preprocessing)
    final_preprocess_cfg = merge_preprocess_dict(final_preprocess_cfg, force_preprocess_cfg or {})

    # Attach the final config to the model
    set_model_preprocess_cfg(model, final_preprocess_cfg)
    logging.info(f"Final image preprocessing configuration set: {final_preprocess_cfg}")

    # Log completion and return the configured model
    logging.info(f"Model {model_name} creation process complete.")
    return model


def get_tokenizer(
        model_name: str = '',
        context_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        **kwargs, # Additional tokenizer kwargs passed to constructor
):
    """
    Gets the appropriate tokenizer based on the model identifier schema or name.

    `model_name` can specify source via schema:
      - 'ViT-B-32': Looks up built-in config to determine tokenizer type.
      - 'hf-hub:org/repo': Loads config from HF Hub to determine tokenizer type.
      - 'local-dir:/path/to/folder': Loads config from local dir to determine tokenizer type.
    """
    schema, identifier = parse_model_name(model_name)

    config = None # Stores the loaded model_cfg relevant section (usually text_cfg)
    config_source_description = "None" # For logging purposes
    local_dir_path = None # Store path if schema is local-dir to resolve relative paths
    hf_fallback_id = None

    # Determine Configuration Source based on Schema
    logging.info(f"Parsing tokenizer identifier. Schema: {schema}, Identifier: {identifier}")

    if schema == 'local-dir':
        # Handle local directory schema
        local_dir_path = Path(identifier) # Store the path for later use
        config_source_description = f"local-dir: {local_dir_path}"
        if not local_dir_path.is_dir():
            raise FileNotFoundError(f"Directory specified via 'local-dir:' schema not found: {local_dir_path}")

        local_config_path = local_dir_path / 'open_clip_config.json'
        logging.info(f"Attempting to load config from {config_source_description} at {local_config_path}")
        if local_config_path.is_file():
            try:
                # Load and parse the JSON config
                with open(local_config_path, 'r', encoding='utf-8') as f:
                    local_json_config = json.load(f)
                if 'model_cfg' in local_json_config:
                    config = local_json_config['model_cfg']
                else:
                    raise ValueError(f"Local config {local_config_path} missing 'model_cfg'.")
            except Exception as e:
                raise ValueError(f"Could not load valid config from 'local-dir:{identifier}': {e}") from e
        else:
             raise FileNotFoundError(f"'local-dir:' specified, but config file missing: {local_config_path}")

    elif schema == 'hf-hub':
        # Handle Hugging Face Hub schema
        model_id = identifier
        config_source_description = f"hf-hub: {model_id}"
        logging.info(f"Attempting to load config from {config_source_description}")
        config_err = ''
        try:
            # Fetch config from HF Hub
            hf_config = _get_hf_config(model_id, cache_dir=cache_dir)
            config = hf_config.get('model_cfg', None)
            if not config:
                config_err = 'model_cfg key not found'
        except Exception as e:
            config_err = str(e)
        if not config:
            hf_fallback_id = model_id
            config = {}
            logging.warning(
                f"Could not load config from {config_source_description}: {config_err}."
                f"Falling back to using model_id for tokenizer.")

    elif schema is None and identifier:
        # Try built-in config lookup using the identifier (original model_name)
        config_source_description = f"built-in: {identifier}"
        logging.info(f"Attempting to load config from {config_source_description}")
        config = get_model_config(identifier)

    # Check if config determination failed completely (should only be possible if initial schema parsing failed badly)
    if config is None:
        logging.warning(f"Model configuration not found, returning default SimpleTokenizer.")
        return SimpleTokenizer(context_length=context_length or DEFAULT_CONTEXT_LENGTH, **kwargs)

    # Safely access text_cfg even if config is {} (from non-builtin name case)
    text_config = config.get('text_cfg', {})

    # Resolve context length: argument > config > default
    if context_length is None:
        # Use context_length from text_cfg if available, otherwise default
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    # Merge tokenizer kwargs: function kwargs override config kwargs
    tokenizer_kwargs = text_config.get('tokenizer_kwargs', {}) # Start with config kwargs
    tokenizer_kwargs.update(kwargs) # Apply caller kwargs, overriding config ones

    # Get the specified HF tokenizer name from config, if any
    hf_tokenizer_name = text_config.get('hf_tokenizer_name', '')
    if not hf_tokenizer_name and hf_fallback_id:
        hf_tokenizer_name = hf_fallback_id

    if hf_tokenizer_name:
        # If 'hf_tokenizer_name' key exists in text_cfg (even if empty string): Use HFTokenizer.
        if schema == 'local-dir':
            # If config came from local-dir, ALWAYS use the local dir path for HFTokenizer.
            # This assumes the tokenizer files are inside that directory.
            tokenizer_source = local_dir_path
        else:
            tokenizer_source = hf_tokenizer_name

        logging.info(f"Using HFTokenizer with source: '{tokenizer_source}'")
        tokenizer = HFTokenizer(
            tokenizer_source,
            context_length=context_length,
            cache_dir=cache_dir,
            **tokenizer_kwargs,
        )

    elif schema is None and 'siglip' in identifier.lower():
        # Check for SigLIP naming convention ONLY if no schema was present AND no hf_tokenizer_name found
        # Avoids misinterpreting 'local-dir:/path/with/siglip/in/name'
        tn_variant = 'gemma' if 'siglip2' in identifier.lower() else 'mc4' if 'i18n' in identifier.lower() else 'c4-en'
        logging.info(f"Using SigLipTokenizer variant: {tn_variant}")
        tokenizer = SigLipTokenizer(
            tn_variant,
            context_length=context_length,
        )
    else:
        # Default to SimpleTokenizer if no HF specified and not SigLIP name match
        logging.info("Using default SimpleTokenizer.")
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer


def _set_model_device_and_precision(
        model: torch.nn.Module,
        device: torch.device,
        precision: str,
        is_timm_model: bool = False
):
    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        # manual mixed precision that matches original OpenAI behaviour
        if is_timm_model:
            from .transformer import LayerNormFp32
            # FIXME this is a bit janky, create timm based model in low-precision and
            # then cast only LayerNormFp32 instances back to float32 so they don't break.
            # Why? The convert_weights_to_lp fn only works with native models.
            model.to(device=device, dtype=dtype)

            def _convert_ln(m):
                if isinstance(m, LayerNormFp32):
                    m.weight.data = m.weight.data.to(torch.float32)
                    m.bias.data = m.bias.data.to(torch.float32)

            model.apply(_convert_ln)
        else:
            model.to(device=device)
            convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)


def create_loss(args):
    if args.distill:
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif "coca" in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif args.siglip:
        assert not args.horovod, "Horovod not currently supported for SigLip"
        return SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
            dist_impl=args.loss_dist_impl,  # siglip has multiple distributed implementations to choose from
        )

    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        load_weights: bool = True,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_context_length: Optional[int] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        pretrained_image: bool = False,
        pretrained_text: bool = True,
        pretrained_image_path: Optional[str] = None,
        pretrained_text_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        weights_only: bool = True,
        **model_kwargs,
):
    """
    Creates a contrastive vision-language model along with preprocessing transforms for training and validation.

    This function combines model creation with the generation of appropriate image preprocessing pipelines,
    making it convenient for training workflows where both model and transforms are needed.

    `model_name` specifies architecture/config source:
      - 'ViT-B-32': Built-in model name. `pretrained` specifies CLIP weights source (tag or file path).
      - 'hf-hub:org/repo': Loads config/weights from HF Hub. `pretrained` is IGNORED.
      - 'local-dir:/path/to/folder': Loads config/weights from local dir. `pretrained` is IGNORED.

    The preprocessing transforms are automatically configured based on the model's requirements,
    with separate pipelines for training (with augmentation) and validation (without augmentation).

    Args:
        model_name: Model identifier, potentially with schema ('hf-hub:', 'local-dir:').
        pretrained: Source for CLIP weights (tag or file path) ONLY if model_name has no schema.
        load_weights: Load the resolved pretrained weights if True, otherwise random init or tower overrides only.
        precision: Model precision ('fp32', 'fp16', 'bf16', ...).
        device: Device ('cpu', 'cuda', ...).
        jit: If True, JIT compile the model.
        force_quick_gelu: Force use of QuickGELU activation in model config.
        force_custom_text: Force use of custom text encoder architecture.
        force_patch_dropout: Override patch dropout value in model config.
        force_image_size: Override image size in model config.
        force_context_length: Override context length in model config.
        image_mean: Override default image normalization mean values (per channel).
        image_std: Override default image normalization std values (per channel).
        image_interpolation: Override default interpolation method for image resizing.
        image_resize_mode: Override resize mode for inference preprocessing ('squash', 'longest', 'shortest').
        aug_cfg: Augmentation configuration for training transforms. Can be dict or AugmentationCfg object.
                 Controls random crop, color jitter, etc. If None, uses model defaults.
        pretrained_image: Load default (timm) base weights for image tower at creation if no CLIP weights loaded.
        pretrained_text: Load default (hf) base weights for text tower at creation if no CLIP weights loaded.
        pretrained_image_path: Path to load weights specifically into image tower after creation.
        pretrained_text_path: Path to load weights specifically into text tower after creation.
        cache_dir: Cache directory for downloads.
        output_dict: If True and model supports it, return dict output.
        weights_only: Use weights_only=True for torch.load (safer).
        **model_kwargs: Additional keyword arguments for model constructor (highest override priority).

    Returns:
        Tuple[torch.nn.Module, Callable, Callable]: A tuple containing:
            - model: The created model instance
            - preprocess_train: Image preprocessing transform for training (includes augmentation)
            - preprocess_val: Image preprocessing transform for validation/inference (no augmentation)

    Example:
        >>> # Basic usage with built-in model
        >>> model, train_transform, val_transform = create_model_and_transforms('ViT-B-32', pretrained='openai')
        >>>
        >>> # With custom augmentation
        >>> aug_cfg = {'scale': (0.9, 1.0), 'ratio': (1.0, 1.0)}
        >>> model, train_transform, val_transform = create_model_and_transforms(
        ...     'ViT-L-14',
        ...     pretrained='datacomp_xl_s13b_b90k',
        ...     aug_cfg=aug_cfg
        ... )
        >>>
        >>> # From Hugging Face Hub
        >>> model, train_transform, val_transform = create_model_and_transforms('hf-hub:org/model-repo')

    Note:
        The training transform includes data augmentation based on `aug_cfg`, while the validation
        transform performs only the necessary preprocessing (resize, center crop, normalize) without
        any random augmentation.
    """
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        load_weights=load_weights,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        force_context_length=force_context_length,
        pretrained_image=pretrained_image,
        pretrained_text=pretrained_text,
        pretrained_image_path=pretrained_image_path,
        pretrained_text_path=pretrained_text_path,
        cache_dir=cache_dir,
        output_dict=output_dict,
        weights_only=weights_only,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_context_length: Optional[int] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        return_transform: bool = True,
        cache_dir: Optional[str] = None,
        weights_only: bool = True,
        **model_kwargs,
):
    """
    Creates a contrastive vision-language model from pretrained weights with optional preprocessing transform.

    This function is a convenience wrapper around `create_model` that enforces loading of pretrained weights
    (require_pretrained=True) and optionally returns the appropriate preprocessing transform for inference.
    It's designed for use cases where a pretrained model is required, such as feature extraction,
    zero-shot classification, or fine-tuning.

    `model_name` specifies architecture/config source:
      - 'ViT-B-32': Built-in model name. `pretrained` specifies CLIP weights source (tag or file path).
      - 'hf-hub:org/repo': Loads config/weights from HF Hub. `pretrained` is IGNORED.
      - 'local-dir:/path/to/folder': Loads config/weights from local dir. `pretrained` is IGNORED.

    Unlike `create_model`, this function will raise an error if pretrained weights cannot be loaded.

    Args:
        model_name: Model identifier, potentially with schema ('hf-hub:', 'local-dir:').
        pretrained: Source for CLIP weights (tag or file path) ONLY if model_name has no schema.
                   If None and schema requires it, will raise an error.
        precision: Model precision ('fp32', 'fp16', 'bf16', ...).
        device: Device ('cpu', 'cuda', ...).
        jit: If True, JIT compile the model.
        force_quick_gelu: Force use of QuickGELU activation in model config.
        force_custom_text: Force use of custom text encoder architecture.
        force_image_size: Override image size in model config. Useful for using models at different resolutions.
        force_context_length: Override context length in model config.
        image_mean: Override default image normalization mean values (per channel).
        image_std: Override default image normalization std values (per channel).
        image_interpolation: Override default interpolation method for image resizing ('bicubic', 'bilinear', 'nearest').
        image_resize_mode: Override resize mode for inference preprocessing ('squash', 'longest', 'shortest').
            Only affects the returned preprocessing transform, not training.
        return_transform: If True, returns (model, preprocess). If False, returns only model.
        cache_dir: Cache directory for downloads.
        weights_only: Use weights_only=True for torch.load (safer).
        **model_kwargs: Additional keyword arguments for model constructor (highest override priority).

    Returns:
        Union[torch.nn.Module, Tuple[torch.nn.Module, Callable]]:
            - If return_transform=False: Just the model instance
            - If return_transform=True: Tuple of (model, preprocess) where preprocess is the
              inference preprocessing transform

    Raises:
        RuntimeError: If pretrained weights are required but cannot be loaded.

    Example:
        >>> # Load model with preprocessing
        >>> model, preprocess = create_model_from_pretrained('ViT-B-32', pretrained='openai')
        >>>
        >>> # Load model without preprocessing (e.g., when using custom preprocessing)
        >>> model = create_model_from_pretrained('ViT-B-32', pretrained='openai', return_transform=False)
        >>>
        >>> # Load from Hugging Face Hub
        >>> model, preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
        >>>
        >>> # Load with custom image size
        >>> model, preprocess = create_model_from_pretrained(
        ...     'ViT-L-14',
        ...     pretrained='openai',
        ...     force_image_size=336
        ... )

    Note:
        This function always requires pretrained weights to be available and loaded successfully.
        For cases where you want to create a model without pretrained weights or with only
        partial weight loading, use `create_model` or `create_model_and_transforms` instead.
    """
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        force_context_length=force_context_length,
        cache_dir=cache_dir,
        require_pretrained=True,
        weights_only=weights_only,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess

# ===== File: src/open_clip/hf_configs.py =====

# HF architecture dict:
arch_dict = {
    # https://huggingface.co/docs/transformers/model_doc/roberta#roberta
    "roberta": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layer",
            "token_embeddings_attr": "embeddings"
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.XLMRobertaConfig
    "xlm-roberta": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
            "layer_attr": "layer",
            "token_embeddings_attr": "embeddings"
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/mt5#mt5
    "mt5": {
        "config_names": {
            # unlimited seqlen
            # https://github.com/google-research/text-to-text-transfer-transformer/issues/273
            # https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/models/t5/modeling_t5.py#L374
            "context_length": "",
            "vocab_size": "vocab_size",
            "width": "d_model",
            "heads": "num_heads",
            "layers": "num_layers",
            "layer_attr": "block",
            "token_embeddings_attr": "embed_tokens"
        },
        "pooler": "mean_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/bert
    "bert": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "hidden_size",
            "heads": "num_attention_heads",
            "layers": "num_hidden_layers",
        },
        "pooler": "cls_pooler",
    },
    # https://huggingface.co/docs/transformers/model_doc/m2m_100
    "m2m_100": {
        "config_names": {
            "context_length": "max_position_embeddings",
            "vocab_size": "vocab_size",
            "width": "d_model",
            "heads": "encoder_attention_heads",
            "layers": "encoder_layers",
        },
        "pooler": "cls_pooler",
    },
}

# ===== File: src/open_clip/hf_model.py =====

""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re

import torch
import torch.nn as nn
from torch import TensorType

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass

from .hf_configs import arch_dict


# utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


# TODO: ?last - for gpt-like models
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:, self.cls_token_position, :]


class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            model_name_or_path: str,
            output_dim: int,
            config: PretrainedConfig = None,
            pooler_type: str = None,
            proj_type: str = None,
            pretrained: bool = True,
            output_tokens: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim

        # TODO: find better way to get this information
        uses_transformer_pooler = (pooler_type == "cls_pooler")

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            create_func, model_args = (AutoModel.from_pretrained, model_name_or_path) if pretrained else (
                AutoModel.from_config, self.config)
            # TODO: do all model configs have this attribute? PretrainedConfig does so yes??
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler)
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)
        if pooler_type is None:  # get default arch pooler
            pooler_type = (arch_dict[self.config.model_type]["pooler"])

        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        self.vocab_size = getattr(self.config, 'vocab_size', 0)
        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )

    def forward(self, x: TensorType):
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)
        projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]
        tokens = (
            out.last_hidden_state[:, torch.arange(seq_len) != self.pooler.cls_token_position, :] 
            if type(self.pooler) == ClsPooler 
            else out.last_hidden_state
        )
        
        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return

        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"])
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(
            self.transformer, arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass

# ===== File: src/open_clip/loss.py =====

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=self.local_loss,
                gather_with_grad=self.gather_with_grad,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        if logit_bias is not None:
            logits_per_image += logit_bias
            logits_per_text += logit_bias

        return logits_per_image, logits_per_text

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            logit_bias=None,
            output_dict=False,
    ):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(
            image_features,
            text_features,
            logit_scale,
            logit_bias=logit_bias,
        )

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss
        else:
            clip_loss = torch.tensor(0, device=logits.device)

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels: bool = False,
            rank: int = 0,
            world_size: int = 1,
            dist_impl: Optional[str] = None,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        return {"contrastive_loss": loss} if output_dict else loss

# ===== File: src/open_clip/model.py =====

""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from .hf_model import HFTextEncoder
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer,\
    text_global_pool
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    proj_type: str = 'linear'  # control final text projection, 'none' forces no projection
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding'}
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        return no_wd

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            image_indices: Optional[Union[int, List[int]]] = None,
            text_indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize: bool = True,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            image_output_fmt: str = 'NCHW',
            image_output_extra_tokens: bool = False,
            text_output_fmt: str = 'NLC',
            text_output_extra_tokens: bool = False,
            output_logits: bool = False,
            output_logit_scale_bias: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            image: Input image tensor
            text: Input text tensor
            image_indices: For image tower, Take last n blocks if int, all if None, select matching indices if sequence
            text_indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply final norm layer to all intermediates
            normalize: L2 Normalize final features
            intermediates_only: Only return intermediate features, do not return final features
            image_output_fmt: Shape of intermediate image feature outputs
            image_output_extra_tokens: Return both prefix and spatial intermediate tokens
            text_output_fmt: Shape of intermediate text feature outputs (ignored for this model)
            text_output_extra_tokens: Return both prefix and spatial intermediate tokens (ignored for this model)
            output_logits: Include logits in output
            output_logit_scale_bias: Include the logit scale bias in the output
        Returns:

        """
        output = {}
        if intermediates_only:
            # intermediates only disables final feature normalization, and include logits
            normalize = False
            output_logits = False
        if output_logits:
            assert image is not None and text is not None, 'Both image and text inputs are required to compute logits'

        if image is not None:
            image_output = self.visual.forward_intermediates(
                image,
                indices=image_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
                output_extra_tokens=image_output_extra_tokens,
            )
            if normalize and "image_features" in image_output:
                image_output["image_features"] = F.normalize(image_output["image_features"], dim=-1)
            output.update(image_output)

        if text is not None:
            cast_dtype = self.transformer.get_cast_dtype()
            x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.to(cast_dtype)
            x, intermediates = self.transformer.forward_intermediates(
                x,
                attn_mask=self.attn_mask,
                indices=text_indices
            )
            if normalize_intermediates:
                intermediates = [self.ln_final(xi) for xi in intermediates]

            # NOTE this model doesn't support cls embed in text transformer, no need for extra intermediate tokens
            output["text_intermediates"] = intermediates

            if not intermediates_only:
                x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
                x = text_global_pool(x, text, self.text_pool_type)
                if self.text_projection is not None:
                    if isinstance(self.text_projection, nn.Linear):
                        x = self.text_projection(x)
                    else:
                        x = x @ self.text_projection
                if normalize:
                    x = F.normalize(x, dim=-1)
                output["text_features"] = x

        logit_scale_exp = self.logit_scale.exp() if output_logits or output_logit_scale_bias else None

        if output_logits:
            image_logits = logit_scale_exp * output["image_features"] @ output["text_features"].T
            if self.logit_bias is not None:
                image_logits += self.logit_bias
            text_logits = image_logits.T
            output["image_logits"] = image_logits
            output["text_logits"] = text_logits

        if output_logit_scale_bias:
            output["logit_scale"] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias

        return output

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = set()
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        if hasattr(self.text, 'no_weight_decay'):
            for n in self.text.no_weight_decay():
                no_wd.add('text.' + n)
        return no_wd

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            image_indices: Optional[Union[int, List[int]]] = None,
            text_indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize: bool = True,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            image_output_fmt: str = 'NCHW',
            image_output_extra_tokens: bool = False,
            text_output_fmt: str = 'NLC',
            text_output_extra_tokens: bool = False,
            output_logits: bool = False,
            output_logit_scale_bias: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            image: Input image tensor
            text: Input text tensor
            image_indices: For image tower, Take last n blocks if int, all if None, select matching indices if sequence
            text_indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize: L2 Normalize final image and text features (if present)
            normalize_intermediates: Apply final encoder norm layer to all intermediates (if possible)
            intermediates_only: Only return intermediate features, do not return final features
            image_output_fmt: Shape of intermediate image feature outputs
            image_output_extra_tokens: Return both prefix and spatial intermediate tokens
            text_output_fmt: Shape of intermediate text feature outputs
            text_output_extra_tokens: Return both prefix and spatial intermediate tokens
            output_logits: Include logits in output
            output_logit_scale_bias: Include the logit scale bias in the output
        Returns:

        """
        output = {}
        if intermediates_only:
            # intermediates only disables final feature normalization, and include logits
            normalize = False
            output_logits = False
        if output_logits:
            assert image is not None and text is not None, 'Both image and text inputs are required to compute logits'

        if image is not None:
            image_output = self.visual.forward_intermediates(
                image,
                indices=image_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
                output_extra_tokens=image_output_extra_tokens,
            )
            if normalize and "image_features" in image_output:
                image_output["image_features"] = F.normalize(image_output["image_features"], dim=-1)
            output.update(image_output)

        if text is not None:
            text_output = self.text.forward_intermediates(
                text,
                indices=text_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=text_output_fmt,
                output_extra_tokens=text_output_extra_tokens,
            )
            if normalize and "text_features" in text_output:
                text_output["text_features"] = F.normalize(text_output["text_features"], dim=-1)
            output.update(text_output)

        logit_scale_exp = self.logit_scale.exp() if output_logits or output_logit_scale_bias else None

        if output_logits:
            image_logits = logit_scale_exp * output["image_features"] @ output["text_features"].T
            if self.logit_bias is not None:
                image_logits += self.logit_bias
            text_logits = image_logits.T
            output["image_logits"] = image_logits
            output["text_logits"] = text_logits

        if output_logit_scale_bias:
            output["logit_scale"] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias

        return output

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)):
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')):
    model.eval()
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device)
    model = torch.jit.trace_module(
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_text_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = False):
    pos_embed_key = 'positional_embedding' if 'positional_embedding' in state_dict else 'text.positional_embedding'
    old_pos_embed = state_dict.get(pos_embed_key, None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    logging.info('Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict[pos_embed_key] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    module.image_mean = preprocess_cfg['mean']  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model):
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg
# ===== File: src/open_clip/modified_resnet.py =====

from collections import OrderedDict
from typing import Dict, List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F

from .utils import freeze_batch_norm_2d, feature_take_indices


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs antialiasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(
            self,
            layers: List[int],
            output_dim: int,
            heads: int,
            image_size: int = 224,
            width: int = 64,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            output_fmt: str = 'NCHW',
            output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply final norm layer to all intermediates
            intermediates_only: Only return intermediate features
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both extra class, eot tokens
        Returns:

        """
        assert output_fmt in ('NCHW',), 'Output format must be == NCHW.'
        # NOTE normalize_intermediates and return_extra_tokens don't apply
        take_indices, max_index = feature_take_indices(5, indices)

        output = {}
        intermediates = []
        blocks = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                intermediates.append(x)

        output['image_intermediates'] = intermediates

        if intermediates_only:
            return output

        x = self.attnpool(x)
        output['image_features'] = x

        return output

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

# ===== File: src/open_clip/openai.py =====

""" OpenAI pretrained model functions

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""

import os
import warnings
from typing import List, Optional, Union

import torch

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .model import build_model_from_openai_state_dict, convert_weights_to_lp, get_cast_dtype
from .pretrained import get_pretrained_url, list_pretrained_models_by_tag, download_pretrained_from_url

__all__ = ["list_openai_models", "load_openai_model"]


def list_openai_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list_pretrained_models_by_tag('openai')


def load_openai_model(
        name: str,
        precision: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        cache_dir: Optional[str] = None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    precision: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : Union[str, torch.device]
        The device to put the loaded model
    cache_dir : Optional[str]
        The directory to cache the downloaded model weights

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if precision is None:
        precision = 'fp32' if device == 'cpu' else 'fp16'

    if get_pretrained_url(name, 'openai'):
        model_path = download_pretrained_from_url(get_pretrained_url(name, 'openai'), cache_dir=cache_dir)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {list_openai_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        state_dict = torch.load(model_path, map_location="cpu")

    # Build a non-jit model from the OpenAI jitted model state dict
    cast_dtype = get_cast_dtype(precision)
    try:
        model = build_model_from_openai_state_dict(state_dict or model.state_dict(), cast_dtype=cast_dtype)
    except KeyError:
        sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
        model = build_model_from_openai_state_dict(sd, cast_dtype=cast_dtype)

    # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
    model = model.to(device)
    # FIXME support pure fp16/bf16 precision modes
    if precision != 'fp16':
        model.float()
        if precision == 'bf16':
            # for bf16, convert back to low-precision
            convert_weights_to_lp(model, dtype=torch.bfloat16)

    # add mean / std attributes for consistency with OpenCLIP models
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD
    return model

# ===== File: src/open_clip/pos_embed.py =====

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

# ===== File: src/open_clip/pretrained.py =====

import copy
import hashlib
import os
import urllib
import warnings
from functools import partial
from typing import Dict, Iterable, Optional, Union

from tqdm import tqdm


try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False


from .constants import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    INCEPTION_MEAN,
    INCEPTION_STD,
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    HF_WEIGHTS_NAME,
    HF_SAFE_WEIGHTS_NAME,
)
from .version import __version__

try:
    from huggingface_hub import hf_hub_download
    hf_hub_download = partial(hf_hub_download, library_name="open_clip", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


def _pcfg(url='', hf_hub='', **kwargs):
    # OpenAI / OpenCLIP defaults
    return {
        'url': url,
        'hf_hub': hf_hub,
        'mean': OPENAI_DATASET_MEAN,
        'std': OPENAI_DATASET_STD,
        'interpolation': 'bicubic',
        'resize_mode': 'shortest',
        **kwargs,
    }


def _slpcfg(url='', hf_hub='', **kwargs):
    # SiGLIP defaults
    return {
        'url': url,
        'hf_hub': hf_hub,
        'mean': INCEPTION_MEAN,
        'std': INCEPTION_STD,
        'interpolation': 'bicubic',
        'resize_mode': 'squash',
        **kwargs,
    }


def _apcfg(url='', hf_hub='', **kwargs):
    # CLIPA defaults
    return {
        'url': url,
        'hf_hub': hf_hub,
        'mean': IMAGENET_MEAN,
        'std': IMAGENET_STD,
        'interpolation': 'bilinear',
        'resize_mode': 'squash',
        **kwargs,
    }


def _mccfg(url='', hf_hub='', **kwargs):
    # MobileCLIP
    return {
        'url': url,
        'hf_hub': hf_hub,
        'mean': (0., 0., 0.),
        'std': (1., 1., 1.),
        'interpolation': 'bilinear',
        'resize_mode': 'shortest',
        **kwargs,
    }



_RN50 = dict(
    openai=_pcfg(
        url="https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
        hf_hub="timm/resnet50_clip.openai/",
        quick_gelu=True,
    ),
    yfcc15m=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt",
        hf_hub="timm/resnet50_clip.yfcc15m/",
        quick_gelu=True,
    ),
    cc12m=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt",
        hf_hub="timm/resnet50_clip.cc12m/",
        quick_gelu=True,
    ),
)

_RN101 = dict(
    openai=_pcfg(
        url="https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
        hf_hub="timm/resnet101_clip.openai/",
        quick_gelu=True,
    ),
    yfcc15m=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt",
        hf_hub="timm/resnet101_clip.yfcc15m/",
        quick_gelu=True,
    ),
)

_RN50x4 = dict(
    openai=_pcfg(
        url="https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
        hf_hub="timm/resnet50x4_clip.openai/",
        quick_gelu=True,
    ),
)

_RN50x16 = dict(
    openai=_pcfg(
        url="https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
        hf_hub="timm/resnet50x16_clip.openai/",
        quick_gelu=True,
    ),
)

_RN50x64 = dict(
    openai=_pcfg(
        url="https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
        hf_hub="timm/resnet50x64_clip.openai/",
        quick_gelu=True,
    ),
)

_VITB32 = dict(
    openai=_pcfg(
        url="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        hf_hub="timm/vit_base_patch32_clip_224.openai/",
        quick_gelu=True,
    ),
    # LAION 400M (quick gelu)
    laion400m_e31=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
        hf_hub="timm/vit_base_patch32_clip_224.laion400m_e31/",
        quick_gelu=True,
    ),
    laion400m_e32=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
        hf_hub="timm/vit_base_patch32_clip_224.laion400m_e32/",
        quick_gelu=True,
    ),
    # LAION 2B-en
    laion2b_e16=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth",
        hf_hub="timm/vit_base_patch32_clip_224.laion2b_e16/",
    ),
    laion2b_s34b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-laion2B-s34B-b79K/'),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K/'),
    # DataComp-M models
    datacomp_m_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-DataComp.M-s128M-b4K/'),
    commonpool_m_clip_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.clip-s128M-b4K/'),
    commonpool_m_laion_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.laion-s128M-b4K/'),
    commonpool_m_image_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.image-s128M-b4K/'),
    commonpool_m_text_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.text-s128M-b4K/'),
    commonpool_m_basic_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M.basic-s128M-b4K/'),
    commonpool_m_s128m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.M-s128M-b4K/'),
    # DataComp-S models
    datacomp_s_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-DataComp.S-s13M-b4K/'),
    commonpool_s_clip_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.clip-s13M-b4K/'),
    commonpool_s_laion_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.laion-s13M-b4K/'),
    commonpool_s_image_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.image-s13M-b4K/'),
    commonpool_s_text_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.text-s13M-b4K/'),
    commonpool_s_basic_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S.basic-s13M-b4K/'),
    commonpool_s_s13m_b4k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-CommonPool.S-s13M-b4K/'),
    # MetaClip models (NOTE quick-gelu activation used)
    metaclip_400m=_pcfg(
        url="https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_400m.pt",
        hf_hub="timm/vit_base_patch32_clip_224.metaclip_400m/",
        quick_gelu=True,
    ),
    metaclip_fullcc=_pcfg(
        url="https://dl.fbaipublicfiles.com/MMPT/metaclip/b32_fullcc2.5b.pt",
        hf_hub="timm/vit_base_patch32_clip_224.metaclip_2pt5b/",
        quick_gelu=True,
    ),
)

_VITB32_256 = dict(
    datacomp_s34b_b86k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K/'),
)

_VITB16 = dict(
    openai=_pcfg(
        url="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
        hf_hub="timm/vit_base_patch16_clip_224.openai/",
        quick_gelu=True,
    ),
    # LAION-400M
    laion400m_e31=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-00efa78f.pt",
        hf_hub="timm/vit_base_patch16_clip_224.laion400m_e31/",
    ),
    laion400m_e32=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt",
        hf_hub="timm/vit_base_patch16_clip_224.laion400m_e32/",
    ),
    # LAION-2B
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-laion2B-s34B-b88K/'),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-DataComp.XL-s13B-b90K/'),
    # DataComp-L models
    datacomp_l_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-DataComp.L-s1B-b8K/'),
    commonpool_l_clip_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.clip-s1B-b8K/'),
    commonpool_l_laion_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.laion-s1B-b8K/'),
    commonpool_l_image_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.image-s1B-b8K/'),
    commonpool_l_text_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.text-s1B-b8K/'),
    commonpool_l_basic_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L.basic-s1B-b8K/'),
    commonpool_l_s1b_b8k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-CommonPool.L-s1B-b8K/'),
    # DFN
    dfn2b=_pcfg(
        hf_hub='apple/DFN2B-CLIP-ViT-B-16/',
        quick_gelu=True,
    ),
    # MetaCLIP (these are quick-gelu)
    metaclip_400m=_pcfg(
        url="https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_400m.pt",
        hf_hub="timm/vit_base_patch16_clip_224.metaclip_400m/",
        quick_gelu=True,
    ),
    metaclip_fullcc=_pcfg(
        url="https://dl.fbaipublicfiles.com/MMPT/metaclip/b16_fullcc2.5b.pt",
        hf_hub="timm/vit_base_patch16_clip_224.metaclip_2pt5b/",
        quick_gelu=True,
    ),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
        hf_hub="timm/vit_base_patch16_plus_clip_240.laion400m_e31/",
    ),
    laion400m_e32=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e32-699c4b84.pt",
        hf_hub="timm/vit_base_patch16_plus_clip_240.laion400m_e31/",
    ),
)

_VITL14 = dict(
    openai=_pcfg(
        url="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
        hf_hub="timm/vit_large_patch14_clip_224.openai/",
        quick_gelu=True,
    ),
    # LAION-400M
    laion400m_e31=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt",
        hf_hub="timm/vit_large_patch14_clip_224.laion400m_e31/",
    ),
    laion400m_e32=_pcfg(
        url="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt",
        hf_hub="timm/vit_large_patch14_clip_224.laion400m_e32/",
    ),
    # LAION-2B-en
    laion2b_s32b_b82k=_pcfg(
        hf_hub='laion/CLIP-ViT-L-14-laion2B-s32B-b82K/',
        mean=INCEPTION_MEAN, std=INCEPTION_STD),
    # DataComp-XL models
    datacomp_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K/'),
    commonpool_xl_clip_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL.clip-s13B-b90K/'),
    commonpool_xl_laion_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL.laion-s13B-b90K/'),
    commonpool_xl_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K/'),
    # MetaCLIP
    metaclip_400m=_pcfg(
        url="https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_400m.pt",
        hf_hub="timm/vit_large_patch14_clip_224.metaclip_400m/",
        quick_gelu=True,
    ),
    metaclip_fullcc=_pcfg(
        url="https://dl.fbaipublicfiles.com/MMPT/metaclip/l14_fullcc2.5b.pt",
        hf_hub="timm/vit_large_patch14_clip_224.metaclip_2pt5b/",
        quick_gelu=True,
    ),
    # DFN-2B (quick-gelu)
    dfn2b=_pcfg(
        hf_hub='apple/DFN2B-CLIP-ViT-L-14/',
        quick_gelu=True,
    ),
    # DFN-2B 39B SS
    dfn2b_s39b=_pcfg(
        hf_hub='apple/DFN2B-CLIP-ViT-L-14-39B/',
    ),
)

_VITL14_336 = dict(
    openai=_pcfg(
        url="https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
        hf_hub="timm/vit_large_patch14_clip_336.openai/",
        quick_gelu=True,
    ),
)

_VITH14 = dict(
    # LAION-2B-en
    laion2b_s32b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-H-14-laion2B-s32B-b79K/'),
    # MetaCLIP (quick-gelu)
    metaclip_fullcc=_pcfg(
        url="https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_fullcc2.5b.pt",
        hf_hub="timm/vit_huge_patch14_clip_224.metaclip_2pt5b/",
        quick_gelu=True,
    ),
    metaclip_altogether=_pcfg(
        url="https://dl.fbaipublicfiles.com/MMPT/metaclip/h14_v1.2_altogether.pt",
        hf_hub="timm/vit_huge_patch14_clip_224.metaclip_altogether/",
        # NOTE unlike other MetaCLIP models, this is not using QuickGELU, yay!
    ),
    # DFN-5B (quick-gelu)
    dfn5b=_pcfg(
        hf_hub='apple/DFN5B-CLIP-ViT-H-14/',
        quick_gelu=True,
        interpolation="bicubic",
        resize_mode="squash"
    ),
)

_VITH14_378 = dict(
    # DFN-5B (quick-gelu)
    dfn5b=_pcfg(
        hf_hub='apple/DFN5B-CLIP-ViT-H-14-378/',
        quick_gelu=True,
        interpolation="bicubic",
        resize_mode="squash"
    ),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s12B-b42K/'),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s34B-b88K/'),
)

_VITbigG14 = dict(
    # LAION-2B-en
    laion2b_s39b_b160k=_pcfg(hf_hub='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/'),
    # MetaCLIP (quick-gelu)
    metaclip_fullcc=_pcfg(
        url='https://dl.fbaipublicfiles.com/MMPT/metaclip/G14_fullcc2.5b.pt',
        hf_hub="timm/vit_gigantic_patch14_clip_224.metaclip_2pt5b/",
        quick_gelu=True,
    ),
)

_robertaViTB32 = dict(
    laion2b_s12b_b32k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-roberta-base-laion2B-s12B-b32k/'),
)

_xlmRobertaBaseViTB32 = dict(
    laion5b_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k/'),
)

_xlmRobertaLargeFrozenViTH14 = dict(
    frozen_laion5b_s13b_b90k=_pcfg(hf_hub='laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k/'),
)

_convnext_base = dict(
    laion400m_s13b_b51k=_pcfg(hf_hub='laion/CLIP-convnext_base-laion400M-s13B-b51K/'),
)

_convnext_base_w = dict(
    laion2b_s13b_b82k=_pcfg(hf_hub='laion/CLIP-convnext_base_w-laion2B-s13B-b82K/'),
    laion2b_s13b_b82k_augreg=_pcfg(hf_hub='laion/CLIP-convnext_base_w-laion2B-s13B-b82K-augreg/'),
    laion_aesthetic_s13b_b82k=_pcfg(hf_hub='laion/CLIP-convnext_base_w-laion_aesthetic-s13B-b82K/'),
)

_convnext_base_w_320 = dict(
    laion_aesthetic_s13b_b82k=_pcfg(hf_hub='laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K/'),
    laion_aesthetic_s13b_b82k_augreg=_pcfg(hf_hub='laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg/'),
)

_convnext_large_d = dict(
    laion2b_s26b_b102k_augreg=_pcfg(hf_hub='laion/CLIP-convnext_large_d.laion2B-s26B-b102K-augreg/'),
)

_convnext_large_d_320 = dict(
    laion2b_s29b_b131k_ft=_pcfg(hf_hub='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft/'),
    laion2b_s29b_b131k_ft_soup=_pcfg(hf_hub='laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/'),
)

_convnext_xxlarge = dict(
    laion2b_s34b_b82k_augreg=_pcfg(hf_hub='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg/'),
    laion2b_s34b_b82k_augreg_rewind=_pcfg(hf_hub='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-rewind/'),
    laion2b_s34b_b82k_augreg_soup=_pcfg(hf_hub='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/'),
)

_coca_VITB32 = dict(
    laion2b_s13b_b90k=_pcfg(hf_hub='laion/CoCa-ViT-B-32-laion2B-s13B-b90k/'),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(hf_hub='laion/mscoco_finetuned_CoCa-ViT-B-32-laion2B-s13B-b90k/')
)

_coca_VITL14 = dict(
    laion2b_s13b_b90k=_pcfg(hf_hub='laion/CoCa-ViT-L-14-laion2B-s13B-b90k/'),
    mscoco_finetuned_laion2b_s13b_b90k=_pcfg(hf_hub='laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k/')
)


_PRETRAINED = {
    "RN50": _RN50,
    "RN101": _RN101,
    "RN50x4": _RN50x4,
    "RN50x16": _RN50x16,
    "RN50x64": _RN50x64,

    "ViT-B-32": _VITB32,
    "ViT-B-32-256": _VITB32_256,
    "ViT-B-16": _VITB16,
    "ViT-B-16-plus-240": _VITB16_PLUS_240,
    "ViT-L-14": _VITL14,
    "ViT-L-14-336": _VITL14_336,
    "ViT-H-14": _VITH14,
    "ViT-H-14-378": _VITH14_378,
    "ViT-g-14": _VITg14,
    "ViT-bigG-14": _VITbigG14,

    "roberta-ViT-B-32": _robertaViTB32,
    "xlm-roberta-base-ViT-B-32": _xlmRobertaBaseViTB32,
    "xlm-roberta-large-ViT-H-14": _xlmRobertaLargeFrozenViTH14,

    "convnext_base": _convnext_base,
    "convnext_base_w": _convnext_base_w,
    "convnext_base_w_320": _convnext_base_w_320,
    "convnext_large_d": _convnext_large_d,
    "convnext_large_d_320": _convnext_large_d_320,
    "convnext_xxlarge": _convnext_xxlarge,

    "coca_ViT-B-32": _coca_VITB32,
    "coca_ViT-L-14": _coca_VITL14,

    "EVA01-g-14": dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_psz14_s11B.pt
        laion400m_s11b_b41k=_pcfg(hf_hub='timm/eva_giant_patch14_clip_224.laion400m_s11b_b41k/'),
    ),
    "EVA01-g-14-plus": dict(
        # from QuanSun/EVA-CLIP/EVA01_CLIP_g_14_plus_psz14_s11B.pt
        merged2b_s11b_b114k=_pcfg(hf_hub='timm/eva_giant_patch14_plus_clip_224.merged2b_s11b_b114k/'),
    ),
    "EVA02-B-16": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_B_psz16_s8B.pt
        merged2b_s8b_b131k=_pcfg(hf_hub='timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k/'),
    ),
    "EVA02-L-14": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_psz14_s4B.pt
        merged2b_s4b_b131k=_pcfg(hf_hub='timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k/'),
    ),
    "EVA02-L-14-336": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_L_336_psz14_s6B.pt
        merged2b_s6b_b61k=_pcfg(hf_hub='timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k/'),
    ),
    "EVA02-E-14": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_s4B.pt
        laion2b_s4b_b115k=_pcfg(hf_hub='timm/eva02_enormous_patch14_clip_224.laion2b_s4b_b115k/'),
    ),
    "EVA02-E-14-plus": dict(
        # from QuanSun/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt
        laion2b_s9b_b144k=_pcfg(hf_hub='timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k/'),
    ),

    "ViT-B-16-SigLIP": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP/'),
    ),
    "ViT-B-16-SigLIP-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP-256/'),
    ),
    "ViT-B-16-SigLIP-i18n-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP-i18n-256/'),
    ),
    "ViT-B-16-SigLIP-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP-384/'),
    ),
    "ViT-B-16-SigLIP-512": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP-512/'),
    ),
    "ViT-L-16-SigLIP-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-L-16-SigLIP-256/'),
    ),
    "ViT-L-16-SigLIP-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-L-16-SigLIP-384/'),
    ),
    "ViT-SO400M-14-SigLIP": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-14-SigLIP/'),
    ),
    "ViT-SO400M-16-SigLIP-i18n-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-16-SigLIP-i18n-256/'),
    ),
    "ViT-SO400M-14-SigLIP-378": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-14-SigLIP-384/'),  # NOTE using 384 weights, but diff img_size used
    ),
    "ViT-SO400M-14-SigLIP-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-14-SigLIP-384/'),
    ),

    "ViT-B-32-SigLIP2-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-32-SigLIP2-256/'),
    ),
    "ViT-B-16-SigLIP2": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP2/'),
    ),
    "ViT-B-16-SigLIP2-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP2-256/'),
    ),
    "ViT-B-16-SigLIP2-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP2-384/'),
    ),
    "ViT-B-16-SigLIP2-512": dict(
        webli=_slpcfg(hf_hub='timm/ViT-B-16-SigLIP2-512/'),
    ),
    "ViT-L-16-SigLIP2-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-L-16-SigLIP2-256/'),
    ),
    "ViT-L-16-SigLIP2-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-L-16-SigLIP2-384/'),
    ),
    "ViT-L-16-SigLIP2-512": dict(
        webli=_slpcfg(hf_hub='timm/ViT-L-16-SigLIP2-512/'),
    ),
    "ViT-SO400M-14-SigLIP2": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-14-SigLIP2/'),
    ),
    "ViT-SO400M-14-SigLIP2-378": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-14-SigLIP2-378/'),
    ),
    "ViT-SO400M-16-SigLIP2-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-16-SigLIP2-256/'),
    ),
    "ViT-SO400M-16-SigLIP2-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-16-SigLIP2-384/'),
    ),
    "ViT-SO400M-16-SigLIP2-512": dict(
        webli=_slpcfg(hf_hub='timm/ViT-SO400M-16-SigLIP2-512/'),
    ),
    "ViT-gopt-16-SigLIP2-256": dict(
        webli=_slpcfg(hf_hub='timm/ViT-gopt-16-SigLIP2-256/'),
    ),
    "ViT-gopt-16-SigLIP2-384": dict(
        webli=_slpcfg(hf_hub='timm/ViT-gopt-16-SigLIP2-384/'),
    ),

    "ViT-L-14-CLIPA": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-L-14-CLIPA-datacomp1B/'),
    ),
    "ViT-L-14-CLIPA-336": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B/'),
    ),
    "ViT-H-14-CLIPA": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-H-14-CLIPA-datacomp1B/'),
    ),
    "ViT-H-14-CLIPA-336": dict(
        laion2b=_apcfg(hf_hub='UCSC-VLAA/ViT-H-14-CLIPA-336-laion2B/'),
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-H-14-CLIPA-336-datacomp1B/'),
    ),
    "ViT-bigG-14-CLIPA": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-bigG-14-CLIPA-datacomp1B/'),
    ),
    "ViT-bigG-14-CLIPA-336": dict(
        datacomp1b=_apcfg(hf_hub='UCSC-VLAA/ViT-bigG-14-CLIPA-336-datacomp1B/'),
    ),

    "nllb-clip-base": dict(
        v1=_pcfg(hf_hub='visheratin/nllb-clip-base-oc/'),
    ),
    "nllb-clip-large": dict(
        v1=_pcfg(hf_hub='visheratin/nllb-clip-large-oc/'),
    ),

    "nllb-clip-base-siglip": dict(
        v1=_slpcfg(hf_hub='visheratin/nllb-clip-base-siglip/'),
        mrl=_slpcfg(hf_hub='visheratin/nllb-siglip-mrl-base/'),
    ),
    "nllb-clip-large-siglip": dict(
        v1=_slpcfg(hf_hub='visheratin/nllb-clip-large-siglip/'),
        mrl=_slpcfg(hf_hub='visheratin/nllb-siglip-mrl-large/'),
    ),

    "MobileCLIP-S1": dict(
        datacompdr=_mccfg(hf_hub='apple/MobileCLIP-S1-OpenCLIP/')),
    "MobileCLIP-S2": dict(
        datacompdr=_mccfg(hf_hub='apple/MobileCLIP-S2-OpenCLIP/')),
    "MobileCLIP-B": dict(
        datacompdr=_mccfg(hf_hub='apple/MobileCLIP-B-OpenCLIP/'),
        datacompdr_lt=_mccfg(hf_hub='apple/MobileCLIP-B-LT-OpenCLIP/'),
    ),

    "ViTamin-S": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-S/pytorch_model.bin'),
    ),
    "ViTamin-S-LTT": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-S-LTT/pytorch_model.bin'),
    ),
    "ViTamin-B": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-B/pytorch_model.bin'),
    ),
    "ViTamin-B-LTT": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-B-LTT/pytorch_model.bin'),
    ),
    "ViTamin-L": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L-224px/pytorch_model.bin'),
    ),
    "ViTamin-L-256": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L-256px/pytorch_model.bin'),
    ),
    "ViTamin-L-336": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L-336px/pytorch_model.bin'),
    ),
    "ViTamin-L-384": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L-384px/pytorch_model.bin'),
    ),
    "ViTamin-L2": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L2-224px/pytorch_model.bin'),
    ),
    "ViTamin-L2-256": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L2-256px/pytorch_model.bin'),
    ),
    "ViTamin-L2-336": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L2-336px/pytorch_model.bin'),
    ),
    "ViTamin-L2-384": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-L2-384px/pytorch_model.bin'),
    ),
    "ViTamin-XL-256": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-XL-256px/pytorch_model.bin'),
    ),
    "ViTamin-XL-336": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-XL-336px/pytorch_model.bin'),
    ),
    "ViTamin-XL-384": dict(
        datacomp1b=_pcfg(hf_hub='jienengchen/ViTamin-XL-384px/pytorch_model.bin'),
    ),
}

_PRETRAINED_quickgelu = {}
for k, v in _PRETRAINED.items():
    quick_gelu_tags = {}
    for tk, tv in v.items():
        if tv.get('quick_gelu', False):
            quick_gelu_tags[tk] = copy.deepcopy(tv)
    if quick_gelu_tags:
        _PRETRAINED_quickgelu[k + '-quickgelu'] = quick_gelu_tags
_PRETRAINED.update(_PRETRAINED_quickgelu)

def _clean_tag(tag: str):
    # normalize pretrained tags
    return tag.lower().replace('-', '_')


def list_pretrained(as_str: bool = False):
    """ returns list of pretrained models
    Returns a tuple (model_name, pretrain_tag) by default or 'name:tag' if as_str == True
    """
    return [':'.join([k, t]) if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_models_by_tag(tag: str):
    """ return all models having the specified pretrain tag """
    models = []
    tag = _clean_tag(tag)
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models


def list_pretrained_tags_by_model(model: str):
    """ return all pretrain tags for the specified model architecture """
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def is_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return False
    return _clean_tag(tag) in _PRETRAINED[model]


def get_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return {}
    model_pretrained = _PRETRAINED[model]
    return model_pretrained.get(_clean_tag(tag), {})


def get_pretrained_url(model: str, tag: str):
    cfg = get_pretrained_cfg(model, _clean_tag(tag))
    return cfg.get('url', '')


def download_pretrained_from_url(
        url: str,
        cache_dir: Optional[str] = None,
):
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub


def _get_safe_alternatives(filename: str) -> Iterable[str]:
    """Returns potential safetensors alternatives for a given filename.

    Use case:
        When downloading a model from the Huggingface Hub, we first look if a .safetensors file exists and if yes, we use it.
    """
    if filename == HF_WEIGHTS_NAME:
        yield HF_SAFE_WEIGHTS_NAME

    if filename not in (HF_WEIGHTS_NAME,) and (filename.endswith(".bin") or filename.endswith(".pth")):
        yield filename[:-4] + ".safetensors"


def download_pretrained_from_hf(
        model_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
):
    has_hf_hub(True)

    filename = filename or HF_WEIGHTS_NAME

    # Look for .safetensors alternatives and load from it if it exists
    if _has_safetensors:
        for safe_filename in _get_safe_alternatives(filename):
            try:
                cached_file = hf_hub_download(
                    repo_id=model_id,
                    filename=safe_filename,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                return cached_file
            except Exception:
                pass

    try:
        # Attempt to download the file
        cached_file = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        return cached_file  # Return the path to the downloaded file if successful
    except Exception as e:
        raise FileNotFoundError(f"Failed to download file ({filename}) for {model_id}. Last error: {e}")


def download_pretrained(
        cfg: Dict,
        prefer_hf_hub: bool = True,
        cache_dir: Optional[str] = None,
):
    target = ''
    if not cfg:
        return target

    if 'file' in cfg:
        return cfg['file']

    has_hub = has_hf_hub()
    download_url = cfg.get('url', '')
    download_hf_hub = cfg.get('hf_hub', '')
    if has_hub and prefer_hf_hub and download_hf_hub:
        # prefer to use HF hub, remove url info
        download_url = ''

    if download_url:
        target = download_pretrained_from_url(download_url, cache_dir=cache_dir)
    elif download_hf_hub:
        has_hf_hub(True)
        # we assume the hf_hub entries in pretrained config combine model_id + filename in
        # 'org/model_name/filename.pt' form. To specify just the model id w/o filename and
        # use 'open_clip_pytorch_model.bin' default, there must be a trailing slash 'org/model_name/'.
        model_id, filename = os.path.split(download_hf_hub)
        if filename:
            target = download_pretrained_from_hf(model_id, filename=filename, cache_dir=cache_dir)
        else:
            target = download_pretrained_from_hf(model_id, cache_dir=cache_dir)

    return target

# ===== File: src/open_clip/push_to_hf_hub.py =====

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple, Union

import torch

try:
    from huggingface_hub import (
        create_repo,
        get_hf_file_metadata,
        hf_hub_download,
        hf_hub_url,
        repo_type_and_id_from_hf_id,
        upload_folder,
        list_repo_files,
    )
    from huggingface_hub.utils import EntryNotFoundError
    _has_hf_hub = True
except ImportError:
    _has_hf_hub = False

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

from .constants import HF_WEIGHTS_NAME, HF_SAFE_WEIGHTS_NAME, HF_CONFIG_NAME
from .factory import create_model_from_pretrained, get_model_config, get_tokenizer
from .tokenizer import HFTokenizer, SigLipTokenizer


def save_config_for_hf(
        model,
        config_path: str,
        model_config: Optional[dict]
):
    preprocess_cfg = {
        'mean': model.visual.image_mean,
        'std': model.visual.image_std,
    }
    other_pp = getattr(model.visual, 'preprocess_cfg', {})
    if 'interpolation' in other_pp:
        preprocess_cfg['interpolation'] = other_pp['interpolation']
    if 'resize_mode' in other_pp:
        preprocess_cfg['resize_mode'] = other_pp['resize_mode']
    hf_config = {
        'model_cfg': model_config,
        'preprocess_cfg': preprocess_cfg,
    }

    with config_path.open('w') as f:
        json.dump(hf_config, f, indent=2)


def save_for_hf(
    model,
    tokenizer: HFTokenizer,
    model_config: dict,
    save_directory: str,
    safe_serialization: Union[bool, str] = 'both',
    skip_weights : bool = False,
):
    config_filename = HF_CONFIG_NAME

    save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True, parents=True)

    if not skip_weights:
        tensors = model.state_dict()
        if safe_serialization is True or safe_serialization == "both":
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            safetensors.torch.save_file(tensors, save_directory / HF_SAFE_WEIGHTS_NAME)
        if safe_serialization is False or safe_serialization == "both":
            torch.save(tensors, save_directory / HF_WEIGHTS_NAME)

    tokenizer.save_pretrained(save_directory)

    config_path = save_directory / config_filename
    save_config_for_hf(model, config_path, model_config=model_config)


def push_to_hf_hub(
    model,
    tokenizer,
    model_config: Optional[dict],
    repo_id: str,
    commit_message: str = 'Add model',
    token: Optional[str] = None,
    revision: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False,
    model_card: Optional[dict] = None,
    safe_serialization: Union[bool, str] = 'both',
):
    if not isinstance(tokenizer, (HFTokenizer, SigLipTokenizer)):
        # FIXME this makes it awkward to push models with new tokenizers, come up with better soln.
        # default CLIP tokenizers use https://huggingface.co/openai/clip-vit-large-patch14
        tokenizer = HFTokenizer('openai/clip-vit-large-patch14')

    # Create repo if it doesn't exist yet
    repo_url = create_repo(repo_id, token=token, private=private, exist_ok=True)

    # Infer complete repo_id from repo_url
    # Can be different from the input `repo_id` if repo_owner was implicit
    _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)
    repo_id = f"{repo_owner}/{repo_name}"

    # Check if repo already exists and determine what needs updating
    repo_exists = False
    repo_files = {}
    try:
        repo_files = set(list_repo_files(repo_id))
        repo_exists = True
        print('Repo exists', repo_files)
    except Exception as e:
        print('Repo does not exist', e)

    try:
        get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
        has_readme = True
    except EntryNotFoundError:
        has_readme = False

    # Dump model and push to Hub
    with TemporaryDirectory() as tmpdir:
        # Save model weights and config.
        save_for_hf(
            model,
            tokenizer=tokenizer,
            model_config=model_config,
            save_directory=tmpdir,
            safe_serialization=safe_serialization,
        )

        # Add readme if it does not exist
        if not has_readme:
            model_card = model_card or {}
            model_name = repo_id.split('/')[-1]
            readme_path = Path(tmpdir) / "README.md"
            readme_text = generate_readme(model_card, model_name)
            readme_path.write_text(readme_text)

        # Upload model and return
        return upload_folder(
            repo_id=repo_id,
            folder_path=tmpdir,
            revision=revision,
            create_pr=create_pr,
            commit_message=commit_message,
        )


def push_pretrained_to_hf_hub(
    model_name,
    pretrained: str,
    repo_id: str,
    precision: str = 'fp32',
    image_mean: Optional[Tuple[float, ...]] = None,
    image_std: Optional[Tuple[float, ...]] = None,
    image_interpolation: Optional[str] = None,
    image_resize_mode: Optional[str] = None,  # only effective for inference
    commit_message: str = 'Add model',
    token: Optional[str] = None,
    revision: Optional[str] = None,
    private: bool = False,
    create_pr: bool = False,
    model_card: Optional[dict] = None,
    hf_tokenizer_self: bool = False,
    **kwargs,
):
    model, preprocess_eval = create_model_from_pretrained(
        model_name,
        pretrained=pretrained,
        precision=precision,
        image_mean=image_mean,
        image_std=image_std,
        image_interpolation=image_interpolation,
        image_resize_mode=image_resize_mode,
        **kwargs,
    )
    model_config = get_model_config(model_name)
    if pretrained == 'openai':
        model_config['quick_gelu'] = True
    assert model_config

    tokenizer = get_tokenizer(model_name)
    if hf_tokenizer_self:
        # make hf tokenizer config in the uploaded model point to self instead of original location
        model_config['text_cfg']['hf_tokenizer_name'] = repo_id

    push_to_hf_hub(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        repo_id=repo_id,
        commit_message=commit_message,
        token=token,
        revision=revision,
        private=private,
        create_pr=create_pr,
        model_card=model_card,
        safe_serialization='both',
    )


def generate_readme(model_card: dict, model_name: str):
    tags = model_card.pop('tags', ('clip',))
    pipeline_tag = model_card.pop('pipeline_tag', 'zero-shot-image-classification')
    readme_text = "---\n"
    if tags:
        readme_text += "tags:\n"
        for t in tags:
            readme_text += f"- {t}\n"
    readme_text += "library_name: open_clip\n"
    readme_text += f"pipeline_tag: {pipeline_tag}\n"
    readme_text += f"license: {model_card.get('license', 'mit')}\n"
    if 'details' in model_card and 'Dataset' in model_card['details']:
        readme_text += 'datasets:\n'
        readme_text += f"- {model_card['details']['Dataset'].lower()}\n"
    readme_text += "---\n"
    readme_text += f"# Model card for {model_name}\n"
    if 'description' in model_card:
        readme_text += f"\n{model_card['description']}\n"
    if 'details' in model_card:
        readme_text += f"\n## Model Details\n"
        for k, v in model_card['details'].items():
            if isinstance(v, (list, tuple)):
                readme_text += f"- **{k}:**\n"
                for vi in v:
                    readme_text += f"  - {vi}\n"
            elif isinstance(v, dict):
                readme_text += f"- **{k}:**\n"
                for ki, vi in v.items():
                    readme_text += f"  - {ki}: {vi}\n"
            else:
                readme_text += f"- **{k}:** {v}\n"
    if 'usage' in model_card:
        readme_text += f"\n## Model Usage\n"
        readme_text += model_card['usage']
        readme_text += '\n'

    if 'comparison' in model_card:
        readme_text += f"\n## Model Comparison\n"
        readme_text += model_card['comparison']
        readme_text += '\n'

    if 'citation' in model_card:
        readme_text += f"\n## Citation\n"
        if not isinstance(model_card['citation'], (list, tuple)):
            citations = [model_card['citation']]
        else:
            citations = model_card['citation']
        for c in citations:
            readme_text += f"```bibtex\n{c}\n```\n"

    return readme_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push to Hugging Face Hub")
    parser.add_argument(
        "--model", type=str, help="Name of the model to use.",
    )
    parser.add_argument(
        "--pretrained", type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--repo-id", type=str,
        help="Destination HF Hub repo-id ie 'organization/model_id'.",
    )
    parser.add_argument(
        "--precision", type=str, default='fp32',
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="image resize mode during inference"
    )
    parser.add_argument(
        "--hf-tokenizer-self",
        default=False,
        action="store_true",
        help="make hf_tokenizer_name point in uploaded config point to itself"
    )
    args = parser.parse_args()

    print(f'Saving model {args.model} with pretrained weights {args.pretrained} to Hugging Face Hub at {args.repo_id}')

    # FIXME add support to pass model_card json / template from file via cmd line

    push_pretrained_to_hf_hub(
        args.model,
        args.pretrained,
        args.repo_id,
        precision=args.precision,
        image_mean=args.image_mean,  # override image mean/std if trained w/ non defaults
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,
        hf_tokenizer_self=args.hf_tokenizer_self,
    )

    print(f'{args.model} saved.')

# ===== File: src/open_clip/spaglam_model.py =====

# 文件路径: src/open_clip/spaglam_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 导入 torch_geometric 的核心组件
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.glob import global_mean_pool

# SOTA 组件 1: 一个简化的图Transformer块
class GraphTransformerLayer(nn.Module):
    """
    一个图Transformer层，通过全局自注意力捕捉长距离依赖。
    """
    def __init__(self, dim: int, num_heads: int, ffn_expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim)
        )

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        # 虽然是全局注意力，但只在每个子图内部进行，以避免信息在不同样本间泄露
        # 我们通过一个技巧实现：将每个子图视为一个独立的序列
        # 注意：这是一个简化实现。真正的Graphormer会更复杂，但这已抓住了核心思想。
        # 对于大规模图，需要更高效的实现，但对于局部邻域图，这是可行的。
        x_norm = self.norm1(x)
        
        # 为了让MultiheadAttention只在子图内操作，我们需要一个注意力掩码
        # attn_mask (num_graphs, num_nodes, num_nodes)
        num_nodes = x.size(0)
        attn_mask = torch.eq(batch_index.unsqueeze(1), batch_index.unsqueeze(0)).logical_not()
        
        # MultiheadAttention期望的掩码是：True的位置被忽略
        attn_output, _ = self.attn(
            x_norm, x_norm, x_norm, 
            attn_mask=attn_mask,
            need_weights=False
        )
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x

# SOTA 组件 2: 深度模态交互模块
class DeepModalityInteractionBlock(nn.Module):
    """
    通过双向交叉注意力，实现图像和基因模态的深度交互。
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm_img = nn.LayerNorm(dim)
        self.norm_gene = nn.LayerNorm(dim)
        self.cross_attn_i2g = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn_g2i = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # 门控机制，学习融合权重
        self.gate_img = nn.Parameter(torch.tensor([0.0]))
        self.gate_gene = nn.Parameter(torch.tensor([0.0]))
        self.ffn_img = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ffn_gene = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x_img: torch.Tensor, x_gene: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x_img_norm, x_gene_norm = self.norm_img(x_img), self.norm_gene(x_gene)
        
        # 基因模态查询图像模态
        gene_updated, _ = self.cross_attn_i2g(x_gene_norm, x_img_norm, x_img_norm)
        # 图像模态查询基因模态
        img_updated, _ = self.cross_attn_g2i(x_img_norm, x_gene_norm, x_gene_norm)
        
        # 通过门控残差连接进行融合
        x_gene = x_gene + torch.sigmoid(self.gate_gene) * self.ffn_gene(gene_updated)
        x_img = x_img + torch.sigmoid(self.gate_img) * self.ffn_img(img_updated)
        
        return x_img, x_gene

# SOTA 组件 3: 非线性投影头
class MLPProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# --- 主模型：SpaGLaM ---
# src/open_clip/spaglam_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.glob import global_mean_pool

# ... (GraphTransformerLayer, DeepModalityInteractionBlock, MLPProjectionHead classes remain unchanged) ...

# --- 主模型：SpaGLaM (CORRECTED) ---
class SpaGLaM(nn.Module):
    def __init__(self, open_clip_model: nn.Module, config: object):
        super().__init__()
        self.config = config
        self.omiclip_model = open_clip_model

        # 【修复】: 统一属性名，直接使用 'use_precomputed_embeddings'。
        # 这个属性将作为训练时的默认行为。
        self.use_precomputed_embeddings = getattr(config, 'use_precomputed_embeddings', False)
        
        # 冻结策略
        if config.freeze_omiclip:
            for param in self.omiclip_model.parameters():
                param.requires_grad = False
            self.omiclip_model.eval()

        # 获取维度信息
        if hasattr(self.omiclip_model.visual, 'proj') and self.omiclip_model.visual.proj is not None:
             gnn_output_dim = self.omiclip_model.visual.proj.shape[1]
        else:
             gnn_output_dim = self.omiclip_model.visual.output_dim
            
        gnn_input_dim = self.omiclip_model.visual.output_dim
        gnn_hidden_dim = config.gnn_hidden_dim
        
        # 构建并行的GNN塔和交互模块
        self.gnn_layers_img = nn.ModuleList()
        self.gnn_layers_gene = nn.ModuleList()
        self.interaction_layers = nn.ModuleList() if config.use_deep_fusion else None
        
        current_dim = gnn_input_dim
        for _ in range(config.gnn_layers):
            if config.gnn_type == 'gat':
                self.gnn_layers_img.append(GATv2Conv(current_dim, gnn_hidden_dim, heads=config.gnn_heads, concat=False, dropout=0.1))
                self.gnn_layers_gene.append(GATv2Conv(current_dim, gnn_hidden_dim, heads=config.gnn_heads, concat=False, dropout=0.1))
            elif config.gnn_type == 'graphtransformer':
                self.gnn_layers_img.append(GraphTransformerLayer(current_dim, num_heads=config.gnn_heads))
                self.gnn_layers_gene.append(GraphTransformerLayer(current_dim, num_heads=config.gnn_heads))
            else:
                raise ValueError(f"Unknown GNN type: {config.gnn_type}")
            
            if self.interaction_layers is not None:
                self.interaction_layers.append(DeepModalityInteractionBlock(gnn_hidden_dim, num_heads=config.gnn_heads))
            
            current_dim = gnn_hidden_dim
        
        self.image_proj_head = MLPProjectionHead(gnn_hidden_dim, gnn_hidden_dim, gnn_output_dim)
        self.gene_proj_head = MLPProjectionHead(gnn_hidden_dim, gnn_hidden_dim, gnn_output_dim)
        self.logit_scale = self.omiclip_model.logit_scale

    def forward(self, batch: "torch_geometric.data.Batch", use_encoder: Optional[bool] = None) -> dict:
        """
        SpaGLaM 的前向传播。
        """
        # 决定当前前向传播的模式
        if use_encoder is None:
            # 【修复】: 直接引用统一后的属性名 self.use_precomputed_embeddings
            should_encode = not self.use_precomputed_embeddings
        else:
            should_encode = use_encoder

        # 1. 初始特征提取
        if should_encode:
            # 模式 A: 实时编码 (用于推理或原始数据训练)
            # 在推理时，即使 freeze_omiclip=True, 也不应该计算梯度
            is_training_and_not_frozen = self.training and not self.config.freeze_omiclip
            with torch.set_grad_enabled(is_training_and_not_frozen):
                E_image = self.omiclip_model.encode_image(batch.x_image)
                E_gene = self.omiclip_model.encode_text(batch.x_text)
        else:
            # 模式 B: 直接使用预计算的 embeddings (用于高效训练)
            E_image = batch.x_image
            E_gene = batch.x_text

        # 2. GNN传播与深度融合
        img_feat, gene_feat = E_image, E_gene
        for i in range(self.config.gnn_layers):
            if self.config.gnn_type == 'graphtransformer':
                img_feat = self.gnn_layers_img[i](img_feat, batch.batch)
                gene_feat = self.gnn_layers_gene[i](gene_feat, batch.batch)
            else: # GAT
                img_feat = self.gnn_layers_img[i](img_feat, batch.edge_index)
                gene_feat = self.gnn_layers_gene[i](gene_feat, batch.edge_index)
            
            img_feat = F.gelu(img_feat)
            gene_feat = F.gelu(gene_feat)

            if self.interaction_layers is not None:
                img_feat, gene_feat = self.interaction_layers[i](img_feat, gene_feat)
        
        # 3. 读出 (Readout)
        Z_image = global_mean_pool(img_feat, batch.batch)
        Z_gene = global_mean_pool(gene_feat, batch.batch)

        # 4. 投影
        final_image_features = self.image_proj_head(Z_image)
        final_text_features = self.gene_proj_head(Z_gene)

        # 5. 返回
        return {
            "image_features": F.normalize(final_image_features, dim=-1),
            "text_features": F.normalize(final_text_features, dim=-1),
            "logit_scale": self.logit_scale.exp(),
        }
# ===== File: src/open_clip/timm_model.py =====

""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    import timm
    from timm.layers import RotAttentionPool2d
    from timm.layers import AttentionPool2d as AbsAttentionPool2d
    from timm.layers import Mlp, to_2tuple
except ImportError:
    timm = None

from .utils import freeze_batch_norm_2d


class TimmModel(nn.Module):
    """ timm model adapter
    """

    def __init__(
            self,
            model_name: str,
            embed_dim: int,
            image_size: Union[int, Tuple[int, int]] = 224,
            pool: str = 'avg',
            proj: str = 'linear',
            proj_bias: bool = False,
            drop: float = 0.,
            drop_path: Optional[float] = None,
            patch_drop: Optional[float] = None,
            pretrained: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise RuntimeError("Please install the latest timm (`pip install timm`) to use timm based models.")
        self.image_size = to_2tuple(image_size)

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if drop_path is not None:
            timm_kwargs['drop_path_rate'] = drop_path
        if patch_drop is not None:
            timm_kwargs['patch_drop_rate'] = patch_drop

        custom_pool = pool in ('abs_attn', 'rot_attn')
        if proj:
            assert proj in ("linear", "mlp", "none")
        extra_proj = proj in ("linear", "mlp")
        if not extra_proj and not custom_pool:
            # use network classifier head as projection if no proj specified and no custom pooling used
            # if projection is explicitly set to "none" will be pass through from network trunk
            proj_dim = 0 if proj == 'none' else embed_dim
            self.trunk = timm.create_model(
                model_name,
                num_classes=proj_dim,
                global_pool=pool,
                pretrained=pretrained,
                **timm_kwargs,
            )
            prev_chs = embed_dim
        else:
            self.trunk = timm.create_model(
                model_name,
                pretrained=pretrained,
                **timm_kwargs,
            )
            feat_size = self.trunk.default_cfg.get('pool_size', None)
            feature_ndim = 1 if not feat_size else 2
            if custom_pool:
                assert feature_ndim == 2
                # if attn pooling used, remove both classifier and default pool
                self.trunk.reset_classifier(0, global_pool='')
            else:
                # reset global pool if pool config set, otherwise leave as network default
                reset_kwargs = dict(global_pool=pool) if pool else {}
                self.trunk.reset_classifier(0, **reset_kwargs)
            prev_chs = self.trunk.num_features

        head_layers = OrderedDict()

        # Add custom pooling to head
        if pool == 'abs_attn':
            head_layers['pool'] = AbsAttentionPool2d(prev_chs, feat_size=feat_size, out_features=embed_dim)
            prev_chs = embed_dim
        elif pool == 'rot_attn':
            head_layers['pool'] = RotAttentionPool2d(prev_chs, out_features=embed_dim)
            prev_chs = embed_dim

        # NOTE attention pool ends with a projection layer, so proj should usually be set to '' if such pooling is used
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=(drop, 0), bias=(True, proj_bias))

        self.head = nn.Sequential(head_layers)

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        """ lock modules
        Args:
            unlocked_groups (int): leave last n layer groups unlocked (default: 0)
        """
        if not unlocked_groups:
            # lock full model
            for param in self.trunk.parameters():
                param.requires_grad = False
            if freeze_bn_stats:
                freeze_batch_norm_2d(self.trunk)
        else:
            # NOTE: partial freeze requires latest timm (master) branch and is subject to change
            try:
                # FIXME import here until API stable and in an official release
                from timm.models.helpers import group_parameters, group_modules
            except ImportError:
                raise RuntimeError(
                    'Please install latest timm `pip install git+https://github.com/rwightman/pytorch-image-models`')
            matcher = self.trunk.group_matcher()
            gparams = group_parameters(self.trunk, matcher)
            max_layer_id = max(gparams.keys())
            max_layer_id = max_layer_id - unlocked_groups
            for group_idx in range(max_layer_id + 1):
                group = gparams[group_idx]
                for param in group:
                    self.trunk.get_parameter(param).requires_grad = False
            if freeze_bn_stats:
                gmodules = group_modules(self.trunk, matcher, reverse=True)
                gmodules = {k for k, v in gmodules.items() if v <= max_layer_id}
                freeze_batch_norm_2d(self.trunk, gmodules)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception as e:
            logging.warning('grad checkpointing not supported for this timm image tower, continuing without...')

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            output_fmt: str = 'NCHW',
            output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply norm layer to all intermediates
            intermediates_only: Only return intermediate features
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both prefix and spatial intermediate tokens
        Returns:
        """
        extra_args = {}
        if output_extra_tokens:
            extra_args['return_prefix_tokens'] = True
        trunk_output = self.trunk.forward_intermediates(
                x,
                indices=indices,
                intermediates_only=intermediates_only,
                norm=normalize_intermediates,
                stop_early=stop_early,
                output_fmt=output_fmt,
                **extra_args,
            )

        return_dict = {}
        intermediates = trunk_output if intermediates_only else trunk_output[1]
        if output_extra_tokens and intermediates and isinstance(intermediates[0], tuple):
            intermediates_prefix = [xi[1] for xi in intermediates]
            intermediates = [xi[0] for xi in intermediates]
            return_dict['image_intermediates_prefix'] = intermediates_prefix

        return_dict['image_intermediates'] = intermediates
        if intermediates_only:
            return return_dict

        image_features = self.trunk.forward_head(trunk_output[0])  # run through timm pooling / projection
        image_features = self.head(image_features) # run through adapter pooling / projection
        return_dict['image_features'] = image_features
        return return_dict

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x

# ===== File: src/open_clip/tokenizer.py =====

""" CLIP tokenizer

Copied from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import gzip
import html
import os
import random
import string
from functools import lru_cache, partial
from typing import Callable, List, Optional, Union
import warnings

import ftfy
import numpy as np
import regex as re
import torch

# https://stackoverflow.com/q/62691279
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_nltk_init = False

DEFAULT_CONTEXT_LENGTH = 77  # default context length for OpenAI CLIP


@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = " ".join(text.split())
    text = text.strip()
    return text


def _clean_canonicalize(x):
    # basic, remove whitespace, remove punctuation, lower case
    return canonicalize_text(basic_clean(x))


def _clean_lower(x):
    # basic, remove whitespace, lower case
    return whitespace_clean(basic_clean(x)).lower()


def _clean_whitespace(x):
    # basic, remove whitespace
    return whitespace_clean(basic_clean(x))


def get_clean_fn(type: str):
    if type == 'canonicalize':
        return _clean_canonicalize
    elif type == 'lower':
        return _clean_lower
    elif type == 'whitespace':
        return _clean_whitespace
    else:
        assert False, f"Invalid clean function ({type})."


def canonicalize_text(
    text,
    *,
    keep_punctuation_exact_string=None,
    trans_punctuation: dict = str.maketrans("", "", string.punctuation),
):
    """Returns canonicalized `text` (lowercase and punctuation removed).

    From: https://github.com/google-research/big_vision/blob/53f18caf27a9419231bbf08d3388b07671616d3d/big_vision/evaluators/proj/image_text/prompt_engineering.py#L94

    Args:
      text: string to be canonicalized.
      keep_punctuation_exact_string: If provided, then this exact string kept.
        For example providing '{}' will keep any occurrences of '{}' (but will
        still remove '{' and '}' that appear separately).
    """
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(trans_punctuation)
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(trans_punctuation)
    text = text.lower()
    text = " ".join(text.split())
    return text.strip()


class SimpleTokenizer(object):
    def __init__(
            self,
            bpe_path: str = default_bpe(),
            additional_special_tokens: Optional[List[str]] = None,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            clean: str = 'lower',
            reduction_mask: str = ''
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        special_tokens = ['<start_of_text>', '<end_of_text>']
        if additional_special_tokens:
            special_tokens += additional_special_tokens
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t:t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.vocab_size = len(self.encoder)
        self.all_special_ids = [self.encoder[t] for t in special_tokens]
        self.sot_token_id = self.all_special_ids[0]
        self.eot_token_id = self.all_special_ids[1]
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.reduction_fn = get_reduction_mask_fn(reduction_mask) if reduction_mask else None

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = self.clean_fn(text)
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.LongTensor:
        """ Returns the tokenized representation of given input string(s)

        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length'

        if self.reduction_fn is not None:
            # use reduction strategy for tokenize if set, otherwise default to truncation below
            return self.reduction_fn(
                texts,
                context_length=context_length,
                sot_token_id=self.sot_token_id,
                eot_token_id=self.eot_token_id,
                encode_fn=self.encode,
            )

        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = self.eot_token_id
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result


_tokenizer = SimpleTokenizer()


def decode(output_ids: torch.Tensor):
    output_ids = output_ids.cpu().numpy()
    return _tokenizer.decode(output_ids)


def tokenize(texts: Union[str, List[str]], context_length: int = DEFAULT_CONTEXT_LENGTH) -> torch.LongTensor:
    return _tokenizer(texts, context_length=context_length)


def random_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
        shuffle: bool = False,
):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        tokens = torch.tensor(tokens)
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            indices = torch.randperm(len(tokens))
            indices = indices[:num_keep]
            if not shuffle:
                indices = indices.msort()
            tokens = tokens[indices]
            num_tokens = num_keep
        result[i, 0] = sot_token_id
        result[i, 1:num_tokens + 1] = tokens
        result[i, num_tokens + 1] = eot_token_id

    return result


def simple_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
):
    all_tokens = [encode_fn(text) for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        num_tokens = len(tokens)
        if num_tokens > context_length - 2:  # 2 for sot and eot token
            num_keep = context_length - 2
            start_index = random.randint(0, num_tokens - num_keep)  # high is incl
            tokens = tokens[start_index: start_index + num_keep]
        tokens = [sot_token_id] + tokens + [eot_token_id]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def syntax_mask_tokenize(
        texts: Union[str, List[str]],
        context_length: int,
        sot_token_id: int,
        eot_token_id: int,
        encode_fn: Callable,
) -> torch.LongTensor:
    """ Returns the tokenized representation of given input string(s).
    Apply syntax masking before tokenize.
    """
    import nltk
    global _nltk_init
    if not _nltk_init:
        # run them for the first time
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        _nltk_init = True

    def get_order(x):
        if x.startswith('NN'):
            return 1
        elif x.startswith('JJ'):
            return 2
        elif x.startswith('VB'):
            return 3
        else:
            return 4

    # syntax masking
    new_texts = []
    for text in texts:
        list_tokens = nltk.tokenize.word_tokenize(text)
        pos_tags = nltk.pos_tag(list_tokens)
        #  sample the words by get_order method
        order_list = [get_order(tag) for _, tag in pos_tags]
        sorted_ids = np.argsort(np.array(order_list))
        sampled_ids = sorted(sorted_ids[:context_length - 2]) # need 2 slots for sot and eot tokens
        sampled_tokens = np.take(np.array(list_tokens), sampled_ids, axis=0)  # sample the tokens

        new_text = ''
        for token in sampled_tokens:
            new_text = new_text + str(token) + ' '
        new_text = new_text.strip()
        new_texts.append(new_text)
    texts = new_texts

    all_tokens = [[sot_token_id] + encode_fn(text) + [eot_token_id] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        # still need first truncate because some words produces two tokens
        if len(tokens) > context_length:
            tokens = tokens[:context_length]  # Truncate
            tokens[-1] = eot_token_id
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def get_reduction_mask_fn(type: str):
    """ Choose strategy for dropping (masking) tokens to achieve target context length"""
    assert type in ('simple', 'random', 'shuffle', 'syntax')
    if type == 'simple':
        return simple_mask_tokenize  # randomly select block [start:end]
    elif type == 'random':
        return random_mask_tokenize  # randomly drop tokens (keep order)
    elif type == 'shuffle':
        return partial(random_mask_tokenize, shuffle=True)  # randomly drop tokens (shuffle order)
    elif type == 'syntax':
        return syntax_mask_tokenize  # randomly drop prioritized by syntax


class HFTokenizer:
    """HuggingFace tokenizer wrapper"""

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = DEFAULT_CONTEXT_LENGTH,
            clean: str = 'whitespace',
            strip_sep_token: bool = False,
            language: Optional[str] = None,
            cache_dir: Optional[str] = None,
            **kwargs
    ):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir, **kwargs)
        set_lang_fn = getattr(self.tokenizer, 'set_src_lang_special_tokens', None)
        if callable(set_lang_fn):
            self.set_lang_fn = set_lang_fn
        if language is not None:
            self.set_language(language)
        self.context_length = context_length
        self.clean_fn = get_clean_fn(clean)
        self.strip_sep_token = strip_sep_token

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [self.clean_fn(text) for text in texts]
        input_ids = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        ).input_ids

        if self.strip_sep_token:
            input_ids = torch.where(
                input_ids == self.tokenizer.sep_token_id,
                torch.zeros_like(input_ids),
                input_ids,
            )

        return input_ids
    
    def set_language(self, src_lang):
        if hasattr(self, 'set_lang_fn'):
            self.set_lang_fn(src_lang)
        else:
            warnings.warn('Cannot set language for the tokenizer.')


class SigLipTokenizer:
    """HuggingFace tokenizer wrapper for SigLIP T5 compatible sentencepiece vocabs

    NOTE: this is not needed in normal library use, but is used to import new sentencepiece tokenizers
    into OpenCLIP. Leaving code here in case future models use new tokenizers.
    """
    VOCAB_FILES = {
        # english, vocab_size=32_000
        "c4-en": "http://storage.googleapis.com/t5-data/vocabs/cc_en.32000/sentencepiece.model",
        # used in multilingual models (mT5, PaLI), vocab_size=250_000
        "mc4": "http://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",
        # used in SigLIP2 models, vocab_size=256000
        "gemma": "http://storage.googleapis.com/big_vision/gemma_tokenizer.model",
    }

    def __init__(
            self,
            tokenizer_name: str,
            context_length: Optional[int] = 64,
    ):
        if 'gemma' in tokenizer_name:
            from transformers import GemmaTokenizerFast
            tokenizer_cls = partial(
                GemmaTokenizerFast, padding_side='right', add_bos_token=False, add_eos_token=True)
        else:
            from transformers import T5TokenizerFast
            tokenizer_cls = partial(T5TokenizerFast, extra_ids=0)

        if tokenizer_name in self.VOCAB_FILES:
            # FIXME temporary hack?
            import tempfile
            import fsspec
            vocab_file = self.VOCAB_FILES[tokenizer_name]
            with tempfile.NamedTemporaryFile('wb') as dst:
                with fsspec.open(vocab_file, 'rb') as src:
                    dst.write(src.read())
                self.tokenizer = tokenizer_cls(dst.name, legacy=False)
        else:
            self.tokenizer = tokenizer_cls(tokenizer_name, legacy=False)

        self.tokenizer.pad_token_id = 0 if 'gemma' in tokenizer_name else 1
        self.tokenizer.eos_token_id = 1
        self.context_length = context_length

    def save_pretrained(self, dest):
        self.tokenizer.save_pretrained(dest)

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> torch.Tensor:
        # same cleaning as for default tokenizer, except lowercasing
        # adding lower (for case-sensitive tokenizers) will make it more robust but less sensitive to nuance
        if isinstance(texts, str):
            texts = [texts]

        context_length = context_length or self.context_length
        assert context_length, 'Please set a valid context length in class init or call.'

        texts = [canonicalize_text(basic_clean(text)) for text in texts]
        output = self.tokenizer(
            texts,
            return_tensors='pt',
            max_length=context_length,
            padding='max_length',
            truncation=True,
        )
        return output.input_ids

# ===== File: src/open_clip/transform.py =====

import numbers
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop, ColorJitter, Grayscale

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .utils import to_2tuple


@dataclass
class PreprocessCfg:
    size: Union[int, Tuple[int, int]] = 224
    mode: str = 'RGB'
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN
    std: Tuple[float, ...] = OPENAI_DATASET_STD
    interpolation: str = 'bicubic'
    resize_mode: str = 'shortest'
    fill_color: int = 0

    def __post_init__(self):
        assert self.mode in ('RGB',)

    @property
    def num_channels(self):
        return 3

    @property
    def input_size(self):
        return (self.num_channels,) + to_2tuple(self.size)

_PREPROCESS_KEYS = set(asdict(PreprocessCfg()).keys())


def merge_preprocess_dict(
        base: Union[PreprocessCfg, Dict],
        overlay: Dict,
):
    """ Merge overlay key-value pairs on top of base preprocess cfg or dict.
    Input dicts are filtered based on PreprocessCfg fields.
    """
    if isinstance(base, PreprocessCfg):
        base_clean = asdict(base)
    else:
        base_clean = {k: v for k, v in base.items() if k in _PREPROCESS_KEYS}
    if overlay:
        overlay_clean = {k: v for k, v in overlay.items() if k in _PREPROCESS_KEYS and v is not None}
        base_clean.update(overlay_clean)
    return base_clean


def merge_preprocess_kwargs(base: PreprocessCfg, **kwargs):
    return merge_preprocess_dict(base, kwargs)


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False

    # params for simclr_jitter_gray
    color_jitter_prob: float = None
    gray_scale_prob: float = None


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class ResizeKeepRatio:
    """ Resize and Keep Ratio

    Copy & paste from `timm`
    """

    def __init__(
            self,
            size,
            longest=0.,
            interpolation=InterpolationMode.BICUBIC,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        if isinstance(size, (list, tuple)):
            self.size = tuple(size)
        else:
            self.size = (size, size)
        self.interpolation = interpolation
        self.longest = float(longest)  # [0, 1] where 0 == shortest edge, 1 == longest
        self.random_scale_prob = random_scale_prob
        self.random_scale_range = random_scale_range
        self.random_aspect_prob = random_aspect_prob
        self.random_aspect_range = random_aspect_range

    @staticmethod
    def get_params(
            img,
            target_size,
            longest,
            random_scale_prob=0.,
            random_scale_range=(0.85, 1.05),
            random_aspect_prob=0.,
            random_aspect_range=(0.9, 1.11)
    ):
        """Get parameters
        """
        source_size = img.size[::-1]  # h, w
        h, w = source_size
        target_h, target_w = target_size
        ratio_h = h / target_h
        ratio_w = w / target_w
        ratio = max(ratio_h, ratio_w) * longest + min(ratio_h, ratio_w) * (1. - longest)
        if random_scale_prob > 0 and random.random() < random_scale_prob:
            ratio_factor = random.uniform(random_scale_range[0], random_scale_range[1])
            ratio_factor = (ratio_factor, ratio_factor)
        else:
            ratio_factor = (1., 1.)
        if random_aspect_prob > 0 and random.random() < random_aspect_prob:
            aspect_factor = random.uniform(random_aspect_range[0], random_aspect_range[1])
            ratio_factor = (ratio_factor[0] / aspect_factor, ratio_factor[1] * aspect_factor)
        size = [round(x * f / ratio) for x, f in zip(source_size, ratio_factor)]
        return size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Resized, padded to at least target size, possibly cropped to exactly target size
        """
        size = self.get_params(
            img, self.size, self.longest,
            self.random_scale_prob, self.random_scale_range,
            self.random_aspect_prob, self.random_aspect_range
        )
        img = F.resize(img, size, self.interpolation)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += f', interpolation={self.interpolation})'
        format_string += f', longest={self.longest:.3f})'
        return format_string


def center_crop_or_pad(img: torch.Tensor, output_size: List[int], fill=0) -> torch.Tensor:
    """Center crops and/or pads the given image.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.
        fill (int, Tuple[int]): Padding color

    Returns:
        PIL Image or Tensor: Cropped image.
    """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    _, image_height, image_width = F.get_dimensions(img)
    crop_height, crop_width = output_size

    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img = F.pad(img, padding_ltrb, fill=fill)
        _, image_height, image_width = F.get_dimensions(img)
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.0))
    crop_left = int(round((image_width - crop_width) / 2.0))
    return F.crop(img, crop_top, crop_left, crop_height, crop_width)


class CenterCropOrPad(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self, size, fill=0):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.fill = fill

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        return center_crop_or_pad(img, self.size, fill=self.fill)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


def _convert_to_rgb(image):
    return image.convert('RGB')


class color_jitter(object):
    """
    Apply Color Jitter to the PIL image with a specified probability.
    """
    def __init__(self, brightness=0., contrast=0., saturation=0., hue=0., p=0.8):
        assert 0. <= p <= 1.
        self.p = p
        self.transf = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


class gray_scale(object):
    """
    Apply Gray Scale to the PIL image with a specified probability.
    """
    def __init__(self, p=0.2):
        assert 0. <= p <= 1.
        self.p = p
        self.transf = Grayscale(num_output_channels=3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def image_transform(
        image_size: Union[int, Tuple[int, int]],
        is_train: bool,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        resize_mode: Optional[str] = None,
        interpolation: Optional[str] = None,
        fill_color: int = 0,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    interpolation = interpolation or 'bicubic'
    assert interpolation in ['bicubic', 'bilinear', 'random']
    # NOTE random is ignored for interpolation_mode, so defaults to BICUBIC for inference if set
    interpolation_mode = InterpolationMode.BILINEAR if interpolation == 'bilinear' else InterpolationMode.BICUBIC

    resize_mode = resize_mode or 'shortest'
    assert resize_mode in ('shortest', 'longest', 'squash')

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()

    normalize = Normalize(mean=mean, std=std)

    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop('use_timm', False)
        if use_timm:
            from timm.data import create_transform  # timm can still be optional
            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)

            aug_cfg_dict.setdefault('color_jitter', None)  # disable by default
            # drop extra non-timm items
            aug_cfg_dict.pop('color_jitter_prob', None)
            aug_cfg_dict.pop('gray_scale_prob', None)

            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.,
                mean=mean,
                std=std,
                re_mode='pixel',
                interpolation=interpolation,
                **aug_cfg_dict,
            )
        else:
            train_transform = [
                RandomResizedCrop(
                    image_size,
                    scale=aug_cfg_dict.pop('scale'),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                _convert_to_rgb,
            ]
            if aug_cfg.color_jitter_prob:
                assert aug_cfg.color_jitter is not None and len(aug_cfg.color_jitter) == 4
                train_transform.extend([
                    color_jitter(*aug_cfg.color_jitter, p=aug_cfg.color_jitter_prob)
                ])
            if aug_cfg.gray_scale_prob:
                train_transform.extend([
                    gray_scale(aug_cfg.gray_scale_prob)
                ])
            train_transform.extend([
                ToTensor(),
                normalize,
            ])
            train_transform = Compose(train_transform)
            if aug_cfg_dict:
                warnings.warn(f'Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())}).')
        return train_transform
    else:
        if resize_mode == 'longest':
            transforms = [
                ResizeKeepRatio(image_size, interpolation=interpolation_mode, longest=1),
                CenterCropOrPad(image_size, fill=fill_color)
            ]
        elif resize_mode == 'squash':
            if isinstance(image_size, int):
                image_size = (image_size, image_size)
            transforms = [
                Resize(image_size, interpolation=interpolation_mode),
            ]
        else:
            assert resize_mode == 'shortest'
            if not isinstance(image_size, (tuple, list)):
                image_size = (image_size, image_size)
            if image_size[0] == image_size[1]:
                # simple case, use torchvision built-in Resize w/ shortest edge mode (scalar size arg)
                transforms = [
                    Resize(image_size[0], interpolation=interpolation_mode)
                ]
            else:
                # resize shortest edge to matching target dim for non-square target
                transforms = [ResizeKeepRatio(image_size)]
            transforms += [CenterCrop(image_size)]

        transforms.extend([
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
        return Compose(transforms)


def image_transform_v2(
        cfg: PreprocessCfg,
        is_train: bool,
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    return image_transform(
        image_size=cfg.size,
        is_train=is_train,
        mean=cfg.mean,
        std=cfg.std,
        interpolation=cfg.interpolation,
        resize_mode=cfg.resize_mode,
        fill_color=cfg.fill_color,
        aug_cfg=aug_cfg,
    )

# ===== File: src/open_clip/transformer.py =====

from collections import OrderedDict
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .utils import to_2tuple, feature_take_indices
from .pos_embed import get_2d_sincos_pos_embed


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(
            self,
            prob: float = 0.5,
            exclude_first_token: bool = True
    ):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            scaled_cosine: bool = False,
            scale_heads: bool = False,
            logit_scale_max: float = math.log(1. / 0.01),
            batch_first: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max
        self.batch_first = batch_first
        self.use_fsdpa = hasattr(nn.functional, 'scaled_dot_product_attention')

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.reshape(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.reshape(L, N * self.num_heads, -1).transpose(0, 1)

        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
            if attn_mask is not None:
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = torch.bmm(attn, v)
        else:
            if self.use_fsdpa:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )
            else:
                q = q * self.scale
                attn = torch.bmm(q, k.transpose(-1, -2))
                if attn_mask is not None:
                    attn += attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = torch.bmm(attn, v)

        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)

        x = x.transpose(0, 1).reshape(L, N, C)

        if self.batch_first:
            x = x.transpose(0, 1)

        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=batch_first)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            batch_first: bool = True,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model,
            n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
            batch_first=batch_first,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def get_reference_weight(self):
        return self.mlp.c_fc.weight

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomTransformer(nn.Module):
    """ A custom transformer that can use different block types. """
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
            block_types: Union[str, List[str]] = 'CustomResidualAttentionBlock',
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first  # run transformer stack in batch first (N, L, D)
        self.grad_checkpointing = False

        if isinstance(block_types, str):
            block_types = [block_types] * layers
        assert len(block_types) == layers

        def _create_block(bt: str):
            if bt == 'CustomResidualAttentionBlock':
                return CustomResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                )
            else:
                assert False

        self.resblocks = nn.ModuleList([
            _create_block(bt)
            for bt in block_types
        ])

    def get_cast_dtype(self) -> torch.dtype:
        weight = self.resblocks[0].get_reference_weight()
        if hasattr(weight, 'int8_original_dtype'):
            return weight.int8_original_dtype
        return weight.dtype

    def forward_intermediates(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
    ):
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()  # NLD -> LND

        intermediates = []
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.resblocks
        else:
            blocks = self.resblocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x.transpose(0, 1) if not self.batch_first else x)

        if not self.batch_first:
            x = x.transpose(0, 1)  # LND -> NLD

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index + 1]  # truncate blocks
        return take_indices

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)  # NLD -> LND
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            batch_first: bool = True,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.batch_first = batch_first
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward_intermediates(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
    ):
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)

        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND

        intermediates = []
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.resblocks
        else:
            blocks = self.resblocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(blk, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask)

            if i in take_indices:
                intermediates.append(x.transpose(0, 1) if not self.batch_first else x)

        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD

        return x, intermediates

    def prune_intermediate_layers(self, indices: Union[int, List[int]] = 1):
        """ Prune layers not required for specified intermediates.
        """
        take_indices, max_index = feature_take_indices(len(self.resblocks), indices)
        self.resblocks = self.resblocks[:max_index + 1]  # truncate blocks
        return take_indices

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        if not self.batch_first:
            x = x.transpose(0, 1).contiguous()    # NLD -> LND

        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = r(x, attn_mask=attn_mask)

        if not self.batch_first:
            x = x.transpose(0, 1)    # LND -> NLD
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1],\
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups: int = 0, freeze_bn_stats: bool = False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding', 'class_embedding'}
        return no_wd

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def _embeds(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)  # shape = [*, dim, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # patch dropout (if active)
        x = self.patch_dropout(x)

        # apply norm before transformer
        x = self.ln_pre(x)
        return x

    def _pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.attn_pool is not None:
            if self.attn_pool_contrastive is not None:
                # This is untested, WIP pooling that should match paper
                x = self.ln_post(x)  # TBD LN first or separate one after each pool?
                tokens = self.attn_pool(x)
                if self.attn_pool_type == 'parallel':
                    pooled = self.attn_pool_contrastive(x)
                else:
                    assert self.attn_pool_type == 'cascade'
                    pooled = self.attn_pool_contrastive(tokens)
            else:
                # this is the original OpenCLIP CoCa setup, does not match paper
                x = self.attn_pool(x)
                x = self.ln_post(x)
                pooled, tokens = self._global_pool(x)
        elif self.final_ln_after_pool:
            pooled, tokens = self._global_pool(x)
            pooled = self.ln_post(pooled)
        else:
            x = self.ln_post(x)
            pooled, tokens = self._global_pool(x)

        return pooled, tokens

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            output_fmt: str = 'NCHW',
            output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            intermediates_only: Only return intermediate features
            normalize_intermediates: Apply final norm layer to all intermediates
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both extra prefix class tokens
        Returns:

        """
        assert output_fmt in ('NCHW', 'NLC'), 'Output format must be one of NCHW or NLC.'
        reshape = output_fmt == 'NCHW'

        # forward pass
        B, _, height, width = x.shape
        x = self._embeds(x)
        x, intermediates = self.transformer.forward_intermediates(
            x,
            indices=indices,
            stop_early=stop_early,
        )

        # process intermediates
        if normalize_intermediates:
            # apply final norm to all intermediates
            intermediates = [self.ln_post(xi) for xi in intermediates]
        num_prefix_tokens = 1  # one class token that's always there (as of now)
        if num_prefix_tokens:
            # split prefix (e.g. class, distill) and spatial feature tokens
            prefix_tokens = [y[:, 0:num_prefix_tokens] for y in intermediates]
            intermediates = [y[:, num_prefix_tokens:] for y in intermediates]
        else:
            prefix_tokens = None
        if reshape:
            # reshape to BCHW output format
            H, W = height // self.patch_size[0], width // self.patch_size[1]
            intermediates = [y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() for y in intermediates]

        output = {'image_intermediates': intermediates}
        if prefix_tokens is not None and output_extra_tokens:
            output['image_intermediates_prefix'] = prefix_tokens

        if intermediates_only:
            return output

        pooled, _ = self._pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        output['image_features'] = pooled

        return output

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices = self.transformer.prune_intermediate_layers(indices)
        if prune_norm:
            self.ln_post = nn.Identity()
        if prune_head:
            self.proj = None
        return take_indices

    def forward(self, x: torch.Tensor):
        x = self._embeds(x)
        x = self.transformer(x)
        pooled, tokens = self._pool(x)

        if self.proj is not None:
            pooled = pooled @ self.proj

        if self.output_tokens:
            return pooled, tokens
        
        return pooled


def text_global_pool(
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        pool_type: str = 'argmax',
) -> torch.Tensor:
    if pool_type == 'first':
        pooled = x[:, 0]
    elif pool_type == 'last':
        pooled = x[:, -1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
    else:
        pooled = x

    return pooled


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: Optional[int] = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            pad_id: int = 0,
            pool_type: str = 'argmax',
            proj_type: str = 'linear',
            proj_bias: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

        if proj_type == 'none' or not output_dim:
            self.text_projection = None
        else:
            if proj_bias:
                self.text_projection = nn.Linear(width, output_dim)
            else:
                self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding'}
        if self.cls_emb is not None:
            no_wd.add('cls_emb')
        return no_wd

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def _embeds(self, text) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]
        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        return x, attn_mask

    def forward_intermediates(
            self,
            text: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            output_fmt: str = 'NCHW',
            output_extra_tokens: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            text: Input text ids
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply norm layer to all intermediates
            intermediates_only: Only return intermediate features
            output_fmt: Shape of intermediate feature outputs
            output_extra_tokens: Return both prefix and intermediate tokens
        Returns:

        """
        assert output_fmt in ('NLC',), 'Output format must be NLC.'
        # forward pass
        x, attn_mask = self._embeds(text)
        x, intermediates = self.transformer.forward_intermediates(
            x,
            attn_mask=attn_mask,
            indices=indices,
            stop_early=stop_early,
        )

        # process intermediates
        if normalize_intermediates:
            # apply final norm to all intermediates
            intermediates = [self.ln_final(xi) for xi in intermediates]

        output = {}

        if self.cls_emb is not None:
            seq_intermediates = [xi[:, :-1] for xi in intermediates]  # separate concat'd class token from sequence
            if output_extra_tokens:
                # return suffix class tokens separately
                cls_intermediates = [xi[:, -1:] for xi in intermediates]
                output['text_intermediates_suffix'] = cls_intermediates
            intermediates = seq_intermediates
        output['text_intermediates'] = intermediates

        if intermediates_only:
            return output

        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        output['text_features'] = pooled

        return output

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
    ):
        """ Prune layers not required for specified intermediates.
        """
        take_indices = self.transformer.prune_intermediate_layers(indices)
        if prune_norm:
            self.ln_final = nn.Identity()
        if prune_head:
            self.text_projection = None
        return take_indices

    def forward(self, text):
        x, attn_mask = self._embeds(text)

        x = self.transformer(x, attn_mask=attn_mask)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
            tokens = x[:, :-1]
        else:
            x = self.ln_final(x)
            pooled = text_global_pool(x, text, pool_type=self.pool_type)
            tokens = x

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
            batch_first: bool = True,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=batch_first,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
                batch_first=batch_first,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward_intermediates(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
    ):
        assert False, "Not currently implemented for MultimodalTransformer w/ xattn"

    def forward(self, image_embs, text_embs):
        seq_len = text_embs.shape[1]
        if not self.batch_first:
            image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
            text_embs = text_embs.permute(1, 0, 2)  # NLD -> LND

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(
                    resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len], use_reentrant=False)
                text_embs = checkpoint(
                    cross_attn, text_embs, image_embs, image_embs, None, use_reentrant=False)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        if not self.batch_first:
            text_embs = text_embs.permute(1, 0, 2)  # LND -> NLD

        out = self.ln_final(text_embs)
        if self.text_projection is not None:
            out = out @ self.text_projection

        return out

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

# ===== File: src/open_clip/utils.py =====

import collections.abc
from itertools import repeat
from typing import List, Optional, Tuple, Union

import torch
from torch import nn as nn
from torch import _assert
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)

# Replaces all linear layers with linear_replacement
# TODO: add int8 support for other linear layers including attn and convnets
def replace_linear(model, linear_replacement, include_modules=['c_fc', 'c_proj'], copy_weights=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, include_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name in include_modules:
            old_module = model._modules[name]
            model._modules[name] = linear_replacement(
                module.in_features,
                module.out_features,
                module.bias is not None,
            )
            if copy_weights:
                model._modules[name].weight.data.copy_(old_module.weight.data)
                if model._modules[name].bias is not None:
                    model._modules[name].bias.data.copy_(old_module.bias)

    return model

def convert_int8_model_to_inference_mode(model):
    for m in model.modules():
        if hasattr(m, 'prepare_for_eval'):
            int8_original_dtype = m.weight.dtype
            m.prepare_for_eval()
            m.int8_original_dtype = int8_original_dtype


def feature_take_indices(
        num_features: int,
        indices: Optional[Union[int, List[int]]] = None,
        as_set: bool = False,
) -> Tuple[List[int], int]:
    """ Determine the absolute feature indices to 'take' from.

    Note: This function can be called in forward() so must be torchscript compatible,
    which requires some incomplete typing and workaround hacks.

    Args:
        num_features: total number of features to select from
        indices: indices to select,
          None -> select all
          int -> select last n
          list/tuple of int -> return specified (-ve indices specify from end)
        as_set: return as a set

    Returns:
        List (or set) of absolute (from beginning) indices, Maximum index
    """
    if indices is None:
        indices = num_features  # all features if None

    if isinstance(indices, int):
        # convert int -> last n indices
        _assert(0 < indices <= num_features, f'last-n ({indices}) is out of range (1 to {num_features})')
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            _assert(0 <= idx < num_features, f'feature index {idx} is out of range (0 to {num_features - 1})')
            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)


def _out_indices_as_tuple(x: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(x, int):
        # if indices is an int, take last N features
        return tuple(range(-x, 0))
    return tuple(x)

# ===== File: src/open_clip/version.py =====

__version__ = '2.32.0'

# ===== File: src/open_clip/zero_shot_classifier.py =====

from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F


def batched(iterable, n):
    """Batch data into lists of length *n*. The last batch may be shorter.
    NOTE based on more-itertools impl, to be replaced by python 3.12 itertools.batched impl
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def build_zero_shot_classifier(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        num_classes_per_batch: Optional[int] = 10,
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names in batches
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        num_classes_per_batch: The number of classes to batch together in each forward, all if None
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    use_format = isinstance(templates[0], str)
    num_templates = len(templates)
    num_classes = len(classnames)
    if use_tqdm:
        import tqdm
        num_iter = 1 if num_classes_per_batch is None else ((num_classes - 1) // num_classes_per_batch + 1)
        iter_wrap = partial(tqdm.tqdm, total=num_iter, unit_scale=num_classes_per_batch)
    else:
        iter_wrap = iter

    def _process_batch(batch_classnames):
        num_batch_classes = len(batch_classnames)
        texts = [template.format(c) if use_format else template(c) for c in batch_classnames for template in templates]
        texts = tokenizer(texts).to(device)
        class_embeddings = model.encode_text(texts, normalize=True)
        class_embeddings = class_embeddings.reshape(num_batch_classes, num_templates, -1).mean(dim=1)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=1, keepdim=True)
        class_embeddings = class_embeddings.T
        return class_embeddings

    with torch.no_grad():
        if num_classes_per_batch:
            batched_embeds = [_process_batch(batch) for batch in iter_wrap(batched(classnames, num_classes_per_batch))]
            zeroshot_weights = torch.cat(batched_embeds, dim=1)
        else:
            zeroshot_weights = _process_batch(classnames)
    return zeroshot_weights


def build_zero_shot_classifier_legacy(
        model,
        tokenizer,
        classnames: Sequence[str],
        templates: Sequence[Union[Callable, str]],
        device: Union[str, torch.device] = 'cpu',
        use_tqdm: bool = False,
):
    """ Build zero-shot classifier weights by iterating over class names 1 by 1
    Args:
        model: CLIP model instance
        tokenizer: CLIP tokenizer instance
        classnames: A sequence of class (label) names
        templates: A sequence of callables or format() friendly strings to produce templates per class name
        device: Device to use.
        use_tqdm: Enable TQDM progress bar.
    """
    assert isinstance(templates, Sequence) and len(templates) > 0
    assert isinstance(classnames, Sequence) and len(classnames) > 0
    if use_tqdm:
        import tqdm
        iter_wrap = tqdm.tqdm
    else:
        iter_wrap = iter

    use_format = isinstance(templates[0], str)

    with torch.no_grad():
        zeroshot_weights = []
        for classname in iter_wrap(classnames):
            texts = [template.format(classname) if use_format else template(classname) for template in templates]
            texts = tokenizer(texts).to(device)  # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)

    return zeroshot_weights


# ===== File: src/open_clip/zero_shot_metadata.py =====


OPENAI_IMAGENET_TEMPLATES = (
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
)


# a much smaller subset of above prompts
# from https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f'itap of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'art of the {c}.',
    lambda c: f'a photo of the small {c}.',
)


IMAGENET_CLASSNAMES = (
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
    "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
    "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
    "box turtle", "banded gecko", "green iguana", "Carolina anole",
    "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
    "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
    "American alligator", "triceratops", "worm snake", "ring-necked snake",
    "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
    "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
    "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
    "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
    "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
    "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
    "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
    "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
    "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
    "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
    "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
    "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
    "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
    "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
    "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
    "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
    "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
    "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
    "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
    "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
    "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
    "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
    "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
    "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
    "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
    "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
    "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
    "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
    "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
    "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
    "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
    "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
    "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
    "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
    "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
    "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
    "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
    "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
    "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
    "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
    "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
    "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
    "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
    "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
    "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
    "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
    "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
    "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
    "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
    "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
    "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
    "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
    "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
    "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
    "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
    "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
    "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
    "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
    "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
    "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
    "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
    "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
    "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
    "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
    "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
    "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
    "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
    "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
    "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
    "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
    "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
    "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
    "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
    "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
    "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
    "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
    "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
    "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
    "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
    "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
    "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
    "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
    "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
    "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
    "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
    "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
    "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
    "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
    "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
    "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
    "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
    "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
    "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
    "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
    "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
    "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
    "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
    "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
    "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
    "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
    "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
    "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
    "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
    "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
    "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
    "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
    "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
    "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
    "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
    "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
    "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
    "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
    "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
    "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
)


# ===== File: src/spaglam_preproc/__init__.py =====

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

# ===== File: src/spaglam_preproc/cli.py =====

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

# ===== File: src/spaglam_preproc/config.py =====

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

# ===== File: src/spaglam_preproc/core/__init__.py =====


# ===== File: src/spaglam_preproc/core/dataset_writer.py =====

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

# ===== File: src/spaglam_preproc/core/gene_encoder.py =====

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

# ===== File: src/spaglam_preproc/core/graph_builder.py =====

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

# ===== File: src/spaglam_preproc/core/image_tiler.py =====

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

# ===== File: src/spaglam_preproc/utils/__init__.py =====


# ===== File: src/spaglam_preproc/utils/anndata_utils.py =====

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

# ===== File: src/spaglam_preproc/utils/logging_setup.py =====

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

# ===== File: src/spaglam_preproc/utils/qc_tools.py =====

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

# ===== File: src/spaglam_preproc/utils/validation.py =====

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

# ===== File: tests/test_download_pretrained.py =====

import requests
import torch
from PIL import Image
import hashlib
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from urllib3 import HTTPResponse
from urllib3._collections import HTTPHeaderDict

import open_clip
from open_clip.pretrained import download_pretrained_from_url


class DownloadPretrainedTests(unittest.TestCase):

    def create_response(self, data, status_code=200, content_type='application/octet-stream'):
        fp = BytesIO(data)
        headers = HTTPHeaderDict({
            'Content-Type': content_type,
            'Content-Length': str(len(data))
        })
        raw = HTTPResponse(fp, preload_content=False, headers=headers, status=status_code)
        return raw

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_openaipublic(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()
        urllib.request.urlopen.return_value = self.create_response(file_contents)
        with tempfile.TemporaryDirectory() as root:
            url = f'https://openaipublic.azureedge.net/clip/models/{expected_hash}/RN50.pt'
            download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_openaipublic_corrupted(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()
        urllib.request.urlopen.return_value = self.create_response(b'corrupted pretrained model')
        with tempfile.TemporaryDirectory() as root:
            url = f'https://openaipublic.azureedge.net/clip/models/{expected_hash}/RN50.pt'
            with self.assertRaisesRegex(RuntimeError, r'checksum does not not match'):
                download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_openaipublic_valid_cache(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()
        urllib.request.urlopen.return_value = self.create_response(file_contents)
        with tempfile.TemporaryDirectory() as root:
            local_file = Path(root) / 'RN50.pt'
            local_file.write_bytes(file_contents)
            url = f'https://openaipublic.azureedge.net/clip/models/{expected_hash}/RN50.pt'
            download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_not_called()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_openaipublic_corrupted_cache(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()
        urllib.request.urlopen.return_value = self.create_response(file_contents)
        with tempfile.TemporaryDirectory() as root:
            local_file = Path(root) / 'RN50.pt'
            local_file.write_bytes(b'corrupted pretrained model')
            url = f'https://openaipublic.azureedge.net/clip/models/{expected_hash}/RN50.pt'
            download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_mlfoundations(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()[:8]
        urllib.request.urlopen.return_value = self.create_response(file_contents)
        with tempfile.TemporaryDirectory() as root:
            url = f'https://github.com/mlfoundations/download/v0.2-weights/rn50-quickgelu-{expected_hash}.pt'
            download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_url_from_mlfoundations_corrupted(self, urllib):
        file_contents = b'pretrained model weights'
        expected_hash = hashlib.sha256(file_contents).hexdigest()[:8]
        urllib.request.urlopen.return_value = self.create_response(b'corrupted pretrained model')
        with tempfile.TemporaryDirectory() as root:
            url = f'https://github.com/mlfoundations/download/v0.2-weights/rn50-quickgelu-{expected_hash}.pt'
            with self.assertRaisesRegex(RuntimeError, r'checksum does not not match'):
                download_pretrained_from_url(url, root)
        urllib.request.urlopen.assert_called_once()

    @patch('open_clip.pretrained.urllib')
    def test_download_pretrained_from_hfh(self, urllib):
        model, _, preprocess = open_clip.create_model_and_transforms('hf-hub:hf-internal-testing/tiny-open-clip-model')
        tokenizer = open_clip.get_tokenizer('hf-hub:hf-internal-testing/tiny-open-clip-model')
        img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
        image = preprocess(Image.open(requests.get(img_url, stream=True).raw)).unsqueeze(0)
        text = tokenizer(["a diagram", "a dog", "a cat"])

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        self.assertTrue(torch.allclose(text_probs, torch.tensor([[0.0597, 0.6349, 0.3053]]), 1e-3))

# ===== File: tests/test_hf_model.py =====

import pytest

import torch
from open_clip.hf_model import _POOLERS, HFTextEncoder
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

# test poolers
def test_poolers():
    bs, sl, d = 2, 10, 5
    h = torch.arange(sl).repeat(bs).reshape(bs, sl)[..., None] * torch.linspace(0.2, 1., d)
    mask = torch.ones(bs, sl, dtype=torch.bool)
    mask[:2, 6:] = False
    x = BaseModelOutput(h)
    for name, cls in _POOLERS.items():
        pooler = cls()
        res = pooler(x, mask)
        assert res.shape == (bs, d), f"{name} returned wrong shape"

# test HFTextEncoder
@pytest.mark.parametrize("model_id", ["arampacha/roberta-tiny", "roberta-base", "xlm-roberta-base", "google/mt5-base"])
def test_pretrained_text_encoder(model_id):
    bs, sl, d = 2, 10, 64
    cfg = AutoConfig.from_pretrained(model_id)
    model = HFTextEncoder(model_id, d, proj_type='linear')
    x = torch.randint(0, cfg.vocab_size, (bs, sl))
    with torch.no_grad():
        emb = model(x)

    assert emb.shape == (bs, d)

# ===== File: tests/test_inference_simple.py =====

import torch
from PIL import Image
from open_clip.factory import get_tokenizer
import pytest
import open_clip
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if hasattr(torch._C, '_jit_set_profiling_executor'):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)


test_simple_models = [
    # model, pretrained, jit, force_custom_text
    ("ViT-B-32", "laion2b_s34b_b79k", False, False),
    ("ViT-B-32", "laion2b_s34b_b79k", True, False),
    ("ViT-B-32", "laion2b_s34b_b79k", True, True),
    ("roberta-ViT-B-32", "laion2b_s12b_b32k", False, False),
]


@pytest.mark.parametrize("model_type,pretrained,jit,force_custom_text", test_simple_models)
def test_inference_simple(
        model_type,
        pretrained,
        jit,
        force_custom_text,
):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_type,
        pretrained=pretrained,
        jit=jit,
        force_custom_text=force_custom_text,
    )
    tokenizer = get_tokenizer(model_type)

    current_dir = os.path.dirname(os.path.realpath(__file__))

    image = preprocess(Image.open(current_dir + "/../docs/CLIP.png")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    assert torch.allclose(text_probs.cpu()[0], torch.tensor([1.0, 0.0, 0.0]))

# ===== File: tests/test_inference.py =====


import os
import pytest
import torch
import open_clip
import util_test

os.environ['CUDA_VISIBLE_DEVICES'] = ''

if hasattr(torch._C, '_jit_set_profiling_executor'):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)

models_to_test = set(open_clip.list_models())

# testing excemptions
models_to_test = models_to_test.difference({
        # not available with timm yet
        # see https://github.com/mlfoundations/open_clip/issues/219
        'convnext_xlarge',
        'convnext_xxlarge',
        'convnext_xxlarge_320',
        'vit_medium_patch16_gap_256',
        # exceeds GH runner memory limit
        'ViT-bigG-14',
        'ViT-e-14',
        'mt5-xl-ViT-H-14',
        'coca_base',
        'coca_ViT-B-32',
        'coca_roberta-ViT-B-32'
})

if 'OPEN_CLIP_TEST_REG_MODELS' in os.environ:
    external_model_list = os.environ['OPEN_CLIP_TEST_REG_MODELS']
    with open(external_model_list, 'r') as f:
        models_to_test = set(f.read().splitlines()).intersection(models_to_test)
    print(f"Selected models from {external_model_list}: {models_to_test}")

# TODO: add "coca_ViT-B-32" onece https://github.com/pytorch/pytorch/issues/92073 gets fixed
models_to_test = list(models_to_test)
models_to_test.sort()
models_to_test = [(model_name, False) for model_name in models_to_test]

models_to_jit_test = {"ViT-B-32"}
models_to_jit_test = list(models_to_jit_test)
models_to_jit_test = [(model_name, True) for model_name in models_to_jit_test]
models_to_test_fully = models_to_test + models_to_jit_test


@pytest.mark.regression_test
@pytest.mark.parametrize("model_name,jit", models_to_test_fully)
def test_inference_with_data(
        model_name,
        jit,
        pretrained = None,
        pretrained_hf = False,
        precision = 'fp32',
        force_quick_gelu = False,
):
    util_test.seed_all()
    model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained,
            precision = precision,
            jit = jit,
            force_quick_gelu = force_quick_gelu,
            pretrained_hf = pretrained_hf
    )
    model_id = f'{model_name}_{pretrained or pretrained_hf}_{precision}'
    input_dir, output_dir = util_test.get_data_dirs()
    # text
    input_text_path = os.path.join(input_dir, 'random_text.pt')
    gt_text_path = os.path.join(output_dir, f'{model_id}_random_text.pt')
    if not os.path.isfile(input_text_path):
        pytest.skip(reason = f"missing test data, expected at {input_text_path}")
    if not os.path.isfile(gt_text_path):
        pytest.skip(reason = f"missing test data, expected at {gt_text_path}")
    input_text = torch.load(input_text_path)
    gt_text = torch.load(gt_text_path)
    y_text = util_test.inference_text(model, model_name, input_text)
    assert (y_text == gt_text).all(), f"text output differs @ {input_text_path}"
    # image
    image_size = model.visual.image_size
    if not isinstance(image_size, tuple):
        image_size = (image_size, image_size)
    input_image_path = os.path.join(input_dir, f'random_image_{image_size[0]}_{image_size[1]}.pt')
    gt_image_path = os.path.join(output_dir, f'{model_id}_random_image.pt')
    if not os.path.isfile(input_image_path):
        pytest.skip(reason = f"missing test data, expected at {input_image_path}")
    if not os.path.isfile(gt_image_path):
        pytest.skip(reason = f"missing test data, expected at {gt_image_path}")
    input_image = torch.load(input_image_path)
    gt_image = torch.load(gt_image_path)
    y_image = util_test.inference_image(model, preprocess_val, input_image)
    assert (y_image == gt_image).all(), f"image output differs @ {input_image_path}"
    
    if not jit:
        model.eval()
        model_out = util_test.forward_model(model, model_name, preprocess_val, input_image, input_text)
        if type(model) not in [open_clip.CLIP, open_clip.CustomTextCLIP]:
            assert type(model_out) == dict
        else:
            model.output_dict = True
            model_out_dict = util_test.forward_model(model, model_name, preprocess_val, input_image, input_text)
            assert (model_out_dict["image_features"] == model_out[0]).all()
            assert (model_out_dict["text_features"] == model_out[1]).all()
            assert (model_out_dict["logit_scale"] == model_out[2]).all()
            model.output_dict = None
    else:
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained,
            precision = precision,
            jit = False,
            force_quick_gelu = force_quick_gelu,
            pretrained_hf = pretrained_hf
        )
        
        test_model = util_test.TestWrapper(model, model_name, output_dict=False)
        test_model = torch.jit.script(test_model)
        model_out = util_test.forward_model(test_model, model_name, preprocess_val, input_image, input_text)
        assert model_out["test_output"].shape[-1] == 2

        test_model = util_test.TestWrapper(model, model_name, output_dict=True)
        test_model = torch.jit.script(test_model)
        model_out = util_test.forward_model(test_model, model_name, preprocess_val, input_image, input_text)
        assert model_out["test_output"].shape[-1] == 2
        
    



# ===== File: tests/test_num_shards.py =====

import pytest

from open_clip_train.data import get_dataset_size

@pytest.mark.parametrize(
    "shards,expected_size",
    [
        ('/path/to/shard.tar', 1),
        ('/path/to/shard_{000..000}.tar', 1),
        ('/path/to/shard_{000..009}.tar', 10),
        ('/path/to/shard_{000..009}_{000..009}.tar', 100),
        ('/path/to/shard.tar::/path/to/other_shard_{000..009}.tar', 11),
        ('/path/to/shard_{000..009}.tar::/path/to/other_shard_{000..009}.tar', 20),
        (['/path/to/shard.tar'], 1),
        (['/path/to/shard.tar', '/path/to/other_shard.tar'], 2),
    ]
)
def test_num_shards(shards, expected_size):
    _, size = get_dataset_size(shards)
    assert size == expected_size, f'Expected {expected_size} for {shards} but found {size} instead.'

# ===== File: tests/test_training_simple.py =====


import os
import sys
import pytest
import torch
from open_clip_train.main import main

os.environ["CUDA_VISIBLE_DEVICES"] = ""

if hasattr(torch._C, '_jit_set_profiling_executor'):
    # legacy executor is too slow to compile large models for unit tests
    # no need for the fusion performance here
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(False)

@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training():
    main([
    '--save-frequency', '1',
    '--zeroshot-frequency', '1',
    '--dataset-type', "synthetic",
    '--train-num-samples', '16',
    '--warmup', '1',
    '--batch-size', '4',
    '--lr', '1e-3',
    '--wd', '0.1',
    '--epochs', '1',
    '--workers', '2',
    '--model', 'RN50'
    ])

@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_coca():
    main([
    '--save-frequency', '1',
    '--zeroshot-frequency', '1',
    '--dataset-type', "synthetic",
    '--train-num-samples', '16',
    '--warmup', '1',
    '--batch-size', '4',
    '--lr', '1e-3',
    '--wd', '0.1',
    '--epochs', '1',
    '--workers', '2',
    '--model', 'coca_ViT-B-32'
    ])

@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_mt5():
    main([
    '--save-frequency', '1',
    '--zeroshot-frequency', '1',
    '--dataset-type', "synthetic",
    '--train-num-samples', '16',
    '--warmup', '1',
    '--batch-size', '4',
    '--lr', '1e-3',
    '--wd', '0.1',
    '--epochs', '1',
    '--workers', '2',
    '--model', 'mt5-base-ViT-B-32',
    '--lock-text',
    '--lock-text-unlocked-layers', '2'
    ])



@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_unfreezing_vit():
    main([
    '--save-frequency', '1',
    '--zeroshot-frequency', '1',
    '--dataset-type', "synthetic",
    '--train-num-samples', '16',
    '--warmup', '1',
    '--batch-size', '4',
    '--lr', '1e-3',
    '--wd', '0.1',
    '--epochs', '1',
    '--workers', '2',
    '--model', 'ViT-B-32',
    '--lock-image',
    '--lock-image-unlocked-groups', '5',
    '--accum-freq', '2'
    ])


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason="macos pickle bug with locals")
def test_training_clip_with_jit():
    main([
    '--save-frequency', '1',
    '--zeroshot-frequency', '1',
    '--dataset-type', "synthetic",
    '--train-num-samples', '16',
    '--warmup', '1',
    '--batch-size', '4',
    '--lr', '1e-3',
    '--wd', '0.1',
    '--epochs', '1',
    '--workers', '2',
    '--model', 'ViT-B-32',
    '--torchscript'
    ])

# ===== File: tests/test_wds.py =====

import os
import pytest
import util_test
import collections
import tarfile
import io
from PIL import Image

from open_clip_train.data import get_wds_dataset
from open_clip_train.params import parse_args
from open_clip_train.main import random_seed

TRAIN_NUM_SAMPLES = 10_000
RTOL = 0.2

# NOTE: we use two test tar files, which are created on the fly and saved to data/input.
# 000.tar has 10 samples, and the captions are 000_0, 000_1, ..., 000_9
# 001.tar has 5 samples, and the captions are 001_0, 001_1, ..., 001_4
def build_inputs(test_name):
    base_input_dir, _ = util_test.get_data_dirs()
    input_dir = os.path.join(base_input_dir, test_name)
    os.makedirs(input_dir, exist_ok=True)
    
    def save_tar(idx, num_samples):
        filename = os.path.join(input_dir, f'test_data_{idx:03d}.tar')
        tar = tarfile.open(filename, 'w')
        
        for sample_idx in range(num_samples):
            # Image
            image = Image.new('RGB', (32, 32))
            info = tarfile.TarInfo(f'{sample_idx}.png')
            bio = io.BytesIO()
            image.save(bio, format='png')
            size = bio.tell()
            bio.seek(0)
            info.size = size
            tar.addfile(info, bio)
            
            # Caption
            info = tarfile.TarInfo(f'{sample_idx}.txt')
            bio = io.BytesIO()
            bio.write(f'{idx:03d}_{sample_idx}'.encode('utf-8'))
            size = bio.tell()
            bio.seek(0)
            info.size = size
            tar.addfile(info, bio)
        
        tar.close()          

    save_tar(0, 10)
    save_tar(1, 5)

    return input_dir


def build_params(input_shards, seed=0):
    args = parse_args([])
    args.train_data = input_shards
    args.train_num_samples = TRAIN_NUM_SAMPLES
    args.dataset_resampled = True
    args.seed = seed
    args.workers = 1
    args.world_size = 1
    args.batch_size = 1
    random_seed(seed)

    preprocess_img = lambda x: x
    tokenizer = lambda x: [x.strip()]

    return args, preprocess_img, tokenizer


def get_dataloader(input_shards):
    args, preprocess_img, tokenizer = build_params(input_shards)
    dataset = get_wds_dataset(args, preprocess_img, is_train=True, tokenizer=tokenizer)
    dataloader = dataset.dataloader
    return dataloader


def test_single_source():
    """Test webdataset with a single tar file."""
    input_dir = build_inputs('single_source')    
    input_shards = os.path.join(input_dir, 'test_data_000.tar')
    dataloader = get_dataloader(input_shards)
    
    counts = collections.defaultdict(int)
    for sample in dataloader:
        txts = sample[1]
        for txt in txts:
            counts[txt] += 1
    
    for key, count in counts.items():
        assert count == pytest.approx(TRAIN_NUM_SAMPLES / 10, RTOL)


def test_two_sources():
    """Test webdataset with a single two tar files."""
    input_dir = build_inputs('two_sources')
    input_shards = os.path.join(input_dir, 'test_data_{000..001}.tar')
    dataloader = get_dataloader(input_shards)

    counts = collections.defaultdict(int)
    for sample in dataloader:
        txts = sample[1]
        for txt in txts:
            counts[txt] += 1
    
    for key, count in counts.items():
        assert count == pytest.approx(TRAIN_NUM_SAMPLES / 15, RTOL), f'{key}, {count}'


def test_two_sources_same_weights():
    """Test webdataset with a two tar files, using --train-data-weights=1::1."""
    input_dir = build_inputs('two_sources_same_weights')
    input_shards = f"{os.path.join(input_dir, 'test_data_000.tar')}::{os.path.join(input_dir, 'test_data_001.tar')}"
    args, preprocess_img, tokenizer = build_params(input_shards)
    args.train_data_upsampling_factors = '1::1'
    dataset = get_wds_dataset(args, preprocess_img, is_train=True, tokenizer=tokenizer)
    dataloader = dataset.dataloader

    counts = collections.defaultdict(int)
    for sample in dataloader:
        txts = sample[1]
        for txt in txts:
            counts[txt] += 1
    
    for key, count in counts.items():
        assert count == pytest.approx(TRAIN_NUM_SAMPLES / 15, RTOL), f'{key}, {count}'

def test_two_sources_with_upsampling():
    """Test webdataset with a two tar files with upsampling."""
    input_dir = build_inputs('two_sources_with_upsampling')
    input_shards = f"{os.path.join(input_dir, 'test_data_000.tar')}::{os.path.join(input_dir, 'test_data_001.tar')}"
    args, preprocess_img, tokenizer = build_params(input_shards)
    args.train_data_upsampling_factors = '1::2'
    dataset = get_wds_dataset(args, preprocess_img, is_train=True, tokenizer=tokenizer)
    dataloader = dataset.dataloader

    counts = collections.defaultdict(int)
    for sample in dataloader:
        txts = sample[1]
        for txt in txts:
            counts[txt] += 1
    
    for key, count in counts.items():
        if key.startswith('000'):
            assert count == pytest.approx(TRAIN_NUM_SAMPLES / 20, RTOL), f'{key}, {count}'
        else:
            assert count == pytest.approx(TRAIN_NUM_SAMPLES / 10, RTOL), f'{key}, {count}'

# ===== File: tests/util_test.py =====

import os
import random
import numpy as np
from PIL import Image
import torch

if __name__ != '__main__':
    import open_clip

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def seed_all(seed = 0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def inference_text(model, model_name, batches):
    y = []
    tokenizer = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        for x in batches:
            x = tokenizer(x)
            y.append(model.encode_text(x))
        return torch.stack(y)

def inference_image(model, preprocess_val, batches):
    y = []
    with torch.no_grad():
        for x in batches:
            x = torch.stack([preprocess_val(img) for img in x])
            y.append(model.encode_image(x))
        return torch.stack(y)
    
def forward_model(model, model_name, preprocess_val, image_batch, text_batch):
    y = []
    tokenizer = open_clip.get_tokenizer(model_name)
    with torch.no_grad():
        for x_im, x_txt in zip(image_batch, text_batch):
            x_im = torch.stack([preprocess_val(im) for im in x_im])
            x_txt = tokenizer(x_txt)
        y.append(model(x_im, x_txt))
    if type(y[0]) == dict:
        out = {}
        for key in y[0].keys():
            out[key] = torch.stack([batch_out[key] for batch_out in y])
    else:
        out = []
        for i in range(len(y[0])):
            out.append(torch.stack([batch_out[i] for batch_out in y]))
    return out

def random_image_batch(batch_size, size):
    h, w = size
    data = np.random.randint(255, size = (batch_size, h, w, 3), dtype = np.uint8)
    return [ Image.fromarray(d) for d in data ]

def random_text_batch(batch_size, min_length = 75, max_length = 75):
    t = open_clip.tokenizer.SimpleTokenizer()
    # every token decoded as string, exclude SOT and EOT, replace EOW with space
    token_words = [
            x[1].replace('</w>', ' ')
            for x in t.decoder.items()
            if x[0] not in t.all_special_ids
    ]
    # strings of randomly chosen tokens
    return [
        ''.join(random.choices(
                token_words,
                k = random.randint(min_length, max_length)
        ))
        for _ in range(batch_size)
    ]

def create_random_text_data(
        path,
        min_length = 75,
        max_length = 75,
        batches = 1,
        batch_size = 1
):
    text_batches = [
            random_text_batch(batch_size, min_length, max_length)
            for _ in range(batches)
    ]
    print(f"{path}")
    torch.save(text_batches, path)

def create_random_image_data(path, size, batches = 1, batch_size = 1):
    image_batches = [
            random_image_batch(batch_size, size)
            for _ in range(batches)
    ]
    print(f"{path}")
    torch.save(image_batches, path)

def get_data_dirs(make_dir = True):
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    input_dir = os.path.join(data_dir, 'input')
    output_dir = os.path.join(data_dir, 'output')
    if make_dir:
        os.makedirs(input_dir, exist_ok = True)
        os.makedirs(output_dir, exist_ok = True)
    assert os.path.isdir(data_dir), f"data directory missing, expected at {input_dir}"
    assert os.path.isdir(data_dir), f"data directory missing, expected at {output_dir}"
    return input_dir, output_dir

def create_test_data_for_model(
        model_name,
        pretrained = None,
        precision = 'fp32',
        jit = False,
        pretrained_hf = False,
        force_quick_gelu = False,
        create_missing_input_data = True,
        batches = 1,
        batch_size = 1,
        overwrite = False
):
    model_id = f'{model_name}_{pretrained or pretrained_hf}_{precision}'
    input_dir, output_dir = get_data_dirs()
    output_file_text = os.path.join(output_dir, f'{model_id}_random_text.pt')
    output_file_image = os.path.join(output_dir, f'{model_id}_random_image.pt')
    text_exists = os.path.exists(output_file_text)
    image_exists = os.path.exists(output_file_image)
    if not overwrite and text_exists and image_exists:
        return
    seed_all()
    model, _, preprocess_val = open_clip.create_model_and_transforms(
            model_name,
            pretrained = pretrained,
            precision = precision,
            jit = jit,
            force_quick_gelu = force_quick_gelu,
            pretrained_hf = pretrained_hf
    )
    # text
    if overwrite or not text_exists:
        input_file_text = os.path.join(input_dir, 'random_text.pt')
        if create_missing_input_data and not os.path.exists(input_file_text):
            create_random_text_data(
                    input_file_text,
                    batches = batches,
                    batch_size = batch_size
            )
        assert os.path.isfile(input_file_text), f"missing input data, expected at {input_file_text}"
        input_data_text = torch.load(input_file_text)
        output_data_text = inference_text(model, model_name, input_data_text)
        print(f"{output_file_text}")
        torch.save(output_data_text, output_file_text)
    # image
    if overwrite or not image_exists:
        size = model.visual.image_size
        if not isinstance(size, tuple):
            size = (size, size)
        input_file_image = os.path.join(input_dir, f'random_image_{size[0]}_{size[1]}.pt')
        if create_missing_input_data and not os.path.exists(input_file_image):
            create_random_image_data(
                    input_file_image,
                    size,
                    batches = batches,
                    batch_size = batch_size
            )
        assert os.path.isfile(input_file_image), f"missing input data, expected at {input_file_image}"
        input_data_image = torch.load(input_file_image)
        output_data_image = inference_image(model, preprocess_val, input_data_image)
        print(f"{output_file_image}")
        torch.save(output_data_image, output_file_image)

def create_test_data(
        models,
        batches = 1,
        batch_size = 1,
        overwrite = False
):
    models = list(set(models).difference({
            # not available with timm
            # see https://github.com/mlfoundations/open_clip/issues/219
            'timm-convnext_xlarge',
            'timm-vit_medium_patch16_gap_256'
    }).intersection(open_clip.list_models()))
    models.sort()
    print(f"generating test data for:\n{models}")
    for model_name in models:
        print(model_name)
        create_test_data_for_model(
                model_name,
                batches = batches,
                batch_size = batch_size,
                overwrite = overwrite
        )
    return models

def _sytem_assert(string):
    assert os.system(string) == 0

class TestWrapper(torch.nn.Module):
    output_dict: torch.jit.Final[bool]
    def __init__(self, model, model_name, output_dict=True) -> None:
        super().__init__()
        self.model = model
        self.output_dict = output_dict
        if type(model) in [open_clip.CLIP, open_clip.CustomTextCLIP]:
            self.model.output_dict = self.output_dict
        config = open_clip.get_model_config(model_name)
        self.head = torch.nn.Linear(config["embed_dim"], 2)

    def forward(self, image, text):
        x = self.model(image, text)
        x = x['image_features'] if self.output_dict else x[0]
        assert x is not None  # remove Optional[], type refinement for torchscript
        out = self.head(x)
        return {"test_output": out}

def main(args):
    global open_clip
    import importlib
    import shutil
    import subprocess
    import argparse
    parser = argparse.ArgumentParser(description = "Populate test data directory")
    parser.add_argument(
        '-a', '--all',
        action = 'store_true',
        help = "create test data for all models"
    )
    parser.add_argument(
        '-m', '--model',
        type = str,
        default = [],
        nargs = '+',
        help = "model(s) to create test data for"
    )
    parser.add_argument(
        '-f', '--model_list',
        type = str,
        help = "path to a text file containing a list of model names, one model per line"
    )
    parser.add_argument(
        '-s', '--save_model_list',
        type = str,
        help = "path to save the list of models that data was generated for"
    )
    parser.add_argument(
        '-g', '--git_revision',
        type = str,
        help = "git revision to generate test data for"
    )
    parser.add_argument(
        '--overwrite',
        action = 'store_true',
        help = "overwrite existing output data"
    )
    parser.add_argument(
        '-n', '--num_batches',
        default = 1,
        type = int,
        help = "amount of data batches to create (default: 1)"
    )
    parser.add_argument(
        '-b', '--batch_size',
        default = 1,
        type = int,
        help = "test data batch size (default: 1)"
    )
    args = parser.parse_args(args)
    model_list = []
    if args.model_list is not None:
        with open(args.model_list, 'r') as f:
            model_list = f.read().splitlines()
    if not args.all and len(args.model) < 1 and len(model_list) < 1:
        print("error: at least one model name is required")
        parser.print_help()
        parser.exit(1)
    if args.git_revision is not None:
        stash_output = subprocess.check_output(['git', 'stash']).decode().splitlines()
        has_stash = len(stash_output) > 0 and stash_output[0] != 'No local changes to save'
        current_branch = subprocess.check_output(['git', 'branch', '--show-current'])
        if len(current_branch) < 1:
            # not on a branch -> detached head
            current_branch = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        current_branch = current_branch.splitlines()[0].decode()
        try:
            _sytem_assert(f'git checkout {args.git_revision}')
        except AssertionError as e:
            _sytem_assert(f'git checkout -f {current_branch}')
            if has_stash:
                os.system(f'git stash pop')
            raise e
    open_clip = importlib.import_module('open_clip')
    models = open_clip.list_models() if args.all else args.model + model_list
    try:
        models = create_test_data(
            models,
            batches = args.num_batches,
            batch_size = args.batch_size,
            overwrite = args.overwrite
        )
    finally:
        if args.git_revision is not None:
            test_dir = os.path.join(os.path.dirname(__file__), 'data')
            test_dir_ref = os.path.join(os.path.dirname(__file__), 'data_ref')
            if os.path.exists(test_dir_ref):
                shutil.rmtree(test_dir_ref, ignore_errors = True)
            if os.path.exists(test_dir):
                os.rename(test_dir, test_dir_ref)
            _sytem_assert(f'git checkout {current_branch}')
            if has_stash:
                os.system(f'git stash pop')
            os.rename(test_dir_ref, test_dir)
    if args.save_model_list is not None:
        print(f"Saving model list as {args.save_model_list}")
        with open(args.save_model_list, 'w') as f:
            for m in models:
                print(m, file=f)


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

