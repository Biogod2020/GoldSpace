# src/open_clip_train/spaglam_data.py

import os
import scanpy as sc
import webdataset as wds
import torch
import torch.utils.data
from torch_geometric.data import Data as PyGData, Batch as PyGBatch
from PIL import Image
import io
import logging

from .data import DataInfo, SharedEpoch, detshuffle2, _SHARD_SHUFFLE_SIZE, _SHARD_SHUFFLE_INITIAL, _SAMPLE_SHUFFLE_SIZE, _SAMPLE_SHUFFLE_INITIAL

class SpaGLaMWDataset(torch.utils.data.IterableDataset):
    """
    生产级的SpaGLaM数据集。它流式处理WebDataset分片，并使用一个
    中央AnnData文件来动态获取每个中心点的邻居信息和图结构。
    """
    def __init__(self, tar_urls, anndata_path, image_processor, tokenizer):
        super().__init__()
        self.tar_urls = tar_urls
        self.anndata_path = anndata_path
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.adata = None

    def _init_worker(self):
        if self.adata is None:
            # 使用只读模式加载，对多进程更安全
            self.adata = sc.read_h5ad(self.anndata_path, backed='r')

    def _process_sample(self, sample):
        self._init_worker()
        center_key = sample['__key__']
        
        try:
            center_idx = self.adata.obs_names.get_loc(center_key)
            neighbor_indices = self.adata.obsp['spatial_connectivities'][center_idx].indices
            neighbor_keys = self.adata.obs_names[neighbor_indices].tolist()
            
            all_keys = [center_key] + neighbor_keys
            num_nodes = len(all_keys)

            paths_df = self.adata.obs.loc[all_keys, ['image_path', 'sentence_path']]
            
            images, sentences = [], []
            for key in all_keys:
                if key == center_key:
                    img = Image.open(io.BytesIO(sample['png'])).convert("RGB")
                    sent = sample['txt'].decode('utf-8')
                else:
                    img_path, sent_path = paths_df.loc[key]
                    try:
                        img = Image.open(img_path).convert("RGB")
                    except (FileNotFoundError, OSError):
                        img = Image.new('RGB', (224, 224), color='grey')
                    try:
                        with open(sent_path, 'r') as f: sent = f.read()
                    except (FileNotFoundError, OSError):
                        sent = ""
                images.append(self.image_processor(img))
                sentences.append(sent)

            tokenized_texts = self.tokenizer(sentences)
            
            key_to_local_idx = {key: i for i, key in enumerate(all_keys)}
            edge_list = []
            center_local_idx = 0 # 中心点始终是第一个
            
            # 从稀疏矩阵构建局部邻接关系
            # 注意：这里的图构建逻辑假设邻接图是对称的，只连接中心点和其直接邻居
            for neighbor_key in neighbor_keys:
                neighbor_local_idx = key_to_local_idx.get(neighbor_key)
                if neighbor_local_idx is not None:
                    edge_list.append([center_local_idx, neighbor_local_idx])
                    edge_list.append([neighbor_local_idx, center_local_idx])

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)

            # 返回一个PyG Data对象，这是下游模型期望的格式
            return PyGData(
                x_image=torch.stack(images),
                x_text=tokenized_texts,
                edge_index=edge_index,
                num_nodes=num_nodes,
                center_key=center_key,
            )
        except Exception as e:
            logging.warning(f"Skipping sample {center_key} due to error: {e}")
            return None

    def __iter__(self):
        pipeline = wds.DataPipeline(
            wds.ResampledShards(self.tar_urls),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode("pilrgb", "txt"),
            wds.map(self._process_sample),
            wds.select(lambda x: x is not None),
        )
        return iter(pipeline)

def get_spaglam_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    这是SpaGLaM数据集的专属获取器/工厂函数。
    """
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None, "训练或验证的WebDataset路径必须被指定"
    assert args.anndata_path is not None, "SpaGLaM数据集需要一个AnnData文件路径"

    resampled = getattr(args, 'dataset_resampled', False) and is_train

    # SpaGLaM Dataset的实例化
    dataset = SpaGLaMWDataset(
        tar_urls=input_shards,
        anndata_path=args.anndata_path,
        image_processor=preprocess_fn,
        tokenizer=tokenizer,
    )
    
    # 注意：对于IterableDataset，我们不使用标准的Sampler。
    # WebDataset的wds.split_by_worker已经处理了分布式数据划分。
    
    # 使用torch_geometric的collate_fn来自动批处理图数据
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=PyGBatch.from_data_list, # 关键！
    )

    # 附加元数据以便于训练循环
    # 对于IterableDataset，num_samples和num_batches需要估算或从args中获取
    dataloader.num_samples = args.train_num_samples if is_train else args.val_num_samples
    if dataloader.num_samples:
        dataloader.num_batches = dataloader.num_samples // (args.batch_size * args.world_size)
    else:
        dataloader.num_batches = 0 # 如果不确定，可以设为0或在训练中动态计算

    return DataInfo(dataloader=dataloader)