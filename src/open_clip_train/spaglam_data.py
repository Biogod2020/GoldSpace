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
            
        # ===== SOTA MODIFICATION START =====
        # Load all embeddings from a single .pth file with one torch.load call.
        # This avoids the massive I/O overhead of loading each node's embedding separately.
        
        embeddings_dict = torch.load(io.BytesIO(sample["embeddings.pth"]))
        image_embeddings = embeddings_dict['image']
        text_embeddings = embeddings_dict['text']
        # ===== SOTA MODIFICATION END =====

        # 4. Create the final PyG Data object
        pyg_data = PyGData(
            x_image=image_embeddings,
            x_text=text_embeddings,
            edge_index=edge_index,
            num_nodes=num_nodes,
        )
        return pyg_data

    except Exception as e:
        logging.warning(f"Skipping corrupted embedding sample {sample.get('__key__', 'UNKNOWN_KEY')}. Error: {e}")
        return None