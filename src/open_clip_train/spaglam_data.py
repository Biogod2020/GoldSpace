# 文件路径: src/open_clip_train/spaglam_data.py

import logging
import scanpy as sc
import torch
import torch.utils.data
import webdataset as wds
from PIL import Image
import io
from torch_geometric.data import Data as PyGData, Batch as PyGBatch

# 从 open-clip 的现有模块导入辅助类，以保持一致性
from .data import DataInfo

# --- SpaGLaM 专属数据集类 ---

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
        self.adata = None # AnnData将在每个工作进程中被独立加载

    def _init_worker(self):
        """
        每个Dataloader工作进程的初始化函数。
        这里加载AnnData，保证每个进程有自己的文件句柄，避免多进程冲突。
        """
        if self.adata is None:
            # 使用只读模式加载，对多进程更安全、更高效
            self.adata = sc.read_h5ad(self.anndata_path, backed='r')

    def _process_sample(self, sample):
        """
        这是核心的map函数，用于处理从WebDataset流出的单个中心点样本。
        """
        self._init_worker() # 保证AnnData已加载
        center_key = sample['__key__']
        
        try:
            # 1. 解码中心点数据 (现在更加稳健)
            center_img = sample['png'] # wds.decode("pilrgb") 保证了这是一个PIL Image
            
            # 防御性编程：检查类型，因为webdataset可能已经自动解码了.txt文件
            txt_data = sample['txt']
            if isinstance(txt_data, bytes):
                center_sent = txt_data.decode('utf-8')
            else:
                center_sent = str(txt_data) # 确保它一定是字符串

            # 2. 从 AnnData 获取邻居信息
            center_idx = self.adata.obs_names.get_loc(center_key)
            neighbor_indices = self.adata.obsp['spatial_connectivities'][center_idx].indices
            neighbor_keys = self.adata.obs_names[neighbor_indices].tolist()
            
            all_keys = [center_key] + neighbor_keys
            num_nodes = len(all_keys)
            
            paths_df = self.adata.obs.loc[all_keys, ['image_path', 'sentence_path']]
            
            # 3. 加载所有节点的数据并预处理
            images_processed, sentences = [], []
            for key in all_keys:
                if key == center_key:
                    img = center_img
                    sent = center_sent
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
                
                images_processed.append(self.image_processor(img))
                sentences.append(sent)

            tokenized_texts = self.tokenizer(sentences)
            
            # 4. 构建局部图结构
            key_to_local_idx = {key: i for i, key in enumerate(all_keys)}
            edge_list = []
            
            for i, key in enumerate(all_keys):
                global_idx = self.adata.obs_names.get_loc(key)
                row_indices = self.adata.obsp['spatial_connectivities'][global_idx].indices
                
                for neighbor_global_idx in row_indices:
                    neighbor_key = self.adata.obs_names[neighbor_global_idx]
                    if neighbor_key in key_to_local_idx:
                        j = key_to_local_idx[neighbor_key]
                        if i < j:
                            edge_list.append([i, j])
                            edge_list.append([j, i])

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
            
            # 5. 返回PyG Data对象
            return PyGData(
                x_image=torch.stack(images_processed),
                x_text=tokenized_texts,
                edge_index=edge_index,
                num_nodes=num_nodes,
                center_key=center_key,
            )
        except Exception as e:
            logging.error(f"Error processing sample {center_key}: {e}", exc_info=True)
            return None

    def __iter__(self):
        # WebDataset数据处理流水线
        pipeline = wds.DataPipeline(
            wds.shardlists.ResampledShards(self.tar_urls), # 使用ResampledShards以支持大规模分布式训练
            wds.split_by_worker,
            wds.tarfile_to_samples(),
            wds.decode("pilrgb"), # 只解码图像
            wds.map(self._process_sample),
            wds.select(lambda x: x is not None),
        )
        return iter(pipeline)


# --- SpaGLaM 专属数据获取器/工厂函数 ---

def get_spaglam_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    这是SpaGLaM数据集的专属获取器/工厂函数。
    它负责实例化数据集和DataLoader。
    """
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None, "训练或验证的WebDataset路径必须被指定 (--train-data/--val-data)"
    assert args.anndata_path is not None, "SpaGLaM数据集需要一个AnnData文件路径 (--anndata-path)"

    dataset = SpaGLaMWDataset(
        tar_urls=input_shards,
        anndata_path=args.anndata_path,
        image_processor=preprocess_fn,
        tokenizer=tokenizer,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=is_train,
        collate_fn=PyGBatch.from_data_list, # 使用torch_geometric的标准批处理函数
    )

    # 附加元数据以便于训练循环
    num_samples = args.train_num_samples if is_train else args.val_num_samples
    dataloader.num_samples = num_samples
    if num_samples and args.world_size > 0:
        dataloader.num_batches = num_samples // (args.batch_size * args.world_size)
    else:
        dataloader.num_batches = 0 

    return DataInfo(dataloader=dataloader)