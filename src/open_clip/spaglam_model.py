# 文件路径: src/open_clip/spaglam_model.py (版本3 - SOTA重构)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 导入 torch_geometric 的核心组件
from torch_geometric.nn import GATv2Conv, TransformerConv
from torch_geometric.nn.glob import global_mean_pool

# ==============================================================================
# ========================      SOTA 组件      ================================
# ==============================================================================

# --- 组件 1: 你的自定义图Transformer层 (保持不变) ---
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
        # 创建注意力掩码，确保注意力只在子图内部计算
        attn_mask = torch.eq(batch_index.unsqueeze(1), batch_index.unsqueeze(0)).logical_not()
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x

# --- 组件 2: 残差连接包装器 (新 SOTA 组件) ---
# 这个包装器可以为任何GNN层添加残差连接，非常模块化。
class ResidualGNNWrapper(nn.Module):
    def __init__(self, gnn_layer: nn.Module):
        super().__init__()
        self.gnn_layer = gnn_layer

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 残差连接的核心：输入 + GNN层的输出
        # 使用 *args 和 **kwargs 确保它可以包装任何类型的GNN层
        return x + self.gnn_layer(x, *args, **kwargs)

# --- 组件 3: GNN层工厂 (新 SOTA 组件) ---
# 这个函数根据配置创建正确的GNN层，使主模型代码更清晰。
def create_gnn_layer(config: object, in_dim: int, out_dim: int) -> nn.Module:
    gnn_type = config.gnn_type.lower()
    if gnn_type == 'gat':
        return GATv2Conv(in_dim, out_dim, heads=config.gnn_heads, concat=False, dropout=0.1)
    elif gnn_type == 'graphtransformer':
        return GraphTransformerLayer(in_dim, num_heads=config.gnn_heads)
    elif gnn_type == 'transformerconv':
        # TransformerConv 是一个更强大的、基于真实图拓扑的注意力层
        return TransformerConv(in_dim, out_dim, heads=config.gnn_heads, concat=False, dropout=0.1)
    else:
        raise ValueError(f"Unsupported GNN type: '{config.gnn_type}'")

# --- 组件 4: 非线性投影头 (保持不变) ---
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

# ==============================================================================
# ========================      主模型: SpaGLaM      ===========================
# ==============================================================================
class SpaGLaM(nn.Module):
    def __init__(self, open_clip_model: nn.Module, config: object):
        # ... (__init__ 方法保持完全不变) ...
        super().__init__()
        self.config = config
        self.omiclip_model = open_clip_model
        self.use_precomputed_embeddings = getattr(config, 'use_precomputed_embeddings', False)
        
        if config.freeze_omiclip:
            for param in self.omiclip_model.parameters():
                param.requires_grad = False
            self.omiclip_model.eval()

        if hasattr(self.omiclip_model.visual, 'proj') and self.omiclip_model.visual.proj is not None:
             gnn_output_dim = self.omiclip_model.visual.proj.shape[1]
        else:
             gnn_output_dim = self.omiclip_model.visual.output_dim
        gnn_input_dim = self.omiclip_model.visual.output_dim
        gnn_hidden_dim = config.gnn_hidden_dim
        
        self.gnn_layers_img = nn.ModuleList()
        self.gnn_layers_gene = nn.ModuleList()
        self.interaction_layers = nn.ModuleList() if config.use_deep_fusion else None
        
        current_dim = gnn_input_dim
        for _ in range(config.gnn_layers):
            img_layer = create_gnn_layer(config, current_dim, gnn_hidden_dim)
            gene_layer = create_gnn_layer(config, current_dim, gnn_hidden_dim)
            
            if getattr(config, 'use_residual_connection', False):
                img_layer = ResidualGNNWrapper(img_layer)
                gene_layer = ResidualGNNWrapper(gene_layer)
            
            self.gnn_layers_img.append(img_layer)
            self.gnn_layers_gene.append(gene_layer)
            current_dim = gnn_hidden_dim
        
        img_proj_in = gnn_hidden_dim if getattr(config, "gnn_mode", "dual") in ("dual", "image_only") else gnn_input_dim
        txt_proj_in = gnn_hidden_dim if getattr(config, "gnn_mode", "dual") in ("dual", "text_only") else gnn_input_dim
        self.image_proj_head = MLPProjectionHead(img_proj_in,  gnn_hidden_dim, gnn_output_dim)
        self.gene_proj_head  = MLPProjectionHead(txt_proj_in,  gnn_hidden_dim, gnn_output_dim)
        self.logit_scale = self.omiclip_model.logit_scale


    @staticmethod
    def _select_centers(x_nodes: torch.Tensor, batch) -> torch.Tensor:
        """为每个子图选出一个中心节点的特征，返回 [G, C]。优先使用 batch.center_mask；否则用 ptr 或 batch 边界。"""
        N = x_nodes.size(0)
        device = x_nodes.device
        if hasattr(batch, "center_mask") and batch.center_mask is not None:
            mask = batch.center_mask
            assert mask.dtype == torch.bool and mask.numel() == N, "center_mask 必须是 [N] 的 bool"
            idx = torch.nonzero(mask, as_tuple=False).view(-1)
            return x_nodes[idx]
        if hasattr(batch, "ptr") and batch.ptr is not None:
            starts = batch.ptr[:-1]
            return x_nodes[starts]
        assert hasattr(batch, "batch"), "需要 batch.batch 或 ptr 才能定位子图"
        graph_ids: torch.Tensor = batch.batch
        starts = []
        prev = -1
        for i in range(N):
            g = int(graph_ids[i].item())
            if g != prev:
                starts.append(i); prev = g
        starts = torch.tensor(starts, device=device, dtype=torch.long)
        return x_nodes[starts]

    def forward_gnn(self, batch: "torch_geometric.data.Batch") -> (torch.Tensor, torch.Tensor):
        """
        Runs the GNN part of the model and returns node-level features before pooling.
        这是新的、用于特征提取的函数。
        """
        # 1. 初始特征提取
        if self.training and not self.use_precomputed_embeddings:
            with torch.set_grad_enabled(not self.config.freeze_omiclip):
                E_image = self.omiclip_model.encode_image(batch.x_image)
                E_gene = self.omiclip_model.encode_text(batch.x_text)
        elif self.use_precomputed_embeddings:
            # SOTA 核心修复：安全地获取输入特征
            # E_image 必须存在
            E_image = batch.x_image
            # E_gene 是可选的。如果不存在，创建一个兼容的零张量作为占位符。
            # 这使得模型在只有图像输入的推理场景下也能正常工作。
            E_gene = getattr(batch, 'x_text', torch.zeros_like(E_image))
        else: # Eval mode with raw data
            with torch.no_grad():
                E_image = self.omiclip_model.encode_image(batch.x_image)
                # 同样地，安全处理 x_text
                if hasattr(batch, 'x_text'):
                    E_gene = self.omiclip_model.encode_text(batch.x_text)
                else:
                    E_gene = torch.zeros_like(E_image)

        # 2. GNN传播 (这部分代码保持不变)
        img_feat, gene_feat = E_image, E_gene
        for i in range(self.config.gnn_layers):
            if self.config.gnn_type in ['gat', 'transformerconv']:
                img_feat = self.gnn_layers_img[i](img_feat, batch.edge_index)
                gene_feat = self.gnn_layers_gene[i](gene_feat, batch.edge_index)
            else:
                img_feat = self.gnn_layers_img[i](img_feat, batch.batch)
                gene_feat = self.gnn_layers_gene[i](gene_feat, batch.batch)
            
            img_feat = F.gelu(img_feat)
            gene_feat = F.gelu(gene_feat)

            if self.interaction_layers is not None:
                img_feat, gene_feat = self.interaction_layers[i](img_feat, gene_feat)
        
        return img_feat, gene_feat

    def forward(self, batch: "torch_geometric.data.Batch") -> dict:
        """
        The default forward pass for end-to-end training.
        这个函数接口和功能都保持不变。
        """
        # 1. 获取节点级别的 GNN 特征
        img_feat, gene_feat = self.forward_gnn(batch)

        # 2. 图级别读出 (Readout) 和 投影 (Projection)
        gnn_mode = getattr(self.config, "gnn_mode", "dual")
        text_pooler = getattr(self.config, "text_pooler", "mean")

        if gnn_mode == "image_only" and text_pooler == "center":
            # image：走 GNN + mean 池化
            Z_image = global_mean_pool(img_feat, batch.batch)

            # text：不进 GNN，直接取中心 spot（不池化）
            if self.use_precomputed_embeddings:
                E_txt_nodes = getattr(batch, "x_text", None)
                assert E_txt_nodes is not None, "需要 batch.x_text 以支持文本分支（precomputed 模式）"
            else:
                with torch.set_grad_enabled(not self.config.freeze_omiclip):
                    E_txt_nodes = self.omiclip_model.encode_text(batch.x_text)
            Z_gene = self._select_centers(E_txt_nodes, batch)
        else:
            # 其他模式保持原逻辑：双侧 mean 池化
            Z_image = global_mean_pool(img_feat, batch.batch)
            Z_gene = global_mean_pool(gene_feat, batch.batch)

        final_image_features = self.image_proj_head(Z_image)
        final_text_features  = self.gene_proj_head(Z_gene)

        # 3. 返回与训练代码兼容的输出
        return {
            "image_features": F.normalize(final_image_features, dim=-1),
            "text_features": F.normalize(final_text_features, dim=-1),
            "logit_scale": self.logit_scale.exp(),
        }
