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
class SpaGLaM(nn.Module):
    def __init__(self, open_clip_model: nn.Module, config: object):
        super().__init__()
        self.config = config
        self.omiclip_model = open_clip_model

        # --- 冻结策略 ---
        if config.freeze_omiclip:
            for param in self.omiclip_model.parameters():
                param.requires_grad = False
            self.omiclip_model.eval()

        # --- 获取维度信息 ---
        gnn_input_dim = self.omiclip_model.visual.output_dim
        gnn_hidden_dim = config.gnn_hidden_dim
        gnn_output_dim = self.omiclip_model.embed_dim

        # --- 构建并行的GNN塔和交互模块 ---
        self.gnn_layers_img = nn.ModuleList()
        self.gnn_layers_gene = nn.ModuleList()
        self.interaction_layers = nn.ModuleList() if config.use_deep_fusion else None

        current_dim = gnn_input_dim
        for _ in range(config.gnn_layers):
            if config.gnn_type == 'gat':
                # GATv2是比GAT更好的选择
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
        
        # --- 构建投影头 ---
        self.image_proj_head = MLPProjectionHead(gnn_hidden_dim, gnn_hidden_dim, gnn_output_dim)
        self.gene_proj_head = MLPProjectionHead(gnn_hidden_dim, gnn_hidden_dim, gnn_output_dim)

        # --- 共享的 Logit Scale ---
        self.logit_scale = self.omiclip_model.logit_scale

    def forward(self, batch: "torch_geometric.data.Batch") -> dict:
        # 1. 初始特征提取
        with torch.set_grad_enabled(not self.config.freeze_omiclip):
            # batch.x_image: (total_nodes, C, H, W), batch.x_text: (total_nodes, L)
            E_image = self.omiclip_model.encode_image(batch.x_image)
            E_gene = self.omiclip_model.encode_text(batch.x_text)

        # 2. GNN传播与深度融合
        img_feat, gene_feat = E_image, E_gene
        for i in range(self.config.gnn_layers):
            # GNN层
            if self.config.gnn_type == 'graphtransformer':
                img_feat = self.gnn_layers_img[i](img_feat, batch.batch)
                gene_feat = self.gnn_layers_gene[i](gene_feat, batch.batch)
            else: # GAT
                img_feat = self.gnn_layers_img[i](img_feat, batch.edge_index)
                gene_feat = self.gnn_layers_gene[i](gene_feat, batch.edge_index)
            
            img_feat = F.gelu(img_feat)
            gene_feat = F.gelu(gene_feat)

            # 深度交互层 (如果启用)
            if self.interaction_layers is not None:
                img_feat, gene_feat = self.interaction_layers[i](img_feat, gene_feat)
        
        # 3. 读出 (Readout)
        # 使用全局平均池化来聚合每个子图的节点信息，得到一个代表该邻域上下文的向量
        Z_image = global_mean_pool(img_feat, batch.batch)
        Z_gene = global_mean_pool(gene_feat, batch.batch)

        # 4. 投影
        final_image_features = self.image_proj_head(Z_image)
        final_text_features = self.gene_proj_head(Z_gene)

        # 5. 返回与open-clip损失函数兼容的字典
        return {
            "image_features": F.normalize(final_image_features, dim=-1),
            "text_features": F.normalize(final_text_features, dim=-1),
            "logit_scale": self.logit_scale.exp(),
        }