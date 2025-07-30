#!/bin/bash

# ==============================================================================
# SpaGLaM SOTA 串行实验主管脚本 v2 (支持预计算 Embeddings 训练和原始数据测试)
# ==============================================================================
# 用法:
# 1. 确认下面的路径配置正确。
# 2. 运行此脚本: `bash sequentially_run_spaglam_v2.sh`

set -e
# 确保在正确的源码目录下执行
cd /cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/git_repo/GoldSpace/src

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
echo "实验开始于: $TIMESTAMP"

# --- 1. 通用配置 ---
NPROC_PER_NODE=4
MASTER_PORT=29501
# OmiCLIP 基础模型路径，用于初始化 SpaGLaM 结构和加载 logit_scale
BASE_MODEL_PATH="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/openclip_train/train_log/finetune_20250623-092915/2025_06_23-09_30_48-model_ViT-B-32-lr_2e-05-b_1800-j_16-p_amp_bfloat16/checkpoints/epoch_25.pt"
# 训练样本总数
TRAIN_NUM_SAMPLES=11503600
# 实验日志总根目录
LOGS_BASE_DIR="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/spaglam_train/train_log/SpaGLaM-Runs-V2-${TIMESTAMP}"

# --- 2. 数据集路径 ---
# 【修改 1】: 训练数据指向 pre-computed embedding shards
TRAIN_DATA_URL="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/my_data/spaglam_embedding_shards/spaglam_embedding_shards/shard-{000000..000085}.tar"

# 【修改 2】: 测试数据指向原始的 raw data shards (包含 .png 和 .txt)
TEST_DATA_URL="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/my_data/spaglam_precom_subgraphs/shard-{000270..000292}.tar"
TEST_NUM_SAMPLES=200000 # 测试样本数

# --- 3. 通用训练参数 ---
# 【修改 3】: 
#   - 删除了不再需要的 --anndata-path
#   - 增加了新的 --use-precomputed-embeddings 标志
#   - 增大了 batch-size 和 workers，因为 IO 瓶颈已解决
#   - 将 --train-data 的路径用引号括起来，防止 shell 提前展开通配符
COMMON_ARGS=" \
  --model               ViT-B-32 \
  --pretrained          \"${BASE_MODEL_PATH}\" \
  --dataset-type        spaglam \
  --train-data          \"${TRAIN_DATA_URL}\" \
  --train-num-samples   ${TRAIN_NUM_SAMPLES} \
  --batch-size          1024 \
  --epochs              50 \
  --wd                  0.1 \
  --warmup              1000 \
  --precision           amp_bfloat16 \
  --workers             16 \
  --logs                ${LOGS_BASE_DIR} \
  --report-to           tensorboard \
  --save-frequency      5 \
  --log-every-n-steps   100 \
  --local-loss \
  --gather-with-grad \
  --use-spaglam-model \
  --use-precomputed-embeddings \
  --freeze-omiclip"  # 强烈建议在预计算模式下冻结，因为编码器不用

# --- 4. 实验定义 (保持不变) ---
EXPERIMENTS=(
    "Exp1_GAT-base;gat;1e-4;false"
    "Exp2_GAT-deep-fusion;gat;1e-4;true"
    "Exp3_GraphTF-base;graphtransformer;1e-4;false"
    "Exp4_GraphTF-deep-fusion;graphtransformer;1e-4;true"
    "Exp5_GAT-base-lr-low;gat;5e-5;false"
)

# --- 5. 串行训练循环 (保持不变) ---
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=';' read -r EXP_NAME GNN_TYPE LR USE_FUSION <<< "$experiment"

    echo -e "\n========================================================================"
    echo "🚀  开始训练实验: ${EXP_NAME}"
    echo "========================================================================"

    EXP_ARGS=" \
      --name                ${EXP_NAME} \
      --gnn-type            ${GNN_TYPE} \
      --lr                  ${LR}" 

    if [ "$USE_FUSION" = "true" ]; then
        EXP_ARGS+=" --use-deep-fusion"
    fi

    # 使用 eval 来正确处理带引号的路径参数
    FULL_CMD="eval torchrun \
      --nproc_per_node=${NPROC_PER_NODE} \
      --master_port=${MASTER_PORT} \
      -m open_clip_train.main \
      ${COMMON_ARGS} \
      ${EXP_ARGS}"

    LOG_DIR="${LOGS_BASE_DIR}/${EXP_NAME}"
    LOG_FILE="${LOG_DIR}/train_output.log"
    mkdir -p "${LOG_DIR}"

    echo "日志将保存在: ${LOG_FILE}"
    echo "运行命令:"
    echo "${FULL_CMD}"

    ${FULL_CMD} 2>&1 | tee "${LOG_FILE}"
    MASTER_PORT=$((MASTER_PORT + 1))
done

# --- 6. 测试阶段 (已更新) ---
echo -e "\n========================================================================"
echo "🏁 所有训练完成，开始统一测试阶段..."
echo "========================================================================"
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=';' read -r EXP_NAME GNN_TYPE _ USE_FUSION <<< "$experiment"
    echo -e "\n------------------------------------------------------------------------"
    echo "🧪  测试实验: ${EXP_NAME}"
    echo "------------------------------------------------------------------------"

    # 找到该实验训练好的最后一个检查点
    MODEL_CKPT="${LOGS_BASE_DIR}/${EXP_NAME}/checkpoints/epoch_50.pt"
    if [ ! -f "$MODEL_CKPT" ]; then
        echo "⚠️  警告: 找不到检查点 ${MODEL_CKPT}，跳过测试。"
        continue
    fi
    
    TEST_LOG_DIR="${LOGS_BASE_DIR}/${EXP_NAME}/test_logs"
    mkdir -p "${TEST_LOG_DIR}"

    # 【修改 4】: 构建测试命令
    #   - 使用 --val-data 指向原始数据分片
    #   - 删除了 --use-precomputed-embeddings 标志，让模型激活内部编码器
    #   - 删除了 --train-* 参数，增加了 --val-num-samples
    #   - 增加了 --name 以便日志清晰
    #   - 把所有 SpaGLaM 相关的结构参数都传进去，确保模型能正确重建
    TEST_CMD="eval torchrun \
      --nproc_per_node=${NPROC_PER_NODE} \
      --master_port=${MASTER_PORT} \
      -m open_clip_train.main \
      --model               ViT-B-32 \
      --pretrained          \"${BASE_MODEL_PATH}\" \
      --resume              \"${MODEL_CKPT}\" \
      --dataset-type        spaglam \
      --val-data            \"${TEST_DATA_URL}\" \
      --val-num-samples     ${TEST_NUM_SAMPLES} \
      --batch-size          64 \
      --workers             8 \
      --precision           amp_bfloat16 \
      --logs                ${TEST_LOG_DIR} \
      --name                ${EXP_NAME}_test \
      --use-spaglam-model \
      --gnn-type            ${GNN_TYPE}"
    
    # 动态添加深度融合标志
    if [ "$USE_FUSION" = "true" ]; then
        TEST_CMD+=" --use-deep-fusion"
    fi

    echo "测试命令:"
    echo "${TEST_CMD}"

    ${TEST_CMD} 2>&1 | tee "${TEST_LOG_DIR}/test_output.log"
    MASTER_PORT=$((MASTER_PORT + 1))
done

echo -e "\n✅ 所有实验训练与测试已完成。查看日志: ${LOGS_BASE_DIR}"