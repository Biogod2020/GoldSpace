#!/bin/bash

# ==============================================================================
# SpaGLaM SOTA 串行实验主管脚本 v3
# (支持2-hop子图, 多架构对比, 残差连接)
# ==============================================================================
# 用法:
# 1. 确认下面的路径配置正确。
# 2. 运行此脚本: `bash sequentially_run_spaglam_v3.sh`

set -e
# 确保在正确的源码目录下执行
cd /cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/git_repo/GoldSpace/src

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
echo "实验开始于: $TIMESTAMP"

# --- 1. 通用配置 ---
NPROC_PER_NODE=4
MASTER_PORT=29502 # 使用一个新端口以防冲突
BASE_MODEL_PATH="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/openclip_train/train_log/finetune_20250623-092915/2025_06_23-09_30_48-model_ViT-B-32-lr_2e-05-b_1800-j_16-p_amp_bfloat16/checkpoints/epoch_25.pt"
# 训练样本总数 (请根据2-hop数据重新估算，或暂时使用一个大概的值)
TRAIN_NUM_SAMPLES=11503600
# 实验日志总根目录
LOGS_BASE_DIR="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/spaglam_train/train_log/SpaGLaM-Runs-2Hops-${TIMESTAMP}"

# --- 2. 数据集路径 ---
# 【修改 1】: 训练数据指向新的 2-hop pre-computed embedding shards
TRAIN_DATA_URL="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/my_data/spaglam_embedding_shards_2hops/shard-{000000..000090}.tar"

# 测试数据保持不变 (用于评估)
TEST_DATA_URL="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/my_data/spaglam_precom_subgraphs/shard-{000270..000292}.tar"
TEST_NUM_SAMPLES=200000

# --- 3. 通用训练参数 (移除deep fusion) ---
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
  --freeze-omiclip \
  --gnn-layers          2" # 固定GNN层数为2

# --- 4. 实验定义 (全新) ---
# 格式: "实验名;GNN类型;学习率;是否使用残差连接(true/false)"
EXPERIMENTS=(
    "Exp1_GAT_no-res;gat;1e-4;false"
    "Exp2_GAT_with-res;gat;1e-4;true"
    "Exp3_GraphTF_no-res;graphtransformer;1e-4;false"
    "Exp4_GraphTF_with-res;graphtransformer;1e-4;true"
    "Exp5_TransformerConv_no-res;transformerconv;1e-4;false"
    "Exp6_TransformerConv_with-res;transformerconv;1e-4;true"
    "Exp7_GAT_with-res_lr-low;gat;5e-5;true"
    "Exp8_TransformerConv_with-res_lr-low;transformerconv;5e-5;true"
)

# --- 5. 串行训练循环 (更新逻辑) ---
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=';' read -r EXP_NAME GNN_TYPE LR USE_RESIDUAL <<< "$experiment"

    echo -e "\n========================================================================"
    echo "🚀  开始训练实验: ${EXP_NAME}"
    echo "========================================================================"

    EXP_ARGS=" \
      --name                ${EXP_NAME} \
      --gnn-type            ${GNN_TYPE} \
      --lr                  ${LR}" 

    # 【修改 2】: 根据实验定义动态添加残差连接标志
    if [ "$USE_RESIDUAL" = "true" ]; then
        EXP_ARGS+=" --use-residual-connection"
    fi

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

# --- 6. 测试阶段 (更新逻辑) ---
echo -e "\n========================================================================"
echo "🏁 所有训练完成，开始统一测试阶段..."
echo "========================================================================"
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=';' read -r EXP_NAME GNN_TYPE _ USE_RESIDUAL <<< "$experiment"
    echo -e "\n------------------------------------------------------------------------"
    echo "🧪  测试实验: ${EXP_NAME}"
    echo "------------------------------------------------------------------------"

    MODEL_CKPT="${LOGS_BASE_DIR}/${EXP_NAME}/checkpoints/epoch_50.pt"
    if [ ! -f "$MODEL_CKPT" ]; then
        echo "⚠️  警告: 找不到检查点 ${MODEL_CKPT}，跳过测试。"
        continue
    fi
    
    TEST_LOG_DIR="${LOGS_BASE_DIR}/${EXP_NAME}/test_logs"
    mkdir -p "${TEST_LOG_DIR}"

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
      --gnn-layers          2 \
      --gnn-type            ${GNN_TYPE}"
    
    # 【修改 3】: 在测试时也需要正确传递残差连接标志，以确保模型结构能正确重建
    if [ "$USE_RESIDUAL" = "true" ]; then
        TEST_CMD+=" --use-residual-connection"
    fi

    echo "测试命令:"
    echo "${TEST_CMD}"

    ${TEST_CMD} 2>&1 | tee "${TEST_LOG_DIR}/test_output.log"
    MASTER_PORT=$((MASTER_PORT + 1))
done

echo -e "\n✅ 所有实验训练与测试已完成。查看日志: ${LOGS_BASE_DIR}"