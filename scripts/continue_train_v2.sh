#!/bin/bash

# ==============================================================================
# SpaGLaM SOTA 串行实验主管脚本 v2.1 (支持断点续训)
# ==============================================================================
# 用法:
# 1. 在 LOGS_BASE_DIR 中填入上次运行的、包含 checkpoint 的日志目录。
# 2. 在 EXPERIMENTS 数组中，注释掉或删除已经完成的实验。
# 3. 运行此脚本: `bash continue_train_v2.sh`

set -e
# 确保在正确的源码目录下执行
cd /cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/git_repo/GoldSpace/src

echo "🚀 开始断点续训流程..."

# --- 1. 通用配置 ---
NPROC_PER_NODE=4
MASTER_PORT=29501
BASE_MODEL_PATH="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/openclip_train/train_log/finetune_20250623-092915/2025_06_23-09_30_48-model_ViT-B-32-lr_2e-05-b_1800-j_16-p_amp_bfloat16/checkpoints/epoch_25.pt"
TRAIN_NUM_SAMPLES=11503600

# 【修改 1】: 将 LOGS_BASE_DIR 指向你上次运行的、已经存在的日志目录
# 将 "SpaGLaM-Runs-V2-..." 替换成你上次运行生成的实际目录名
LOGS_BASE_DIR="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/spaglam_train/train_log/SpaGLaM-Runs-V2-20250720-043613" # <--- 替换成你真实的目录名

echo "将从以下目录恢复并继续实验: ${LOGS_BASE_DIR}"
if [ ! -d "$LOGS_BASE_DIR" ]; then
    echo "❌ 错误: 日志目录不存在! 请检查 LOGS_BASE_DIR 路径。"
    exit 1
fi

# --- 2. 数据集路径 (保持不变) ---
TRAIN_DATA_URL="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/my_data/spaglam_embedding_shards/spaglam_embedding_shards/shard-{000000..000085}.tar"
TEST_DATA_URL="/cpfs01/projects-HDD/cfff-afe2df89e32e_HDD/jjh_19301050235/my_data/spaglam_precom_subgraphs/shard-{000270..000292}.tar"
TEST_NUM_SAMPLES=200000

# --- 3. 通用训练参数 ---
# 【修改 2】: 添加 --resume latest 标志
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
  --resume              latest" # <-- 核心修改点，激活断点续训

# --- 4. 实验定义 ---
# 【修改 3】: 只保留需要继续或重新开始的实验
EXPERIMENTS=(
    "Exp1_GAT-base;gat;1e-4;false"
    # "Exp2_GAT-deep-fusion;gat;1e-4;true"               # 已完成，注释掉
    # "Exp3_GraphTF-base;graphtransformer;1e-4;false"   # 已完成，注释掉
    # "Exp4_GraphTF-deep-fusion;graphtransformer;1e-4;true" # 已完成，注释掉
    "Exp5_GAT-base-lr-low;gat;5e-5;false"
)

# --- 5. 串行训练循环 (代码逻辑不变) ---
for experiment in "${EXPERIMENTS[@]}"; do
    IFS=';' read -r EXP_NAME GNN_TYPE LR USE_FUSION <<< "$experiment"

    echo -e "\n========================================================================"
    echo "🚀  处理实验: ${EXP_NAME}"
    echo "========================================================================"

    EXP_ARGS=" \
      --name                ${EXP_NAME} \
      --gnn-type            ${GNN_TYPE} \
      --lr                  ${LR}" 

    if [ "$USE_FUSION" = "true" ]; then
        EXP_ARGS+=" --use-deep-fusion"
    fi

    FULL_CMD="eval torchrun \
      --nproc_per_node=${NPROC_PER_NODE} \
      --master_port=${MASTER_PORT} \
      -m open_clip_train.main \
      ${COMMON_ARGS} \
      ${EXP_ARGS}"

    LOG_DIR="${LOGS_BASE_DIR}/${EXP_NAME}"
    LOG_FILE="${LOG_DIR}/train_output_resumed.log" # 使用新的日志文件名以避免覆盖
    mkdir -p "${LOG_DIR}"

    echo "日志将追加到: ${LOG_FILE}"
    echo "运行命令:"
    echo "${FULL_CMD}"

    # 使用 'tee -a' 来追加日志，而不是覆盖
    ${FULL_CMD} 2>&1 | tee -a "${LOG_FILE}"
    MASTER_PORT=$((MASTER_PORT + 1))
done

# --- 6. 测试阶段 (代码逻辑不变) ---
echo -e "\n========================================================================"
echo "🏁 所有训练完成，开始统一测试阶段..."
echo "========================================================================"
# 【修改 4】: 在测试部分，我们可以重新启用所有实验进行统一评估
ALL_EXPERIMENTS=(
    "Exp1_GAT-base;gat;1e-4;false"
    "Exp2_GAT-deep-fusion;gat;1e-4;true"
    "Exp3_GraphTF-base;graphtransformer;1e-4;false"
    "Exp4_GraphTF-deep-fusion;graphtransformer;1e-4;true"
    "Exp5_GAT-base-lr-low;gat;5e-5;false"
)

for experiment in "${ALL_EXPERIMENTS[@]}"; do
    IFS=';' read -r EXP_NAME GNN_TYPE _ USE_FUSION <<< "$experiment"
    echo -e "\n------------------------------------------------------------------------"
    echo "🧪  测试实验: ${EXP_NAME}"
    echo "------------------------------------------------------------------------"

    MODEL_CKPT="${LOGS_BASE_DIR}/${EXP_NAME}/checkpoints/epoch_50.pt"
    if [ ! -f "$MODEL_CKPT" ]; then
        echo "⚠️  警告: 找不到最终检查点 ${MODEL_CKPT}，跳过测试。"
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
      --gnn-type            ${GNN_TYPE}"
    
    if [ "$USE_FUSION" = "true" ]; then
        TEST_CMD+=" --use-deep-fusion"
    fi

    echo "测试命令:"
    echo "${TEST_CMD}"

    ${TEST_CMD} 2>&1 | tee "${TEST_LOG_DIR}/test_output.log"
    MASTER_PORT=$((MASTER_PORT + 1))
done

echo -e "\n✅ 所有实验训练与测试已完成。查看日志: ${LOGS_BASE_DIR}"