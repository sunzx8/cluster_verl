#!/usr/bin/env bash
set -uxo pipefail

# 设置默认路径 - 下载到BNTO_verl/data目录
export VERL_HOME=${VERL_HOME:-"${PWD}"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}

# 创建数据目录
mkdir -p "${VERL_HOME}/data"

echo "开始下载DAPO数据集..."

# 下载训练数据 - DAPO-Math-17k
if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  echo "下载训练数据: DAPO-Math-17k..."
  wget -O "${TRAIN_FILE}" "https://hf-mirror.com/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
  echo "训练数据下载完成: ${TRAIN_FILE}"
else
  echo "训练数据已存在: ${TRAIN_FILE}"
fi

# 下载测试数据 - AIME-2024
if [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; then
  echo "下载测试数据: AIME-2024..."
  wget -O "${TEST_FILE}" "https://hf-mirror.com/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
  echo "测试数据下载完成: ${TEST_FILE}"
else
  echo "测试数据已存在: ${TEST_FILE}"
fi

echo "数据集下载完成！"
echo "训练数据: ${TRAIN_FILE}"
echo "测试数据: ${TEST_FILE}"
echo ""
echo "使用说明:"
echo "1. 设置环境变量: export TRAIN_FILE=\"${TRAIN_FILE}\""
echo "2. 设置环境变量: export TEST_FILE=\"${TEST_FILE}\""
echo "3. 强制重新下载: OVERWRITE=1 bash scripts/prepare_dapo_data.sh" 