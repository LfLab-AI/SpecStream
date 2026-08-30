# ======================== 模型 ========================
export DRAFT_MODEL=/root/autodl-tmp/model/Qwen2.5-0.5B-Instruct
export TARGET_MODEL=/root/autodl-tmp/model/Qwen2.5-7B-Instruct
export BASE_URL=http://127.0.0.1:30000
##export SHAREGPT_V3_ROOT=/common_data/dataset/ShareGPT_V3
export PREPARED_ROOT=/root/lifei/SpecStream/specstream_prepared ##/home/lifei/lifei/specdecode/baseline/sglang/specstream_prepared
export DATASET=$PREPARED_ROOT/sharegpt_v3_merged.json
export RUN_ROOT=$PWD/results/innovation2_step3_$(date +%Y%m%d_%H%M%S)

export SMCTRL_LIB=$PWD/csrc/specstream_smctrl/build/libsmctrl.so
export MPS_PIPE=/tmp/specstream-mps-$USER
export MPS_LOG=/tmp/specstream-mps-log-$USER
export REPO=/root/lifei/SpecStream

# ======================== 数据集 ========================
# 你的服务器当前数据集根目录：
export DATA_ROOT=$HOME/autodl-tmp/dataset

# 原始 GSM8K 目录（Hugging Face 仓库布局：README.md / eval.yaml / main / socratic）
export GSM8K_ROOT=$DATA_ROOT/gsm8k

# 原始 LongBench v2 文件（你当前下载的是 data.json）
export LONGBENCH_V2_RAW=$DATA_ROOT/LongBench-v2/data.json

# 统一把论文实验实际读取的数据放到 prepared 目录，避免改动原始数据集。


# 下面两个文件由第 3 节的“本地数据适配”命令生成。
export GSM8K_TEST=$PREPARED_ROOT/gsm8k_main_test.jsonl
export LONGBENCH_V2=$PREPARED_ROOT/longbench_v2.jsonl

# ShareGPT V3：仅用于性能测试
export SHAREGPT_V3_ROOT=$DATA_ROOT/ShareGPT_V3
export SHAREGPT_JSON=$PREPARED_ROOT/sharegpt_v3_merged.json

# ======================== 服务端口 ========================
export BASE_URL=http://127.0.0.1:30000
export TARGET_PORT=30000
export DRAFT_PORT=30001
export ZMQ_PORT=29000

# ======================== GPU ========================
# 创新点一双卡实验：Target 与 Draft 分卡
export DRAFT_GPU=0
export TARGET_GPU=1

# 创新点二同卡实验必须使用物理 GPU UUID，而不是仅写逻辑编号。
# 先执行 nvidia-smi -L 后修改：
export SINGLE_GPU_UUID=GPU-342220f7-2293-1a7e-08df-73cec29f44f5

# ======================== 结果目录 ========================
export RESULT_ROOT=$REPO/results/paper
mkdir -p \
  $RESULT_ROOT/{accuracy,attention,bench,profiles,resource_profiles,smctrl,nsys,gpu_monitor,logs,source_data}
