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
