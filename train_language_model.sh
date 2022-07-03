export MODEL_NAME="GPT3_8B"
export BS=16
export CS=184
export CPU_EBD=0
export SP=1
# export LIGHTSEQ_FLAG=0
export ACT_OFFLOAD=0
export NO_RETRY=0
export SKIP_LOG_EXSIT=0
export AMM=1
export MSC=1
export CACHE=0
export GPU_NUM=4
export RES_CHECK=0
export MEM_PROF=1
# export CS_SEARCH=1
export SUFFIX="round1"
# export MASTER_PORT="12345"

bash ./run_transformers.sh