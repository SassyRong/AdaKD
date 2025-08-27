#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2098}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-4}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"./"}
CKPT_NAME="gpt2-base"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
TEACHER_CKPT_NAME="gpt2-xlarge-sft"
TEACHER_CKPT="${BASE_PATH}/results/gpt2/train/sft_xlarge/e20-bs16-lr0.0001-G2-N4-NN1/ckpt_epochs/epoch-20_step-1780"
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
# hp
BATCH_SIZE=32
LR=5e-4
GRAD_ACC=1
EVAL_BATCH_SIZE=128
# length
MAX_LENGTH=512
# seed
SEED=${4-10}
# runtime
RULE=${5-"nmtkd"}
LOSS_FN=${6-"rkd"}
T_SCALE=${7-0.3}
SIGMA=${8-0.05}
EPISILON=${9-0.05}
SAVE_PATH="${BASE_PATH}/results/gpt2/train/daad/daad_rule[${RULE}]_loss[${LOSS_FN}]_c${T_SCALE}_sigma${SIGMA}_episilon${EPISILON}/seed${SEED}"

START_TIME=$(date +"%Y-%m-%d_%H-%M-%S")

OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --model-type gpt2"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 20"
OPTS+=" --kd-ratio 1.0"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type adaptive-focuskd"
OPTS+=" --rule ${RULE}"
OPTS+=" --loss-fn ${LOSS_FN}"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# DAAD
OPTS+=" --loss-ratio-eps ${EPISILON}"
OPTS+=" --sample-ratio-eps ${SIGMA}"
OPTS+=" --adaptive-temperature"
# OPTS+=" --temperature-soften"
OPTS+=" --temperature-scale ${T_SCALE}"

OPTS+=" --distillm"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"
# OPTS+=" --gkd"
# OPTS+=" --gkd-alpha 0.5"
# OPTS+=" --wandb-offline"
OPTS+=" --ema-decay 0.985"
# OPTS+=" --model-ema"
OPTS+=" --wandb-name abkd_1.5Bto0.1B_bs${BATCH_SIZE}_lr${LR}_G${GRAD_ACC}_seed${SEED}"
OPTS+=" --wandb-project distill_gpt2_new"

export NCCL_DEBUG=""
export WANDB_DISABLED=False
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/distillation.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
