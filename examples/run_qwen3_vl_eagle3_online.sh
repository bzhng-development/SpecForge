#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)

# Configuration
MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct
CACHE_DIR=${CACHE_DIR:-$ROOT_DIR/cache}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT_DIR/outputs/Qwen3-VL-30B-A3B-eagle3}

# support tp1 train eagle3 for Qwen3-VL-30B-A3B-Instruct
NUM_GPUS=${1:-1}

# Optional: SGLang server setup for data generation (uncomment if needed)
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m sglang.launch_server \
#     --model $MODEL_PATH \
#     --cuda-graph-bs 1 2 4 8 16 32 64 128 256 \
#     --mem-frac=0.8 --port 30001 --tp 4

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $ROOT_DIR/configs/qwen3-vl-30b-a3b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/allava4v_train.jsonl \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --chat-template qwen3-vl \
    --cache-dir $CACHE_DIR \
    --embedding-key model.language_model.embed_tokens.weight \
    --tp-size 1 \
    --is-vlm \
    --min-pixels 50176 \
    --max-pixels 802816 \
    --attention-backend flex_attention \
    --resume \
    --wandb-project qwen3-vl-30b-eagle3 \
    --wandb-name vl-eagle3-training \
    --report-to wandb

# Optional: Upload trained model to HuggingFace Hub (uncomment if needed)
# HF_REPO_NAME="your-username/qwen3-vl-30b-eagle3"
# hf repo create $HF_REPO_NAME
# hf upload $HF_REPO_NAME \
#     $OUTPUT_DIR \
#     --commit-message "Upload trained Qwen3-VL-30B EAGLE3 model" \
#     --repo-type model
