#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash experiments/run_llama3_1_8b_oe_ct.sh [outputs|labels|train|scale|eval|all]

Purpose:
  Run the paper-faithful open-ended calibration-tuning pipeline adapted to
  Meta Llama 3.1 8B Instruct:
    1. generate outputs on all_20k_uniform
    2. generate fuzzy GPT query labels
    3. calibration-tune the query adapter
    4. temperature-scale the query logits
    5. evaluate on mmlu_all with mode=query

Key environment variables:
  PROJECT               default: /home/bnreed/projects/calibration-tuning
  DATA_DIR              default: $PROJECT/datasets
  MODEL                 default: llama3_1:8b-instruct
  RUN                   default: llama3_1_8b_instruct_oe
  GPU                   default: 0,1
  NPROC_PER_NODE        default: 2
  OPENAI_API_KEY        required for labels/eval in the faithful OE pipeline

Optional overrides:
  LABEL_STRATEGY        default: gpt-4.1-2025-04-14 # BNR fuzzy_gpt-3.5-turbo-1106
  EVAL_GRADE_STRATEGY   default: $LABEL_STRATEGY
  OUTPUT_SOURCE_RUN     default: empty
  OUTPUT_SOURCE_LOG_DIR default: empty
  OUTPUT_BATCH_SIZE     default: 4
  LABEL_BATCH_SIZE      default: 8
  TRAIN_BATCH_SIZE      default: 1
  TRAIN_GRAD_ACCUM      default: 8
  TRAIN_MAX_STEPS       default: 5000
  TRAIN_SAVE_STEPS      default: 200
  TRAIN_LR              default: 1e-4
  SCALE_BATCH_SIZE      default: 8
  SCALE_MAX_STEPS       default: 2000
  SCALE_SAVE_STEPS      default: 200
  SCALE_LR              default: 1e-3
  EVAL_BATCH_SIZE       default: 8
  FORCE_LABELS          default: 0

Examples:
  bash experiments/run_llama3_1_8b_oe_ct.sh all
  bash experiments/run_llama3_1_8b_oe_ct.sh train
  bash experiments/run_llama3_1_8b_oe_ct.sh eval
  OUTPUT_SOURCE_RUN=llama3_1_8b_instruct_oe ./experiments/run_llama3_1_8b_oe_ct.sh labels

Notes:
  - This script uses the open-ended pipeline, not the choice pipeline.
  - Any LABEL_STRATEGY / EVAL_GRADE_STRATEGY other than substring is treated as
    an OpenAI equivalency grader model. Bare model IDs are accepted.
  - OUTPUT_SOURCE_RUN / OUTPUT_SOURCE_LOG_DIR lets a new RUN reuse an older
    outputs directory without copying files.
  - It resumes training/temperature scaling from the latest checkpoint if found.
  - Output generation is resumed by reusing the same log directory.
  - NPROC_PER_NODE affects train/scale only. outputs/eval remain single-process.
EOF
}

PROJECT="${PROJECT:-/home/bnreed/projects/calibration-tuning}"
DATA_DIR="${DATA_DIR:-$PROJECT/datasets}"
MODEL="${MODEL:-llama3_1:8b-instruct}"
RUN="${RUN:-llama3_1_8b_instruct_oe}"
GPU="${GPU:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PROMPT_STYLE="${PROMPT_STYLE:-oe}"
MMLU_DATASET="${MMLU_DATASET:-mmlu_all}"
#LABEL_STRATEGY="${LABEL_STRATEGY:-fuzzy_gpt-3.5-turbo-1106}"
LABEL_STRATEGY="${LABEL_STRATEGY:-gpt-4.1-2025-04-14}"
EVAL_GRADE_STRATEGY="${EVAL_GRADE_STRATEGY:-$LABEL_STRATEGY}"
OUTPUT_SOURCE_RUN="${OUTPUT_SOURCE_RUN:-}"
OUTPUT_SOURCE_LOG_DIR="${OUTPUT_SOURCE_LOG_DIR:-}"

OUTPUT_BATCH_SIZE="${OUTPUT_BATCH_SIZE:-4}"
LABEL_BATCH_SIZE="${LABEL_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-8}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-5000}"
TRAIN_SAVE_STEPS="${TRAIN_SAVE_STEPS:-200}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_INT8="${TRAIN_INT8:-True}"
KL_DECAY="${KL_DECAY:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-30}"

SCALE_BATCH_SIZE="${SCALE_BATCH_SIZE:-8}"
SCALE_MAX_STEPS="${SCALE_MAX_STEPS:-2000}"
SCALE_SAVE_STEPS="${SCALE_SAVE_STEPS:-200}"
SCALE_LR="${SCALE_LR:-1e-3}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
FORCE_LABELS="${FORCE_LABELS:-0}"

OUTPUT_LOG_DIR="${OUTPUT_LOG_DIR:-$DATA_DIR/tmp/${RUN}_outputs}"
LABEL_LOG_DIR="${LABEL_LOG_DIR:-$DATA_DIR/tmp/${RUN}_labels}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-$PROJECT/logs/${RUN}_ct}"
SCALE_LOG_DIR="${SCALE_LOG_DIR:-$PROJECT/logs/${RUN}_ts}"

export WANDB_MODE="${WANDB_MODE:-offline}"

latest_checkpoint() {
  local root="$1"
  ls -d "$root"/checkpoint-* 2>/dev/null | sort -V | tail -1
}

gpu_count() {
  awk -F',' '{print NF}' <<< "$GPU"
}

validate_distributed_config() {
  if ! [[ "$NPROC_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NPROC_PER_NODE" -lt 1 ]]; then
    echo "NPROC_PER_NODE must be a positive integer. Got: $NPROC_PER_NODE" >&2
    exit 1
  fi

  local ngpu
  ngpu="$(gpu_count)"
  if [[ "$NPROC_PER_NODE" -gt "$ngpu" ]]; then
    echo "NPROC_PER_NODE=$NPROC_PER_NODE but GPU=$GPU exposes only $ngpu device(s)." >&2
    exit 1
  fi
}

have_csv_outputs() {
  local root="$1"
  find "$root" -name '*.csv' -print -quit 2>/dev/null | grep -q .
}

have_split_csvs() {
  local root="$1"
  local split="$2"
  find "$root/$split" -maxdepth 1 -name '*.csv' -print -quit 2>/dev/null | grep -q .
}

ensure_openai_key() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required for this step." >&2
    exit 1
  fi
}

ensure_offline_links_dir() {
  mkdir -p "$DATA_DIR/offline" "$DATA_DIR/tmp" "$PROJECT/logs"
}

offline_outputs_path() {
  echo "$DATA_DIR/offline/${RUN}_outputs-${PROMPT_STYLE}"
}

offline_labels_path() {
  echo "$DATA_DIR/offline/${RUN}_labels-${PROMPT_STYLE}"
}

resolve_output_source_log_dir() {
  if [[ -n "$OUTPUT_SOURCE_LOG_DIR" ]]; then
    echo "$OUTPUT_SOURCE_LOG_DIR"
  elif [[ -n "$OUTPUT_SOURCE_RUN" ]]; then
    echo "$DATA_DIR/tmp/${OUTPUT_SOURCE_RUN}_outputs"
  else
    echo "$OUTPUT_LOG_DIR"
  fi
}

link_outputs() {
  ensure_offline_links_dir
  local source_log_dir
  source_log_dir="$(resolve_output_source_log_dir)"
  ln -sfn "$source_log_dir/outputs" "$(offline_outputs_path)"
}

link_labels() {
  ensure_offline_links_dir
  ln -sfn "$LABEL_LOG_DIR/labels" "$(offline_labels_path)"
}

ensure_outputs_dataset() {
  local source_log_dir
  source_log_dir="$(resolve_output_source_log_dir)"

  if [[ -d "$source_log_dir/outputs" ]]; then
    link_outputs
  fi

  if [[ ! -e "$(offline_outputs_path)" ]]; then
    echo "Missing generated outputs at $source_log_dir/outputs" >&2
    echo "Run: ./experiments/run_llama3_1_8b_oe_ct.sh outputs" >&2
    exit 1
  fi

  if ! have_csv_outputs "$source_log_dir/outputs"; then
    echo "No output CSVs found under $source_log_dir/outputs" >&2
    echo "Run the outputs step first or check the run name/path." >&2
    exit 1
  fi

  for split in train validation; do
    if ! have_split_csvs "$source_log_dir/outputs" "$split"; then
      echo "Missing $split output CSVs under $source_log_dir/outputs" >&2
      echo "Re-run: ./experiments/run_llama3_1_8b_oe_ct.sh outputs" >&2
      exit 1
    fi
  done
}

ensure_labels_dataset() {
  if [[ -d "$LABEL_LOG_DIR/labels" ]]; then
    link_labels
  fi

  if [[ ! -e "$(offline_labels_path)" ]]; then
    echo "Missing generated labels at $LABEL_LOG_DIR/labels" >&2
    echo "Run: ./experiments/run_llama3_1_8b_oe_ct.sh labels" >&2
    exit 1
  fi

  if ! have_csv_outputs "$LABEL_LOG_DIR/labels"; then
    echo "No label CSVs found under $LABEL_LOG_DIR/labels" >&2
    echo "Run the labels step first or check the run name/path." >&2
    exit 1
  fi

  for split in train validation; do
    if ! have_split_csvs "$LABEL_LOG_DIR/labels" "$split"; then
      echo "Missing $split label CSVs under $LABEL_LOG_DIR/labels" >&2
      echo "Re-run: ./experiments/run_llama3_1_8b_oe_ct.sh labels" >&2
      exit 1
    fi
  done
}

run_outputs() {
  ensure_offline_links_dir

  local source_log_dir
  source_log_dir="$(resolve_output_source_log_dir)"

  if [[ "$source_log_dir" != "$OUTPUT_LOG_DIR" ]]; then
    if [[ ! -d "$source_log_dir/outputs" ]]; then
      echo "Requested output reuse, but source outputs dir does not exist: $source_log_dir/outputs" >&2
      exit 1
    fi
    echo "Reusing outputs from $source_log_dir/outputs"
    link_outputs
    return
  fi

  CUDA_VISIBLE_DEVICES="$GPU" python experiments/generate.py outputs \
    --dataset=all_20k_uniform \
    --prompt-style="$PROMPT_STYLE" \
    --model-name="$MODEL" \
    --max-new-tokens="$MAX_NEW_TOKENS" \
    --batch-size="$OUTPUT_BATCH_SIZE" \
    --int8=True \
    --data-dir="$DATA_DIR" \
    --log-dir="$OUTPUT_LOG_DIR"

  link_outputs
}

run_labels() {
  ensure_offline_links_dir
  ensure_outputs_dataset

  if [[ "$LABEL_STRATEGY" != "substring" ]]; then
    ensure_openai_key
  fi

  if [[ "$FORCE_LABELS" != "1" ]] && have_csv_outputs "$LABEL_LOG_DIR/labels"; then
    echo "Skipping label generation because label CSVs already exist under $LABEL_LOG_DIR/labels"
    echo "Set FORCE_LABELS=1 to regenerate them."
    link_labels
    return
  fi

  python experiments/generate.py labels \
    --dataset="offline:${RUN}_outputs" \
    --prompt-style="$PROMPT_STYLE" \
    --model-name="$MODEL" \
    --strategy="$LABEL_STRATEGY" \
    --batch-size="$LABEL_BATCH_SIZE" \
    --data-dir="$DATA_DIR" \
    --log-dir="$LABEL_LOG_DIR"

  link_labels
}

run_train() {
  ensure_offline_links_dir
  ensure_labels_dataset
  validate_distributed_config

  local resume_ckpt
  resume_ckpt="$(latest_checkpoint "$TRAIN_LOG_DIR" || true)"

  local cmd=(
    torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" experiments/calibration_tune.py
    --dataset="offline:${RUN}_labels"
    --prompt-style="$PROMPT_STYLE"
    --data-dir="$DATA_DIR"
    --model-name="$MODEL"
    --int8="$TRAIN_INT8"
    --batch-size="$TRAIN_BATCH_SIZE"
    --gradient-accumulation-steps="$TRAIN_GRAD_ACCUM"
    --lora-rank=8
    --lora-alpha=32
    --lora-dropout=0.1
    --lr="$TRAIN_LR"
    --kl-decay="$KL_DECAY"
    --max-steps="$TRAIN_MAX_STEPS"
    --save-steps="$TRAIN_SAVE_STEPS"
    --log-dir="$TRAIN_LOG_DIR"
  )

  if [[ -n "$resume_ckpt" ]]; then
    echo "Resuming calibration tuning from $resume_ckpt"
    cmd+=(--resume-from-checkpoint="$resume_ckpt")
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
}

run_scale() {
  ensure_offline_links_dir
  ensure_labels_dataset
  validate_distributed_config

  local ct_ckpt
  ct_ckpt="$(latest_checkpoint "$TRAIN_LOG_DIR" || true)"
  if [[ -z "$ct_ckpt" ]]; then
    echo "No calibration-tune checkpoint found under $TRAIN_LOG_DIR" >&2
    exit 1
  fi

  local resume_ckpt
  resume_ckpt="$(latest_checkpoint "$SCALE_LOG_DIR" || true)"

  local cmd=(
    torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" experiments/calibration_tune.py
    --dataset="offline:${RUN}_labels"
    --prompt-style="$PROMPT_STYLE"
    --data-dir="$DATA_DIR"
    --model-name="$MODEL"
    --int8="$TRAIN_INT8"
    --peft-dir="$ct_ckpt"
    --scale-temp=True
    --batch-size="$SCALE_BATCH_SIZE"
    --lr="$SCALE_LR"
    --max-steps="$SCALE_MAX_STEPS"
    --save-steps="$SCALE_SAVE_STEPS"
    --log-dir="$SCALE_LOG_DIR"
  )

  if [[ -n "$resume_ckpt" ]]; then
    echo "Resuming query temperature scaling from $resume_ckpt"
    cmd+=(--resume-from-checkpoint="$resume_ckpt")
  fi

  CUDA_VISIBLE_DEVICES="$GPU" "${cmd[@]}"
}

run_eval() {
  if [[ "$EVAL_GRADE_STRATEGY" != "substring" ]]; then
    ensure_openai_key
  fi

  local ts_ckpt
  ts_ckpt="$(latest_checkpoint "$SCALE_LOG_DIR" || true)"
  if [[ -z "$ts_ckpt" ]]; then
    echo "No temperature-scaled checkpoint found under $SCALE_LOG_DIR" >&2
    echo "Run the 'scale' step first." >&2
    exit 1
  fi

  python experiments/evaluate.py \
    --dataset="$MMLU_DATASET" \
    --prompt-style="$PROMPT_STYLE" \
    --model-name="$MODEL" \
    --query-peft-dir="$ts_ckpt" \
    --scale-temp=query \
    --mode=query \
    --grade-strategy="$EVAL_GRADE_STRATEGY" \
    --batch-size="$EVAL_BATCH_SIZE" \
    --data-dir="$DATA_DIR"
}

step="${1:-all}"

case "$step" in
  outputs)
    run_outputs
    ;;
  labels)
    run_labels
    ;;
  train)
    run_train
    ;;
  scale)
    run_scale
    ;;
  eval)
    run_eval
    ;;
  all)
    run_outputs
    run_labels
    run_train
    run_scale
    run_eval
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown step: $step" >&2
    usage >&2
    exit 1
    ;;
esac
