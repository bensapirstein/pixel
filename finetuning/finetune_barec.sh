# Optional wandb environment vars
export WANDB_PROJECT="pixel-readability-orthography-experiments"

# Settings
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
# export MODEL="bensapir/pixel-barec-pretrain" # also works with "bert-base-cased", "roberta-base", etc.
# export REPO="pixel-base-finetune-sent"
export REPO="pixel-base-finetune-d3tok-space-sent"
export SEQ_LEN=256
export BSZ=64
export GRAD_ACCUM=1
export LR=5e-05
export SEED=42
export NUM_STEPS=25000
export NUM_EPOCHS=40
export MODE="${MODE:-arabic-default}"  # default if not set

export RUN_NAME="$(basename ${MODEL})-${MODE}-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"
python scripts/training/run_readability.py \
  --model_name_or_path=${MODEL} \
  --dataset_name="CAMeL-Lab/BAREC-Shared-Task-2025-sent" \
  --processing_config_name=${MODE} \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --do_predict \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=${NUM_EPOCHS} \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=100 \
  --run_name="${RUN_NAME}" \
  --output_dir="runs/${RUN_NAME}" \
  --overwrite_cache \
  --text_renderer_name_or_path="configs/renderers/noto_renderer" \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=500 \
  --save_strategy=steps \
  --save_steps=500 \
  --save_total_limit=3 \
  --report_to=wandb \
  --log_predictions \
  --metric_for_best_model="eval_QWK" \
  --fallback_fonts_dir=data/fallback_fonts \
  --seed=${SEED} \
  --overwrite_output_dir \
  --early_stopping=False \
  --load_best_model_at_end=True \
#   --push_to_hub \
#   --early_stopping_patience=20 \