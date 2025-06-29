# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export MODEL="bensapir/pixel-barec-pretrain" # also works with "bert-base-cased", "roberta-base", etc.
export SEQ_LEN=256
export BSZ=64
export GRAD_ACCUM=1
export LR=5e-5
export SEED=42
export NUM_STEPS=20000

export RUN_NAME="$(basename ${MODEL})-${SEQ_LEN}-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_readability.py \
  --model_name_or_path=${MODEL} \
  --dataset_name=CAMeL-Lab/BAREC-Shared-Task-2025-sent \
  --remove_unused_columns=False \
  --do_train \
  --do_eval \
  --do_predict \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --early_stopping \
  --early_stopping_patience=5 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=100 \
  --run_name="arabic-readability-strict" \
  --output_dir="runs/arabic-readability-strict" \
  --overwrite_output_dir \
  --overwrite_cache \
  --text_renderer_name_or_path="configs/renderers/noto_renderer" \
  --logging_strategy=steps \
  --logging_steps=1000 \
  --evaluation_strategy=steps \
  --eval_steps=2000 \
  --save_strategy=steps \
  --save_steps=2000 \
  --save_total_limit=5 \
  --report_to=wandb \
  --log_predictions \
  --load_best_model_at_end=True \
  --metric_for_best_model=eval_accuracy \
  --fallback_fonts_dir=data/fallback_fonts \
  --seed=${SEED}