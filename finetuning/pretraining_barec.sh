# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export DATA_DIR=""
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export MODEL="bensapir/pixel-barec-pretrain" # also works with "bert-base-cased", "roberta-base", etc.
export BSZ=16
export GRAD_ACCUM=1
export LR=1.5e-4
export SEED=42
export NUM_STEPS=200000

export RUN_NAME="pixel-barec-pretrain-$(basename ${MODEL})-${BSZ}-${GRAD_ACCUM}-${LR}-${NUM_STEPS}-${SEED}"

python scripts/training/run_pretraining.py \
  --model_name_or_path=${MODEL} \
  --train_dataset_names="bensapir/pixel-barec-pretrain" \
  --validation_dataset_name="bensapir/pixel-barec-pretrain" \
  --dataset_cache="data/pixel-barec-pretrain" \
  --train_splits="train" \
  --validation_split="validation" \
  --label_names="pixel_values" \
  --do_train \
  --do_eval \
  --max_steps=${NUM_STEPS} \
  --base_learning_rate=${LR} \
  --lr_scheduler_type="cosine" \
  --weight_decay=0.05 \
  --num_train_epochs=10 \
  --warmup_ratio=0.5 \
  --per_device_train_batch_size=${BSZ} \
  --gradient_accumulation_steps=1 \
  --logging_strategy="steps" \
  --logging_steps=10000 \
  --evaluation_strategy="steps" \
  --eval_steps=10000 \
  --save_strategy="steps" \
  --save_steps=10000 \
  --seed=42 \
  --remove_unused_columns=False \
  --streaming=False \
  --report_to=wandb \
  --push_to_hub \
  --text_renderer_name_or_path="configs/renderers/noto_renderer" \
  --mask_ratio=0.25 \
  --span_masking \
  --masking_max_span_length=6 \
  --masking_cumulative_span_weights="0.2,0.4,0.6,0.8,0.9,1" \
  --dropout_prob=0.1 \
  --output_dir="pixel-barec-pretrain"