# Optional wandb environment vars
export WANDB_PROJECT="pixel-experiments"

# Settings
export DATA_DIR="Team-PIXEL/rendered-bookcorpus,Team-PIXEL/rendered-wikipedia-english"
export FALLBACK_FONTS_DIR="data/fallback_fonts"  # let's say this is where we downloaded the fonts to
export MODEL="Team-PIXEL/pixel-base" # also works with "bert-base-cased", "roberta-base", etc.
export BSZ=64
export GRAD_ACCUM=1
export LR=1.5e-4
export SEED=42
export NUM_STEPS=50000


python scripts/training/run_pretraining.py \
  --train_dataset_names="Team-PIXEL/rendered-bookcorpus,Team-PIXEL/rendered-wikipedia-english" \
  --validation_dataset_name="plip/wiki_dev" \
  --dataset_cache="data/rendered-bookcorpus,data/rendered-wikipedia-en" \
  --train_splits="train,train" \
  --validation_split="en" \
  --label_names="pixel_values" \
  --do_train \
  --do_eval \
  --max_steps=50000 \
  --base_learning_rate=1.5e-4 \
  --lr_scheduler_type="cosine" \
  --weight_decay=0.05 \
  --num_train_epochs=10 \
  --warmup_ratio=0.5 \
  --per_device_train_batch_size=64 \
  --gradient_accumulation_steps=1 \
  --logging_strategy="steps" \
  --logging_steps=1000 \
  --evaluation_strategy="steps" \
  --eval_steps=5000 \
  --save_strategy="steps" \
  --save_steps=10000 \
  --seed=42 \
  --use_auth_token="<redacted_token_with_read_access>" \
  --remove_unused_columns=False \
  --streaming=False \
  --report_to=wandb \
  --push_to_hub \
  --hub_model_id="Team-PIXEL/pixel-base" \
  --hub_strategy="checkpoint" \
  --hub_token="<redacted_token_with_write_access>" \
  --hub_private_repo=True \
  --config_name="Team-PIXEL/pixel-base" \
  --text_renderer_name_or_path="configs/renderers/noto_renderer" \
  --mask_ratio=0.25 \
  --span_masking \
  --masking_max_span_length \
  --masking_cumulative_span_weights="0.2,0.4,0.6,0.8,0.9,1" \
  --dropout_prob=0.1