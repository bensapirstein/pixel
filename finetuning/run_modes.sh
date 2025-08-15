#!/bin/bash

modes=(
  hsb-nonorm-diac
  no-unicode-normalize
#   arabic-nonorm-diac
#   arabic-norm-dediac
#   arabic-default
#   buckwalter-nonorm-diac
#   buckwalter-norm-dediac
#   buckwalter-default
#   hsb-norm-dediac
#   hsb-default
)

for mode in "${modes[@]}"; do
  echo "=== Running MODE: $mode ==="
  MODE="$mode" bash finetuning/finetune_barec.sh
done
