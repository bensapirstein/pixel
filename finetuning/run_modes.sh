#!/bin/bash

modes=(
#   no-unicode-normalize
#   arabic-nonorm-diac
#   arabic-norm-dediac
#   buckwalter-nonorm-diac
#   arabic-dediac
#   arabic-norm
#   buckwalter-norm-dediac
#   buckwalter-default
#   hsb-nonorm-diac
#   hsb-norm-dediac
#   hsb-default
  morph-d3tok-default
  morph-d3tok-tatweel
  morph-d3tok-space
  morph-d3tok-tatweel2
  morph-d3tok-tatweel3
  arabic-default
)

for mode in "${modes[@]}"; do
  echo "=== Running MODE: $mode ==="
  MODE="$mode" bash finetuning/finetune_barec.sh
done
