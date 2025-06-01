#!/bin/bash
export CUDA_VISIBLE_DEVICES=8

for preLen in 96 192; do
  # 1. PreTrain
  echo "========== Start PreTrain pred_len=$preLen =========="
  python pretrain.py \
    --root_path datasets/electricity \
    --data_path electricity.csv \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len $preLen \
    --pretrain_epochs 20 \
    --pretrain_lr 1e-4 \
    --device cuda:0
  

  # 2. Formal Training (Predict)
  echo "========== Start DownStream Task (Prediction Task) pred_len=$preLen =========="
  python -u main.py \
    --is_training True \
    --use_pretrain \
    --root_path datasets/electricity \
    --data_path electricity.csv \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 96 \
    --pred_len $preLen \
    --learning_rate 1e-3

done