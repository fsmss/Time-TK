#!/bin/bash
set -e
# Auto-generated script for ETTh1
export CUDA_VISIBLE_DEVICES=0

root_path="../../data/ETT-small/"
data_path="ETTh1.csv"
data_type="ETTh1"
enc_in=7
c_out=7

echo "Running ETTh1 Pred_Len=96 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_ETTh1_H96" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 4 \
  --batch_size 128 \
  --learning_rate 0.002 \
  --d_model 256 \
  --dropout 0.3 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1

echo "Running ETTh1 Pred_Len=192 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_ETTh1_H192" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 3 \
  --batch_size 64 \
  --learning_rate 0.0002 \
  --d_model 512 \
  --dropout 0.5 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1

echo "Running ETTh1 Pred_Len=336 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_ETTh1_H336" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 1 \
  --batch_size 32 \
  --learning_rate 0.001 \
  --d_model 256 \
  --dropout 0.5 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1

echo "Running ETTh1 Pred_Len=720 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_ETTh1_H720" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 1 \
  --batch_size 16 \
  --learning_rate 0.0002 \
  --d_model 64 \
  --dropout 0.1 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1
