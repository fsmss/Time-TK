#!/bin/bash
set -e
# Auto-generated script for PEMS04
export CUDA_VISIBLE_DEVICES=0

root_path="../../data/PEMS/"
data_path="PEMS04.npz"
data_type="PEMS"
enc_in=307
c_out=307

echo "Running PEMS04 Pred_Len=12 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_PEMS04_H12" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 12 \
  --use_revin 0 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 1 \
  --batch_size 16 \
  --learning_rate 0.005 \
  --d_model 128 \
  --dropout 0.1 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1

echo "Running PEMS04 Pred_Len=24 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_PEMS04_H24" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 24 \
  --use_revin 0 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 1 \
  --batch_size 16 \
  --learning_rate 0.002 \
  --d_model 512 \
  --dropout 0.1 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1

echo "Running PEMS04 Pred_Len=48 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_PEMS04_H48" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 48 \
  --use_revin 0 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 4 \
  --batch_size 64 \
  --learning_rate 0.005 \
  --d_model 512 \
  --dropout 0.2 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1

echo "Running PEMS04 Pred_Len=96 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_PEMS04_H96" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 96 \
  --use_revin 0 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 2 \
  --batch_size 32 \
  --learning_rate 0.002 \
  --d_model 512 \
  --dropout 0.1 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1
