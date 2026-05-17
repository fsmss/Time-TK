#!/bin/bash
set -e
# Auto-generated script for Weather
export CUDA_VISIBLE_DEVICES=0

root_path="../../data/weather/"
data_path="weather.csv"
data_type="custom"
enc_in=21
c_out=21

echo "Running Weather Pred_Len=96 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_Weather_H96" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 1 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --d_model 64 \
  --dropout 0.2 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1

echo "Running Weather Pred_Len=192 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_Weather_H192" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 4 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --d_model 64 \
  --dropout 0.5 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1

echo "Running Weather Pred_Len=336 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_Weather_H336" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 1 \
  --batch_size 128 \
  --learning_rate 0.001 \
  --d_model 128 \
  --dropout 0.1 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1

echo "Running Weather Pred_Len=720 ..."
python -u ../../run.py \
  --is_training 1 \
  --root_path "$root_path" \
  --data_path "$data_path" \
  --model_id "best_Weather_H720" \
  --model "TimeTK" \
  --data "$data_type" \
  --features "M" \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in "$enc_in" \
  --dec_in "$enc_in" \
  --c_out "$c_out" \
  --e_layers 1 \
  --batch_size 64 \
  --learning_rate 0.0002 \
  --d_model 512 \
  --dropout 0.5 \
  --lradj "type1" \
  --train_epochs 30 \
  --itr 1
