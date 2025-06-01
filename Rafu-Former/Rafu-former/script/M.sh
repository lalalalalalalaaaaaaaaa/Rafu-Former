export CUDA_VISIBLE_DEVICES=8

for preLen in 96 192 336 720
do

python -u main.py \
  --is_training True \
  --root_path datasets/electricity \
  --data_path electricity.csv \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len $preLen \
  --learning_rate 1e-3
  

done