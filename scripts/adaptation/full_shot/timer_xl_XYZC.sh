export CUDA_VISIBLE_DEVICES=0
model_name=timer_xl
token_num=30
token_len=96
seq_len=$[$token_num*$token_len]

#/data/qiuyunzhong/Training-LTSM/dataset/XYZC_electricity_price_forecast/预测负荷预测备用日前价格合并数据.csv

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path ./dataset/XYZC/ \
  --data_path XYZC_forecast.csv \
  --model_id XYZC_full_shot \
  --model $model_name \
  --data UnivariateDatasetBenchmark  \
  --seq_len $seq_len \
  --input_token_len $token_len \
  --output_token_len $token_len \
  --test_seq_len $seq_len \
  --test_pred_len 96 \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --batch_size 512 \
  --learning_rate 5e-6 \
  --train_epochs 10 \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --use_norm \
  --adaptation \
  --covariate \
  --visualize \
  --pretrain_model_path checkpoints/timerxl_checkpoint.pth
#  --pretrain_model_path checkpoints/timer_xl/checkpoint.pth