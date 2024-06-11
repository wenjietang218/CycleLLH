if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

seq_len=384
model_name=CycleLLH

root_path_name=./dataset/
data_path_name=NetTm.csv
model_id_name=NetTm
data_name=NetTm

random_seed=2021

for data_path_name in "1%/NetTm.csv" "5%/NetTm.csv" "10%/NetTm.csv"
do
  for pred_len in  96 168 192 240 336 720
  do
      python -u run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features S \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 1 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 256 \
        --weight_1 1\
        --weight_2 16\
        --C 48\
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 60\
        --patience 20\
        --lradj 'TST'\
        --pct_start 0.4\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/univariate/$model_name'_fS_'$model_id_name'_'$seq_len'_'$pred_len.log
  done
done
