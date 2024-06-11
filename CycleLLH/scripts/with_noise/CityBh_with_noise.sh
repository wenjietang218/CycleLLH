if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

seq_len=336
model_name=CycleLLH

root_path_name=./dataset/
model_id_name=CityBh
data_name=CityBh

random_seed=2021

for data_path_name in "1%/CityBh.csv" "5%/CityBh.csv" "10%/CityBh.csv"
do
  for pred_len in 96 168 192 240 336 720
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
        --n_heads 4 \
        --d_model 16 \
        --d_ff 128 \
        --weight_1 6\
        --weight_2 9\
        --C 24\
        --dropout 0.3\
        --fc_dropout 0.3\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 60\
        --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/univariate/$model_name'_fS_'$model_id_name'_'$seq_len'_'$pred_len.log
  done
done
