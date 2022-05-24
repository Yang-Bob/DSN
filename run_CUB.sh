python train.py --arch DSN --dataset CUB200 --gpu 1 --lr 0.05 --batch_size 128 --max_epoch 100 --decay_epoch 10 60 80 &&
python Inc_train.py --arch DSN --dataset CUB200 --gpu 1 --max_epoch 35 --DS True  --lr 0.05 --delay_estimation 2 --delay_testing 100 --r 0.10 --gamma 0.20 --optimizer full --batch_size 128 --sample_k 4 --newsample_num 4 --oldsample_num_min 5 --basesample_num_min 5 --top_k 5
