python train.py --arch DSN --dataset CUB200 --gpu 0 --lr 0.01 --batch_size 128 --max_epoch 200 --decay_epoch 80 120 160 &&
python Inc_train.py --arch DSN --gpu 0 --max_epoch 50 --DS True --dataset CUB200 --lr 0.02 --delay_estimation 2 --delay_testing 100 --r 0.10 --gamma 0.20 --batch_size 256 --sample_k 5 --newsample_num 5 --oldsample_num_min 3 --top_k 5
