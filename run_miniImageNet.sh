python train.py --arch DSN --dataset miniImageNet --gpu 3 --lr 0.1 --batch_size 128 --max_epoch 200 --decay_epoch 80 120 160 &&
python Inc_train.py --arch DSN --dataset miniImageNet --gpu 3 --max_epoch 50 --DS True --lr 0.1 --delay_estimation 2 --delay_testing 100 --r 0.10 --gamma 0.20 --optimizer part --batch_size 256 --sample_k 4 --newsample_num 4 --oldsample_num_min 4 --basesample_num_min 4 --top_k 4
