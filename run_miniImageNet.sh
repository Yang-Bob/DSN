python train.py --arch DSN --dataset miniImageNet --gpu 2 --lr 0.1 --batch_size 128 --max_epoch 200 --decay_epoch 80 120 160 &&
python Inc_train.py --arch DSN --gpu 2 --max_epoch 50 --DS True --dataset miniImageNet --lr 0.02 --delay_estimation 5 --delay_testing 100 --r 0.10 --gamma 0.20 --batch_size 256 --sample_k 3 --newsample_num 3 --oldsample_num_min 3 --top_k 1
