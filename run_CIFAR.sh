python train.py --arch DSN --dataset CIFAR100 --gpu 1 --lr 0.1 --batch_size 128 --max_epoch 200 --decay_epoch 80 120 160 &&
python Inc_train.py --arch DSN --gpu 1 --max_epoch 50 --DS True --dataset CIFAR100 --lr 0.02 --delay_estimation 2 --delay_testing 100 --r 0.10 --gamma 0.20 --batch_size 256 --sample_k 5 --newsample_num 5 --oldsample_num_min 3 --top_k 5
