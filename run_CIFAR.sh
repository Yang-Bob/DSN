python train.py --arch DSN --dataset CIFAR100 --gpu 2 --lr 0.1 --batch_size 128 --max_epoch 200 --decay_epoch 80 120 160 &&
python Inc_train.py --arch DSN --dataset CIFAR100 --gpu 2 --max_epoch 70 --DS True --lr 0.1 --delay_estimation 2 --delay_testing 100 --r 0.10 --gamma 0.20 --optimizer part --batch_size 256 --sample_k 5 --newsample_num 5 --oldsample_num_min 5 --basesample_num_min 5 --top_k 5
