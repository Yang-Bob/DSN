# DSN
Dynamic Support Network for Few-shot Class-Incremental Learning

## Overview
- `trian.py' is the code for base training (0-th session);
- `Inc_train.py' is the code for incremental training;
- `models/` contains the implementation of the DSN(`DSN.py`) and the backbone network;
- `data/` contains the dataloader and the dataset files;
- `data_list/` contains the list of data;
- `config/' contains the setting file;
- `utils/` contains the log and restore files;


## Dependencies
python == 3.7,
pytorch1.0,

torchvision,
pillow,
opencv-python,
pandas,
matplotlib,
scikit-image
