# PyTorch-YOLOv3
Minimal implementation of YOLOv3 in PyTorch.

Source:
https://github.com/eriklindernoren/PyTorch-YOLOv3

## Credit
```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

## Get files
Use get_data_files.py to get a list of the images that will be saved into data/"**DATASET**"

> python get_data_files.py -n **DATASET_NAME(visdrone)** -t **TYPE(train/val)** -p **IMAGE_PATH(../visdrone_data/ETC)** -d **DIMENSIONS(Boolean)**

Note: do not add "-d"



## Train

Make sure /config/visdrone.data holds path to training list txt created using get_data_files.py

Note valid is not necessary for training.

> python train.py
