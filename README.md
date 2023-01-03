# Using Gradual Extrapolation for Automating Latent Feature Detection

Repository with code used for semi-automated latent features detection.



## Steps

0. Download imagenet_9 images or imagenet.
0. Traing chosen network using `im9_train.py`
0. Perform ROI detection using `roi.py`
0. Now You can use model-with-9-classes or model's Zoo one.
0. Perform calculation of Graudal Extrapolation inside ROIs using `GE_in_ROI_counter.py`
0. Count average per class using `maping-roi-average.py`

## Datasets
All dataset are stored in [GDrive](https://drive.google.com/drive/folders/1atlC696a0WEakXzSGFkcOtQE3S6yyIU6?usp=share_link)

It includes:
- ImageNet 9: ~500 images each class
- ImageNet: ~60 images each class


## See also
- Gradual Extrapolation paper in [IEEE Access](https://ieeexplore.ieee.org/document/9468713)
- My greatest project [TorchPRISM](https://github.com/szandala/TorchPRISM)

## Citation

**TODO:** *after I publish this paper...*
