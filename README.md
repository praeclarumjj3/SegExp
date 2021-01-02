# PPM vs ASPP

**The base code for this experiment was taken from the official [open-mmlab repos](https://github.com/open-mmlab)**

This repo contains the code for comparing the performance among a PSPHead (PPM), an ASPPHead (ASPP), and a Depth-wiseSeparableASPPHead (Depth-wise Separable ASPP) on the semantic segmentation task. For easy comparison, I use a **PSPNet Encoder** with these different **decoder heads**.

## Environment Setup

- Create a conda virtual environment and activate it.

```
conda create -n mm
conda activate mm
```

- Install PyTorch and torchvision:
```
conda install pytorch=1.6.0 torchvision cudatoolkit=10.2 -c pytorch
```

- Install mmcv requirements for Linux:

```
cd mmcv
pip install -e .
```

- Install MMSegmentation requrements

```
cd mmsegmentation
pip install -e .
```

## Cityscapes Dataset Setup

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.

```
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
```

- Install [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)

```
pip install cityscapesscripts
```

- Download the zip files

```
# gtFine
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
```

```
# leftImgBit
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

- Unzip files into the `mmsegmentation/data/cityscapes/`:

```
python unzip.py
```

- Prepare the dataset:

```
cd mmsegmentation/

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8 
```

## Training

### PPM Module (used in PSPNet)

```
cd mmsegmentation/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/pspnet/pspnet_r50-d8_test_cityscapes.py 8
```

### ASPP Module (used in Deeplab-v3)

```
cd mmsegmentation/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/pspnet/pspnet_r50-d8_test_cityscapes_aspp.py 8
```

### Depth-wise ASPP Module (used in Deeplab-v3+)

```
cd mmsegmentation/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/pspnet/pspnet_r50-d8_test_cityscapes_d_aspp.py 8
```

## Experiments

- I conduct experiments on the [Cityscapes Fine Annotations](https://www.cityscapes-dataset.com/examples/#fine-annotations) dataset.

- For faster training and testing, I passed cropped images of size 300x300 to the models.

- ResNet-50 is used as a backbone network.

- The models are trained for `10000 Epochs` on 2975 images, with periodical testing every `1000 Epochs` on 500 images.

- Each Model takes around `5 hours` during training.

### Class-wise

| Class         | IoU (PPM)  | IoU (ASPP)  | IoU (Depth-wise ASPP)  | Acc (PPM)   | Acc (ASPP)  | Acc (Depth-wise ASPP)  |
----------------|------------|-------------|------------------------|-------------|-------------|------------------------|
| road          | 96.45      | 97.12       | 96.68                  | 97.57       | 98.25       | 97.74                  |
| sidewalk      | 78.2       | 80.15       | 78.83                  | 90.34       | 90.08       | 90.41                  |
| building      | 89.29      | 89.2        | 89.0                   | 94.47       | 93.02       | 93.81                  |
| wall          | 33.16      | 37.54       | 38.35                  | 41.4        | 48.04       | 51.38                  |
| fence         | 52.93      | 53.63       | 53.8                   | 67.58       | 69.93       | 70.2                   |
| pole          | 57.13      | 57.93       | 57.6                   | 68.19       | 69.87       | 67.51                  |
| traffic light | 63.92      | 61.78       | 61.45                  | 77.72       | 77.37       | 79.55                  |
| traffic sign  | 72.36      | 70.49       | 71.29                  | 82.57       | 83.31       | 81.67                  |
| vegetation    | 91.14      | 90.75       | 91.17                  | 96.29       | 96.66       | 96.05                  |
| terrain       | 60.41      | 59.19       | 55.26                  | 72.82       | 71.83       | 72.47                  |
| sky           | 92.5       | 91.0        | 91.49                  | 97.34       | 97.81       | 96.2                   |
| person        | 77.19      | 77.07       | 77.9                   | 88.5        | 89.38       | 89.23                  |
| rider         | 51.03      | 50.96       | 51.28                  | 65.08       | 66.43       | 65.38                  |
| car           | 92.3       | 92.68       | 92.37                  | 97.2        | 97.22       | 97.09                  |
| truck         | 46.44      | 42.97       | 38.32                  | 56.71       | 58.99       | 44.6                   |
| bus           | 61.7       | 65.2        | 56.5                   | 71.36       | 77.89       | 76.43                  |
| train         | 37.39      | 39.27       | 25.71                  | 76.65       | 78.06       | 74.93                  |
| motorcycle    | 55.29      | 56.21       | 46.03                  | 68.57       | 75.01       | 56.09                  |
| bicycle       | 72.83      | 72.62       | 71.63                  | 88.13       | 86.35       | 87.83                  |

### Average over the classes

| Scope  | mIoU (PPM)  | mIoU (ASPP) | mIoU (Depth-wise ASPP)| mAcc (PPM) | mAcc (ASPP)| mAcc (Depth-wise ASPP)|
|--------|-------------|-------------|-----------------------|------------|------------|-----------------------|
| global | 67.46       | 67.67       | 65.51                 | 78.87      | 80.29      | 78.35                 |
