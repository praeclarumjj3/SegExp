# PPM vs ASPP

**The base code for this experiment was taken from the official [open-mmlab repos](https://github.com/open-mmlab)**

This repo contains the code for comparing the performance between a PSPHead (PPM) and an ASPPHead (ASPP) on the semantic segmentation task. For easy comparison, I replaced the **PPM module in the PSPNet by the ASPP module**.

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

### PPM Module

```
cd mmsegmentation/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/pspnet/pspnet_r50-d8_test_cityscapes.py 8
```

### ASPP Module

```
cd mmsegmentation/
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh configs/pspnet/pspnet_r50-d8_test_cityscapes_aspp.py 8
```

## Experiments

- I conduct experiments on the [Cityscapes Fine Annotations](https://www.cityscapes-dataset.com/examples/#fine-annotations) dataset.

- For faster training and testing, I passed cropped images of size 300x300 to the models.

- ResNet-50 is used as a backbone network.

- The models are trained for `10000 Epochs` on 2975 images, with periodical testing every `1000 Epochs` on 500 images.

### Class-wise

| Class         | IoU (PPM)  | IoU (ASPP)  | Acc (PPM)   | Acc (ASPP)  |
----------------|------------|-------------|-------------|-------------|
| road          | 96.45      | 96.45       | 97.57       | 97.57       |
| sidewalk      | 78.2       | 78.2        | 90.34       | 90.34       |
| building      | 89.29      | 89.29       | 94.47       | 94.47       |
| wall          | 33.16      | 33.16       | 41.4        | 41.4        |
| fence         | 52.93      | 52.93       | 67.58       | 67.58       |
| pole          | 57.13      | 57.13       | 68.19       | 68.19       |
| traffic light | 63.92      | 63.92       | 77.72       | 77.72       |
| traffic sign  | 72.36      | 72.36       | 82.57       | 82.57       |
| vegetation    | 91.14      | 91.14       | 96.29       | 96.29       |
| terrain       | 60.41      | 60.41       | 72.82       | 72.82       |
| sky           | 92.5       | 92.5        | 97.34       | 97.34       |
| person        | 77.19      | 77.19       | 88.5        | 88.5        |
| rider         | 51.03      | 51.03       | 65.08       | 65.08       |
| car           | 92.3       | 92.3        | 97.2        | 97.2        |
| truck         | 46.44      | 46.44       | 56.71       | 56.71       |
| bus           | 61.7       | 61.7        | 71.36       | 71.36       |
| train         | 37.39      | 37.39       | 76.65       | 76.65       |
| motorcycle    | 55.29      | 55.29       | 68.57       | 68.57       |
| bicycle       | 72.83      | 72.83       | 88.13       | 88.13       |
|---------------|------------|-------------|-------------|-------------|

### Average over the classes

| Scope  | mIoU (PPM)  | mIoU (ASPP) | mAcc (PPM) | mAcc (ASPP)| aAcc (PPM) | aAcc (ASPP)|
|--------|-------------|-------------|------------|------------|------------|------------|
| global | 67.46       | 67.46       | 78.87      | 78.87      | 94.28      | 94.28      |
|--------|-------------|-------------|------------|------------|------------|------------|
