# Pneumonia Detector

## Setup
```sh
git clone git@github.com:rafaelglikis/pneumonia-detector.git
cd pneumonia-detector
pip install -r requirements.txt
```

### Download and extract dataset
```sh
mkdir dataset
wget https://data.mendeley.com/datasets/rscbjbr9sj/2/files/f12eaf6d-6023-432f-acc9-80c9d7393433/ChestXRay2017.zip?dl=1 -O dataset/dataset.zip
unzip dataset/dataset.zip -d dataset/
# Also remove unnecessary files
rm -r dataset/__MACOSX
rm dataset/dataset.zip
rm dataset/chest_xray/test/.DS_Store
rm dataset/chest_xray/train/.DS_Store
```

## Usage
```sh
usage: pneumdet.py [-h]
                   [--train {inception,vgg16,resnet50,densenet121,xception,mobilenet}]
                   [--evaluate EVALUATE [EVALUATE ...]]
                   [--ensemble {evaluate}]

Detect pneumonia from chest x rays.

optional arguments:
  -h, --help            show this help message and exit
  --train {inception,vgg16,resnet50,densenet121,xception,mobilenet}
                        Train a model.
  --evaluate EVALUATE [EVALUATE ...]
                        Evaluate trained model.
  --ensemble {evaluate}
                        Use ensemble.
```

### Examples

Train inception model
```sh
 python pneumdet.py --train inception
```
Evaluate ensemble/inception_v3_transfer_20200725-123618 model
```sh
python pneumdet.py --evaluate ensemble/inception_v3_transfer_20200725-123618 
```

Evaluate all models in the ensemble directory
```sh
 python pneumdet.py --evaluate ensemble/*
```

Evaluate the ensemble created
```sh
 python pneumdet.py --ensemble evaluate 
```