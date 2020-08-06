# Pneumonia Detector

Download and extract dataset
```bash
mkdir dataset
wget https://data.mendeley.com/datasets/rscbjbr9sj/2/files/f12eaf6d-6023-432f-acc9-80c9d7393433/ChestXRay2017.zip?dl=1 -O dataset/dataset.zip
unzip dataset/dataset.zip -d dataset/
rm -r dataset/__MACOSX
rm dataset/dataset.zip
rm dataset/chest_xray/test/.DS_Store
rm dataset/chest_xray/train/.DS_Store
```
