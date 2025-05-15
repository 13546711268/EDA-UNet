# EDA-UNet  
Lightweight Dual-Attention Network for Remote-Sensing Segmentation

---

## 1  Installation

```bash
# create environment
conda create -n edaunet python=3.8 -y
conda activate edaunet

# install PyTorch (CUDA 11.3) & others
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
````

---

## 2  Folder Structure

```none
EDA-UNet
├── network
├── config
├── tools
├── data
│   ├── cloud
│   │   ├── ann_dir
│   │   │   ├── train  val  test
│   │   ├── img_dir
│   │   │   ├── train  val  test
│   └── vaihingen
│       ├── train_images  train_masks
│       ├── test_images   test_masks   
│       ├── train  test   
```

---

## 3  Dataset Preparation

| Dataset                  | Link                                                                                                                                           |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| ISPRS **Vaihingen**      | [https://www.isprs.org/education/benchmarks/UrbanSemLab/vaihingen.aspx](https://www.isprs.org/education/benchmarks/UrbanSemLab/vaihingen.aspx) |
| **Cloud** (fine-grained) | [https://huggingface.co/datasets/XavierJiezou/ktda-datasets](https://huggingface.co/datasets/XavierJiezou/ktda-datasets)                       |

---

## 4  Training

```bash
# Vaihingen
python train.py -c config/vaihingen/edaunet.py

# Cloud
python train.py -c config/cloud/edaunet.py
```
