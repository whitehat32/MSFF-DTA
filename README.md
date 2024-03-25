# MSFF-DTA

## Description
The MSFF-DTA repository contains the implementation code for the paper "Prediction of Drug-Target Binding Affinity Based on Multi-Scale Feature Fusion". It provides comprehensive scripts for running and training models specifically designed for two datasets: Davis and Kiba.

## Getting Started
To initiate the MSFF-DTA project, adhere to the following guidelines:

## Prerequisites
Python: 3.8.0
Dependencies: Install all necessary dependencies by executing 
```bash  
conda create pli python=3.8
conda activate pli
```
```
bash  
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

```
bash  
pip install pytorch_lightning==2.1.2
conda install pyg=*=*cu* -c pyg
pip install torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install networkx
```

## Installing and Running
### Clone the Repository:
```
bash
git clone https://github.com/whitehat32/MSFF-DTA.git
```

### Navigate to the Project Directory:
```
bash
cd msff-dta
```
## To execute the models on the respective datasets, use the designated commands:
### Davis Dataset
#### Protein Cold Start:
```
python main.py --root_data_path data/Davis/prot_cold_start
```
#### Drug Cold Start:
```
python main.py --root_data_path data/Davis/drug_cold_start
```
#### Protein and Drug Cold Start:
```
python main.py --root_data_path data/Davis/drug_and_prot_cold_start
```
### Kiba Dataset
#### Drug Cold Start:
```
python main.py --root_data_path data/Kiba/drug_cold_start
```
#### Protein and Drug Cold Start:
```
python main.py --root_data_path data/Kiba/prot_and_drug_cold_start
```

