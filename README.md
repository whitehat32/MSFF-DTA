# MSFF-DTA

## Description
The MSFF-DTA repository contains the implementation code for the paper "Prediction of Drug-Target Binding Affinity Based on Multi-Scale Feature Fusion". It provides comprehensive scripts for running and training models specifically designed for two datasets: Davis and Kiba.

## Getting Started
To initiate the MSFF-DTA project, adhere to the following guidelines:

## Prerequisites
Python: Ensure that Python is installed on your machine.
Dependencies: Install all necessary dependencies by executing 
pip install -r requirements.txt within the project's root directory(To do).

## Installing and Running
### Clone the Repository:
bash
git clone [https://github.com/whitehat32/MSFF-DTA.git](https://github.com/whitehat32/MSFF-DTA.git)


### Navigate to the Project Directory:
```bash
cd msff-dta

## To execute the models on the respective datasets, use the designated commands:
### Davis Dataset
#### Protein Cold Start:
```bash
python main.py --root_data_path data/Davis/prot_cold_start
#### Drug Cold Start:
```bash
python main.py --root_data_path data/Davis/drug_cold_start
#### Protein and Drug Cold Start:
```bash
python main.py --root_data_path data/Davis/prot_and_drug_cold_start

### Kiba Dataset
#### Drug Cold Start:
```bash
python main.py --root_data_path data/Kiba/drug_cold_start
#### Protein and Drug Cold Start:
```bash
python main.py --root_data_path data/Kiba/prot_and_drug_cold_start


