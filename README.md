# PTB-XL+, a comprehensive electrocardiographic feature dataset

This is the official code repository accompanying the new dataset **PTB-XL+** hosted at physionet.com

## Usage information
### Preparation
1. Install dependencies from `environment.yml` by running `conda env create -f environment.yml`, activate the environment via `conda activate ptbfeat`

2. Follow the instructions in `data_preprocessing.ipynb` on how to download and preprocess PTB-XL. Download the features files `12sl_features.csv`,  `ecgdeli_features.csv`,  `ge12sl_glasgow_kit_ECGFeaturesMapToOMOP_draft1.xlsx`  `unig_features.csv` and put them in `./data/features`. Further get the additional label informations PTB-XL+ contains about PTB-XL: `12sl_statements.csv` and `ptb_statements.csv` and put them in `./data/statements`. In the following, we assume for definiteness that the preprocessed PTB-XL can be found at  `./data/ptb_xl_fs100`, the features in `./data/features` and the statements in `./data/statements`.

### Code Structure 
The most functionality is provided in the files `code/run_feature_benchmark.py` and `code/run_raw_benchmark.py`.

### Feature Experiments 
`code/run_feature_benchmark.py` was used to perform the feature experiments, run 

`python code/run_feature_benchmark.py --dataset unig --modelname rf`

to train a Random Forest on the Uni-G Features. Try a different dataset by changing the dataset parameter after `--dataset`. Available datasets are [`unig`, `ecgdeli`, `12sl`].

The results of the runs will be written to `./output`


### Training a ResNet on PTB-XL labels
Train a xresnet1d50 on the normal labels of PTB-XL (at the most finegrained level) by running:

`python code/run_raw_benchmark.py --dataset data/ptb_xl_fs100  --label_class label_all`

Logs and trained models are saved in `./logs`. Hence you can monitor training by `tensorboard --logdir=./logs --port 6006`. You can use a different label_set by varying `--label_class`
Available label sets include:

| Label set   |      Description      |  
|----------|:-------------:|
| `label_all` | original PTB-XL label set| 
| `label_all_12sl` |    the 12SL label set of PTB-XL   |   
| `label_all_12sl_ext_snomed` |  the 12SL label set of PTB-XL, mapped onto SNOMED labels | 
| `label_all_ptb_ext_snomed` |  the original label set of PTB-XL, mapped onto SNOMED labels  | 
| `label_all_12sl_ext_snomed_union` |  the original label set of PTB-XL, mapped onto SNOMED labels, but only considering labels that also occur in the 12SL SNOMED label set     |   
| `label_all_ptb_ext_snomed_union` | the 12SL label set of PTB-XL, mapped onto SNOMED labels, but only considering labels that also occur in the orignal PTB-XL SNOMED label set  |    

You can test a trained model by specifiying the location of the trained model (checkpoint) with the parameter `--checkpoint_path`

`python code/run_raw_benchmark.py --dataset data/ptb_xl_fs100  --label_class label_all --checkpoint_path=path/to/checkpoint --test_only`


