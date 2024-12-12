# biom-benchmark

## Setting up

### Model Weights

Model configurations and weights should be downloaded and placed in the `model/pretrained/[Model Name]` directory.

**RNAFM**:
The pretrained model can be downloaded from the following link:
`https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth`

**RNABERT** and **RNAMSM**:
Training data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1flh2rXiMKIreHE2l4sbjMmwAqfURj4vv?usp=sharing). This link is referenced from [RNAErnie](https://github.com/CatIIIIIIII/RNAErnie_baselines/tree/main).


**RNAErnie**:
The original model used in the RNAErnie publication is based on the PaddlePaddle framework, which is incompatible with other models. We used the PyTorch version of the model released by the authors. (https://huggingface.co/LLM-EDA/RNAErnie/tree/main)

**SpliceBERT**:
The model weights are available on [Zenodo](https://doi.org/10.5281/zenodo.7995778).

**DNABERT**:
DNABERT provides a series of models using different k-mer settings. We use the most popular version, [DNA_bert_3](https://huggingface.co/zhihan1996/DNA_bert_3).

**DNABERT2**:
The models are available at this [link](https://huggingface.co/zhihan1996/DNABERT-2-117M).

**Nucleotide Transformer**:
The authors provided a series of models. We use the best version reported in the article, [nucleotide-transformer-v2-500m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species).

### Hardware Requirements
All analyses were conducted on a cluster node equipped with 32 CPU cores and 4 Nvidia Tesla A100 40G GPUs. At least one GPU is necessary for executing a single task.

### Software environment

A Linux system is required.

We recommend using conda and pip to manage the software environments:

```bash
conda env create -f environment_1019.yml
```

## Run pipelines

The datasets required for the analyses can be sourced from the Data Availability sections of our previous publications. For convenience, we have also uploaded essential data files to [Google Drive](https://drive.google.com/file/d/18DccTVbd62PdOA8NLh55rdjTSO1faKsI/view?usp=sharing), which you can download and place in the `./dataset` folder.
All data files should be placed in a subfolder of `./dataset` folder. e.g. `./dataset/m6a_data/`

### nRC prediction

An example script for training and testing the models is provided at `scripts/cls/seq_cls_nRC_1e-4.sh`

### m6A prediction

An example script for training and testing the models is provided at  `scripts/m6A/m6a_miCLIP_101_1e-4.sh`

### Splicing prediction

Due to its size (over 100GB), the splicing dataset cannot be directly uploaded. Please refer to the Methods section for instructions on how to generate the dataset. We do provide intermediate files through Google Drive to facilitate this process.

1. Run `scripts/makedata_splice.sh` to create datasets. 
2. An example of training and testing all models is available at `scripts/splice/splice_3.sh`

The entire process, depending on the GPU, may require several days to complete.

### MRL prediction

An example of training and testing all models is available at `scripts/mrl/mrl_1e-3.sh`

### Gather results

1. We extract the test results from the textual output of the program and compile them into a table. 

- For clear output information, we recommend separating standard output from error messages when executing the scripts, for example:
```bash
bash scripts/run_splice_train_test_53.sh > output.txt 2>error_output.txt
```

- Our test was performed on a slurm cluster, thus the outputs could be split easily. An example is available at `scripts/cls/HPC_run_seq_cls_1.sh`


2. Then, to convert the output into a table, run the `parse_output.py` script located in the analyzer folder:

```bash
cd analyzer
python parse_output.py -i analyzer/nRC_1_4.out
```

The generated table will serve as the source data for subsequent plotting. An example of this process can be found in the Jupyter Notebook located at 'analyzer/analyze.ipynb'.


## Code Description

The source code is structured into several folders:

- dataset: Contains scripts and utilities for creating and loading datasets.
- evaluator: Houses the functionality for loading models, conducting training, and performing evaluations.
- logs: Designated output directory for all log files.
- model: Includes the definitions and implementations of the various models used.
- scripts: Provides reference scripts to guide the execution of the project.

The main entry points of the program are the following Python scripts: `seq_cls.py`, `m6a_cls.py`, `splice_cls.py` and `mrl_pred.py`. These scripts can be customized to accommodate specific testing requirements.