# biom-benchmark

## Setup

### Model Weights

Download model configurations and weights, and place them in `model/pretrained/[Model Name]`.

**RNAFM**:  
Download the pretrained model from:  
`https://proj.cse.cuhk.edu.hk/rnafm/api/download?filename=RNA-FM_pretrained.pth`

**RNABERT** and **RNAMSM**:  
Download weights from [Link](https://drive.google.com/file/d/10gJBl24OGQ_aZfxtj09dik0rrM4dOk_R/view?usp=sharing) and [Link](https://drive.google.com/file/d/1-Gl9LGjR_dmDfOARrIVvuOmb7w_tJvGA/view?usp=sharing). Referenced from [RNAErnie](https://github.com/CatIIIIIIII/RNAErnie_baselines/tree/main).

**RNAErnie**:  
We use the PyTorch version of the model provided by the authors:  
(https://huggingface.co/LLM-EDA/RNAErnie/tree/main)

**SpliceBERT**:  
Model weights are available on [Zenodo](https://doi.org/10.5281/zenodo.7995778).

**DNABERT**:  
We use the popular [DNA_bert_3](https://huggingface.co/zhihan1996/DNA_bert_3).

**DNABERT2**:  
Available at [link](https://huggingface.co/zhihan1996/DNABERT-2-117M).

**GENA-LM**:  
Available at [link](https://github.com/AIRI-Institute/GENA_LM).

**UTRLM**:  
The model is available at this [link](https://codeocean.com/capsule/4214075/tree/v1).

**Nucleotide Transformer**:  
We use the best-reported version: [nucleotide-transformer-v2-500m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species).

We are currently in the process of packaging and uploading all model weights to Google Drive for your convenience. The upload will take some additional time to complete.

### Hardware Requirements

All analyses were conducted on a cluster node with 32 CPU cores and 4 Nvidia Tesla A100 40G GPUs. At least one GPU is required for a single task.

### Software Environment

A Linux system is required. Use conda and pip to manage dependencies:

```bash
conda env create -f environment_1019.yml
```

## Running Pipelines

### Prepare Datasets

Datasets can be sourced from the manuscript's Data Availability sections. We are preparing a repository to release the code for building final datasets.  

Essential data files are also available on [Google Drive](https://drive.google.com/drive/folders/1bNvG5JRnUmADC1PXzeCQqkrUBtYKKaxu?usp=sharing). Download and place them in `./dataset`.  
- Datasets for ncRNA, m6a, and MRL are directly available.  
- For splicing prediction, run `scripts/makedata_splice.sh` to generate the final dataset (~50GB).

### nRC Prediction

Example script: `scripts/cls/HPC_run_1.sh`.

### m6A Prediction

Example script: `scripts/m6A/HPC_run_1.sh`.

### Splicing Prediction

1. Run `scripts/makedata_splice.sh` to create datasets.  
2. Example script: `scripts/splice/HPC_run_1.sh`.

### MRL Prediction

Example script: `scripts/mrl/HPC_run_1.sh`.

### Gather Results

1. Extract test results from program output and compile them into a table.  
   - Separate stdout and stderr for clarity:  
   ```bash
   bash scripts/run_splice_train_test_53.sh > output.txt 2>error_output.txt
   ```
   - On Slurm clusters, stdout and stderr are automatically separated.

2. Convert output to a table using `parse_output.py` in the `analyzer` folder:  
   ```bash
   cd analyzer
   python parse_output.py -i tables/m6a101_4_0.1.txt
   ```
   Example output: `analyzer/tables/m6a101_4_0.1_collected_data.csv`.

The generated table serves as input for plotting. See `analyzer/analyze.ipynb` for an example.

## Code Structure

- `dataset`: Scripts and utilities for dataset creation and loading.  
- `evaluator`: Functions for model loading, training, and evaluation.  
- `logs`: Directory for log files.  
- `model`: Model definitions and implementations.  
- `scripts`: Reference scripts for running the project.

Main entry points: `seq_cls.py`, `m6a_cls.py`, `splice_cls.py`, and `mrl_pred.py`. Customize these scripts for specific tests.