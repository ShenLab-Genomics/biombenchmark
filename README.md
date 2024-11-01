# biom-benchmark

## Run pipelines

#### nRC prediction

To train and test all supported models, run `scripts/run_seq_cls_test.sh`

#### m6A prediction

To train and test all supported models, execute `scripts/run_m6a_test_m6A_101.sh` and `scripts/run_m6a_test_m6A_512.sh`.

#### Splicing prediction

1. Run `scripts/run_splicing_makedata.sh` to create datasets. 
2. To train and test all supported models, execute `scripts/run_splice_train_test_3.sh`, `scripts/run_splice_train_test_15.sh`, and `scripts/run_splice_train_test_53.sh`.

Depending on the GPU used, the complete training process may take several days.

#### MRL prediction

To train and test all supported models, run `scripts/run_mrl_test.sh`.

## Gather results

We extract the test results from the textual output of the program and compile them into a table. For clear output information, we recommend separating standard output from error messages when executing the scripts, for example:

```bash
bash scripts/run_splice_train_test_53.sh > output.txt 2>error_output.txt
```

Then, to convert the output into a table, run the `parse_output` script located in the analyzer folder:

```bash
cd analyzer
python parse_output_new.py -i output.txt
```

The resulting table will be used for subsequent plotting.


## Description of code

### Overall preparation

The source code is organized by folders:
- dataset: Used for creating and loading datasets.
- evaluator: Used for loading models, training, and evaluation.
- logs: Output path for logs.
- model: Definitions of the models.
- scripts: Reference scripts for running the project

### All models

#### RNAFM

#### RNAMSM

#### SpTransformer
Loads the model trained by the original paper but retains only the encoder weights; the Transformer part is retrained each time.
模型文件: `model/SpTransformer`
权重文件: `fine_tuned/Splicing/SpTransformer`

#### SpliceBERT
A pre-trained BERT model that requires fine-tuning for all tasks.
模型文件: 预训练权重在`pretrained/SpliceBERT/models`，模型结构由Huggingface自动导入
权重文件: `fine_tuned/Splicing/SpliceBERT`
