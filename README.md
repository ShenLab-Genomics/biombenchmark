# biom-benchmark

## Progress

### Overall preparation

dataloader: 每个任务数据集的加载器
Model_dataloader: 每个模型的数据集加载器

dataset: 描述源数据的获取方式，制作数据集的代码

evaluator：评估器，评估模型的性能

runner：训练每个模型的代码

### All models

#### SpTransformer
15tis使用原本训练好的模型。0tis和54tis使用重新训练的模型
模型文件: `model/SpTransformer`
权重文件: `fine_tuned/Splicing/SpTransformer`

#### SpliceBERT
预训练BERT模型，在所有任务上都需要fine-tune
模型文件: 预训练权重在`pretrained/SpliceBERT/models`，模型结构由Huggingface自动导入
权重文件: `fine_tuned/Splicing/SpliceBERT`



### All tasks

#### Splicing prediction

运行 `scripts/run_splicing_makedata.sh` 制作数据集

运行 `scripts/run_splicing_test.sh` 测试所有受支持的模型

### Results
