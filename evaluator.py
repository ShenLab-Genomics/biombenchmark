from evaluator.splice_evaluator import SpTransformerEvaluator
from dataset.splice_dataset import SpTransformerDataset
from torch.utils.data import DataLoader
import torch
if __name__ == '__main__':
    # RNA classification

    # Splicing
    dataset = SpTransformerDataset(
        'dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5')
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=6,
        num_workers=2)
    runner = SpTransformerEvaluator(
        model_weight='/public/home/shenninggroup/yny/code/CellSplice/runs/T-Ex2-1000-128-7/best.ckpt')
    result = runner.run(dataloader)

    print('SpTransformer result', result)
