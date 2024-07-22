from evaluator.splice_evaluator import SpTransformerEvaluator
from dataset.splice_dataset import SpliceDataset
from torch.utils.data import DataLoader

if __name__ == '__main__':

    # RNA classification

    # Splicing
    dataset = SpliceDataset(
        'dataset/splice_data/gtex_500_15tissue/dataset_test_0.h5')
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=12,
        num_workers=2)
    runner = SpTransformerEvaluator(
        '/public/home/shenninggroup/yny/code/CellSplice/runs/T-Ex2-1000-128-7/best.ckpt')
    result = runner.run(dataloader)

    print('SpTransformer result', result)
