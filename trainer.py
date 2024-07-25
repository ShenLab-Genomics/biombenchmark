import argparse
from evaluator.splice_evaluator import SpliceBERTEvaluator
from dataset.splice_dataset import SpliceBERTDataset
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument("--resume", default=False)
    parser.add_argument("--debug", default=False)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--patience", default=5)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ev = SpliceBERTEvaluator(train=True, tissue_num=15,
                             model_path=args.model_path,
                             tokenizer=tokenizer)
    train_dataset = SpliceBERTDataset(
        'dataset/splice_data/gtex_500_15tis/dataset_train_debug.h5',
        tokenizer=tokenizer,
        dnabert_k=None,
        max_len=500)
    test_dataset = SpliceBERTDataset(
        'dataset/splice_data/gtex_500_15tis/dataset_test_debug.h5',
        tokenizer=tokenizer,
        dnabert_k=None,
        max_len=500)
    ev.train(args, train_data=train_dataset, test_data=test_dataset)
