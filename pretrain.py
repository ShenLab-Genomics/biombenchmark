import argparse
from transformers import AutoTokenizer, MambaConfig, Mamba2Model
from evaluator import seq_cls_evaluator
from dataset import seq_cls_dataset
from evaluator import splice_evaluator
from dataset import splice_dataset
from model.BERT_like import RNATokenizer
from model.RNAErnie.tokenization_rnaernie import RNAErnieTokenizer

MAX_SEQ_LEN = {"RNAMSM": 512,
               "RNAFM": 512,
               'DNABERT': 512,
               "SpliceBERT": 510,
               'RNAMSM': 512,
               "RNAErnie": 510,
               "SpTransformer": 9000,
               "RNAMAMBA": 1024
               }


def str2list(v):
    if isinstance(v, list):
        return v
    elif isinstance(v, str):
        vs = v.split(",")
        return [v.strip() for v in vs]
    else:
        raise argparse.ArgumentTypeError(
            "Str value seperated by ', ' expected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument(
        "--method", choices=['RNAMAMBA'], default='RNABERT')
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=6, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--class_num", default=3, type=int)
    parser.add_argument("--model_path", default=False)
    parser.add_argument("--output_dir", default=False)
    parser.add_argument("--model_config", default=False)
    parser.add_argument("--vocab_path", default='model/RNABERT/vocab.txt')
    parser.add_argument("--use_kmer", default=0)
    parser.add_argument("--dataset_train", type=str)
    parser.add_argument("--dataset_test", type=str)
    parser.add_argument('--metrics', type=str2list,
                        default="topk,pr_auc,roc_auc",)

    args = parser.parse_args()

    # 自定义分词器
    tokenizer = AutoTokenizer.from_pretrained('/public/home/shenninggroup/yny/code/biombenchmark/model/pretrained/SpliceBERT/models/SpliceBERT.1024nt',
                                              from_slow=True,
                                              legacy=False)

    # # 示例文本
    # text = "Hey how are you doing?"
    # input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    args.max_seq_len = MAX_SEQ_LEN["RNAMAMBA"]
    args.replace_T = False
    args.replace_U = True

    dataset_train = splice_dataset.SpliceNormalDataset(
        h5_filename=args.dataset_train)
    dataset_test = splice_dataset.SpliceNormalDataset(
        h5_filename=args.dataset_test)

    ev = splice_evaluator.RNAMAMBAEvaluator(args, tokenizer=tokenizer)
    ev.run(args, dataset_train, dataset_test)
