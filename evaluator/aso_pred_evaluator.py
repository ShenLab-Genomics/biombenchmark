import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from evaluator.base_evaluator import BaseMetrics
from evaluator.seq_cls_evaluator import BaseCollator
from evaluator.seq_cls_evaluator import SeqClsTrainer
from model.RNABERT.bert import get_config
from model.RNABERT.rnabert import BertModel
import model.RNAFM.fm as fm
from model.BERT_like import RNABertForSeqCls, RNAMsmForSeqCls, RNAFmForSeqCls, SeqClsLoss
from model.RNAMSM.model import MSATransformer
from model.wrap_for_cls import DNABERTForSeqCls

LABEL2ID = {
    0: 0,
    1: 1,
}


class ASOPredMetrics(BaseMetrics):
    def __call__(self, outputs, labels):
        return super().__call__(outputs, labels)


def seq2kmer(seq, kmer=1):
    kmer_text = ""
    i = 0
    while i < len(seq):
        kmer_text += (seq[i: i + 1] + " ")
        i += 1
    kmer_text = kmer_text.strip()
    return kmer_text


class ASOPredCollator(BaseCollator):
    def __init__(self, max_seq_len, tokenizer, label2id,
                 replace_T=True, replace_U=False):

        super(ASOPredCollator, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.label2id = label2id
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U

    def __call__(self, raw_data_b):
        # print('raw:', raw_data_b)
        input_ids_b = []
        label_b = []
        for raw_data in raw_data_b:
            seq = raw_data["seq"]
            seq = seq.upper()
            seq = seq.replace(
                "T", "U") if self.replace_T else seq.replace("U", "T")
            kmer_text = seq2kmer(seq)
            kmer_text = kmer_text.replace("S", " [CLS] ")
            input_text = "[CLS] " + kmer_text
            # print(input_text)
            input_ids = self.tokenizer(input_text)["input_ids"]
            if None in input_ids:
                # replace all None with 0
                input_ids = [0 if x is None else x for x in input_ids]
            input_ids_b.append(input_ids)

            label = raw_data["label"]
            label_b.append(self.label2id[label])

        if self.max_seq_len == 0:
            self.max_seq_len = max([len(x) for x in input_ids_b])

        input_ids_stack = []
        labels_stack = []

        for i_batch in range(len(input_ids_b)):
            input_ids = input_ids_b[i_batch]
            label = label_b[i_batch]

            if len(input_ids) > self.max_seq_len:
                # move [SEP] to end
                # input_ids[self.max_seq_len-1] = input_ids[-1]
                input_ids = input_ids[:self.max_seq_len]

            input_ids += [0] * (self.max_seq_len - len(input_ids))
            input_ids_stack.append(input_ids)
            labels_stack.append(label)

        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)),
            "labels": torch.from_numpy(self.stack_fn(labels_stack))}


class ASOPredTrainer(SeqClsTrainer):
    def __init__(self, args, tokenizer):
        super(ASOPredTrainer, self).__init__(args, tokenizer)


# class ASOPredEvaluator:
#     def __init__(self, tokenizer=None) -> None:
#         if torch.cuda.is_available():
#             self.device = torch.device("cuda")
#         else:
#             self.device = torch.device("cpu")
#         self.tokenizer = tokenizer


class ASOPredEvaluator:

    def __init__(self, tokenizer=None) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = tokenizer
        self.seq_cls_trainer = None

    def buildTrainer(self, args):
        self._loss_fn = SeqClsLoss().to(self.device)
        self._collate_fn = ASOPredCollator(
            max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
            label2id=LABEL2ID, replace_T=args.replace_T, replace_U=args.replace_U)
        self._optimizer = AdamW(params=self.model.parameters(), lr=args.lr)
        self._metric = ASOPredMetrics(metrics=args.metrics)

    def run(self, args, train_data, eval_data):
        self.buildTrainer(args)
        args.device = self.device
        self.seq_cls_trainer = ASOPredTrainer(
            args=args,
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=self._collate_fn,
            loss_fn=self._loss_fn,
            optimizer=self._optimizer,
            compute_metrics=self._metric,
        )
        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            self.seq_cls_trainer.train(i_epoch)
            self.seq_cls_trainer.eval(i_epoch)


class RNABertEvaluator(ASOPredEvaluator):
    def __init__(self, args, tokenizer=None):
        super().__init__(tokenizer)
        # ========== Build tokenizer, model, criterion
        model_config = get_config(args.model_config)
        self.model = BertModel(model_config)
        self.model = RNABertForSeqCls(self.model)
        self.model._load_pretrained_bert(args.model_path)
        self.model.to(self.device)
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Trainable parameters: {}".format(trainable_params))


class RNAMsmEvaluator(ASOPredEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        model_config = get_config(args.model_config)
        self.model = MSATransformer(**model_config)
        self.model = RNAMsmForSeqCls(self.model)
        self.model._load_pretrained_bert(
            args.model_path)
        self.model.to(self.device)


class RNAFMEvaluator(ASOPredEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
        self.model = RNAFmForSeqCls(self.model)
        self.model.to(self.device)


class DNABERTEvaluator(ASOPredEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=13).to(self.device)
        self.model = DNABERTForSeqCls(self.model)


class SpliceBERTEvaluator(DNABERTEvaluator):
    # SpliceBERT和DNABERT结构相同，权重不同
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer=tokenizer)
