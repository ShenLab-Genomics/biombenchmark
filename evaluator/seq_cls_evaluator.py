from tqdm import tqdm
import time
from collections import defaultdict
import torch
import numpy as np

from multimolecule import RnaTokenizer, RnaErnieForSequencePrediction
from model.RNABERT.bert import get_config
from model.RNABERT.rnabert import BertModel
from model.RNAMSM.model import MSATransformer
import model.RNAFM.fm as fm
from model.BERT_like import RNABertForSeqCls, RNAMsmForSeqCls, RNAFmForSeqCls, SeqClsLoss
from model.wrap_for_cls import DNABERTForSeqCls, RNAErnieForSeqCls
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from evaluator.base_evaluator import BaseMetrics, BaseCollator, BaseTrainer

LABEL2ID = {
    "nRC": {
        "5S_rRNA": 0,
        "5_8S_rRNA": 1,
        "tRNA": 2,
        "ribozyme": 3,
        "CD-box": 4,
        "Intron_gpI": 5,
        "Intron_gpII": 6,
        "riboswitch": 7,
        "IRES": 8,
        "HACA-box": 9,
        "scaRNA": 10,
        "leader": 11,
        "miRNA": 12
    },
    "lncRNA_H": {
        "lnc": 0,
        "pc": 1
    },
    "lncRNA_M": {
        "lnc": 0,
        "pc": 1
    },
}


class SeqClsMetrics(BaseMetrics):
    def __call__(self, outputs, labels):
        return super().__call__(outputs, labels)


def seq2kmer(seq, kmer=1):
    kmer_text = ""
    i = 0
    while i+kmer < len(seq):
        kmer_text += (seq[i: i + kmer] + " ")
        i += 1
    kmer_text = kmer_text.strip()
    return kmer_text


class SeqClsCollator(BaseCollator):
    def __init__(self, max_seq_len, tokenizer, label2id,
                 replace_T=True, replace_U=False, use_kmer=True):

        super(SeqClsCollator, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.label2id = label2id
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.use_kmer = use_kmer

    def __call__(self, raw_data_b):
        # print('raw:', raw_data_b)
        input_ids_b = []
        label_b = []
        for raw_data in raw_data_b:
            seq = raw_data["seq"]
            seq = seq.upper()
            seq = seq.replace(
                "T", "U") if self.replace_T else seq.replace("U", "T")
            if self.use_kmer > 0:
                kmer_text = seq2kmer(seq, kmer=self.use_kmer)
                input_text = "[CLS] " + kmer_text
            else:
                kmer_text = seq
                input_text = kmer_text
            # input_text = "[CLS] " + kmer_text + " [SEP]"
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

            if len(input_ids) < self.max_seq_len:
                input_ids += [0] * (self.max_seq_len - len(input_ids))
            input_ids_stack.append(input_ids)
            labels_stack.append(label)

        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)),
            "labels": torch.from_numpy(self.stack_fn(labels_stack))}


class SeqClsTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset)) as pbar:
            for i, data in enumerate(self.train_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                logits = self.model(input_ids)
                # print(logits.shape, labels.shape)
                loss = self.loss_fn(logits, labels)

                # clear grads
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # log to pbar
                num_total += self.args.batch_size
                loss_total += loss.item()

                # reset loss if too many steps
                if num_total >= self.args.logging_steps:
                    pbar.set_postfix(train_loss='{:.4f}'.format(
                        loss_total / num_total))
                    pbar.update(num_total)
                    num_total, loss_total = 0, 0

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(
            loss.item(), time_ed))

    def eval(self, epoch):
        self.model.eval()
        time_st = time.time()
        num_total = 0
        with tqdm(total=len(self.eval_dataset)) as pbar:
            outputs_dataset, labels_dataset = [], []
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                with torch.no_grad():
                    logits = self.model(input_ids)

                num_total += self.args.batch_size
                outputs_dataset.append(logits)
                labels_dataset.append(labels)

                if num_total >= self.args.logging_steps:
                    pbar.update(num_total)
                    num_total = 0

        outputs_dataset = torch.concat(outputs_dataset, axis=0)
        labels_dataset = torch.concat(labels_dataset, axis=0)
        # save best model
        metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset)

        # log results to screen/bash
        results = {}
        log = 'Test\t' + self.args.method + "\t"
        # log results to visualdl
        tag_value = defaultdict(float)
        # extract results
        for k, v in metrics_dataset.items():
            log += k + ": {" + k + ":.4f}\t"
            results[k] = v
            tag = "eval/" + k
            tag_value[tag] = v

        time_ed = time.time() - time_st
        print(log.format(**results), "; Time: {:.4f}s".format(time_ed))


class SeqClsEvaluator:
    """
    RNA sequence classification任务主要参考RNAErnie的结果 https://github.com/CatIIIIIIII/RNAErnie_baselines/blob/main/datasets.py#L10
    """

    def __init__(self, tokenizer=None) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = tokenizer

    def run(self, args, train_data, eval_data):
        _loss_fn = SeqClsLoss().to(self.device)

        # ========== Create the data collator
        _collate_fn = SeqClsCollator(
            max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
            label2id=LABEL2ID['nRC'], replace_T=args.replace_T, replace_U=args.replace_U, use_kmer=args.use_kmer)

        # ========== Create the learning_rate scheduler (if need) and optimizer
        optimizer = AdamW(params=self.model.parameters(), lr=args.lr)

        # ========== Create the metrics
        _metric = SeqClsMetrics(metrics=args.metrics)

        args.device = self.device
        # ========== Create the trainer
        seq_cls_trainer = SeqClsTrainer(
            args=args,
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=_collate_fn,
            loss_fn=_loss_fn,
            optimizer=optimizer,
            compute_metrics=_metric,
        )

        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            seq_cls_trainer.train(i_epoch)
            seq_cls_trainer.eval(i_epoch)


class RNABertEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
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

    def run(self, args, train_data, eval_data):
        super().run(args, train_data, eval_data)


class RNAMsmEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        model_config = get_config(args.model_config)
        self.model = MSATransformer(**model_config)
        self.model = RNAMsmForSeqCls(self.model)
        self.model._load_pretrained_bert(
            args.model_path)
        self.model.to(self.device)

    def run(self, args, train_data, eval_data):
        _loss_fn = SeqClsLoss().to(self.device)

        # ========== Create the data collator
        _collate_fn = SeqClsCollator(
            max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
            label2id=LABEL2ID['nRC'], replace_T=args.replace_T, replace_U=args.replace_U)

        # ========== Create the learning_rate scheduler (if need) and optimizer
        optimizer = AdamW(params=self.model.parameters(), lr=args.lr)

        # ========== Create the metrics
        _metric = SeqClsMetrics(metrics=args.metrics)

        args.device = self.device
        # ========== Create the trainer
        seq_cls_trainer = SeqClsTrainer(
            args=args,
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=_collate_fn,
            loss_fn=_loss_fn,
            optimizer=optimizer,
            compute_metrics=_metric,
        )

        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            seq_cls_trainer.train(i_epoch)
            seq_cls_trainer.eval(i_epoch)
        pass


class RNAFMEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
        self.model = RNAFmForSeqCls(self.model)
        self.model.to(self.device)

    def run(self, args, train_data, eval_data):
        _loss_fn = SeqClsLoss().to(self.device)

        # ========== Create the data collator
        _collate_fn = SeqClsCollator(
            max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
            label2id=LABEL2ID['nRC'], replace_T=args.replace_T, replace_U=args.replace_U)

        # ========== Create the learning_rate scheduler (if need) and optimizer
        optimizer = AdamW(params=self.model.parameters(), lr=args.lr)

        # ========== Create the metrics
        _metric = SeqClsMetrics(metrics=args.metrics)

        args.device = self.device
        # ========== Create the trainer
        seq_cls_trainer = SeqClsTrainer(
            args=args,
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=_collate_fn,
            loss_fn=_loss_fn,
            optimizer=optimizer,
            compute_metrics=_metric,
        )

        for i_epoch in range(args.num_train_epochs):
            print("Epoch: {}".format(i_epoch))
            seq_cls_trainer.train(i_epoch)
            seq_cls_trainer.eval(i_epoch)
        pass


class DNABERTEvaluatorSeqCls(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num).to(self.device)
        self.model = DNABERTForSeqCls(self.model)


class SpliceBERTEvaluatorSeqCls(DNABERTEvaluatorSeqCls):
    # SpliceBERT和DNABERT结构相同，权重不同
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer=tokenizer)


class RNAErnieEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        tokenizer = RnaTokenizer.from_pretrained('model/pretrained/rnaernie')
        super().__init__(tokenizer)
        self.model = RnaErnieForSequencePrediction.from_pretrained(
            'model/pretrained/rnaernie', num_labels=args.class_num)
        self.model = RNAErnieForSeqCls(
            self.model).to(self.device)


class DNABERT2Evaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num
        )
