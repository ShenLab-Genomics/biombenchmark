import os
from tqdm import tqdm
import time
from collections import defaultdict
import torch
import numpy as np
from model.RNABERT.bert import get_config
from model.RNABERT.rnabert import BertModel
from model.RNAMSM.model import MSATransformer
import model.RNAFM.fm as fm
from model.wrap_for_cls import RNABertForSeqCls, RNAMsmForSeqCls, RNAFmForSeqCls, SeqClsLoss, DNABERTForSeqCls, DNABERT2ForSeqCls, RNAErnieForSeqCls
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from evaluator.base_evaluator import BaseMetrics, BaseCollator, BaseTrainer
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc)


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
    def __init__(self, metrics, save_path=None):
        super().__init__(metrics)
        self.save_path = save_path
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

    def __call__(self, outputs, labels, epoch=0):
        """
        Args:
            kwargs: required args of model (dict)

        Returns:
            metrics in dict
        """
        preds = torch.argmax(outputs, axis=-1)
        preds = preds.cpu().numpy().astype('int32')
        labels = labels.cpu().numpy().astype('int32')

        res = {}
        for name in self.metrics:
            func = getattr(self, name)
            if func:
                if (func == self.auc) or (func == self.pr_auc):
                    # given two neural outputs, calculate their logits
                    # and then calculate auc
                    logits = torch.sigmoid(outputs).cpu().numpy()
                    m = func(logits, labels)
                else:
                    m = func(preds, labels)
                if isinstance(m, tuple) and len(m) > 1:
                    res[name] = m[0]
                    if self.save_path is not None:
                        for idx, item in enumerate(m):
                            fsave = os.path.join(
                                self.save_path, f'epoch_{epoch}_{name}_{idx}')
                            np.save(fsave, item)
                else:
                    res[name] = m
            else:
                raise NotImplementedError
        return res

    @staticmethod
    def auc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        # labels += 1
        preds = preds[:, 1]

        fpr, tpr, _ = roc_curve(labels, preds)
        return auc(fpr, tpr), fpr, tpr

    @staticmethod
    def pr_auc(preds, labels):
        """
        All args have same shapes.
        Args:
            preds: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            precision
        """
        # labels += 1
        preds = preds[:, 1]

        precision, recall, _ = precision_recall_curve(labels, preds)

        prauc = auc(recall, precision)
        return prauc, precision, recall

    def emb(self, pred, labels, epoch=0):
        fsave = os.path.join(
            self.save_path, f'epoch_{epoch}_emb_pred')
        np.save(fsave, pred)
        fsave = os.path.join(
            self.save_path, f'epoch_{epoch}_emb_labels')
        np.save(fsave, labels)


def seq2kmer(seq, kmer=1):
    kmer_text = ""
    i = 0
    while i+kmer <= len(seq):
        kmer_text += (seq[i: i + kmer] + " ")
        i += 1
    kmer_text = kmer_text.strip()
    return kmer_text


class SeqClsCollator(BaseCollator):
    def __init__(self, max_seq_len, tokenizer, label2id,
                 replace_T=True, replace_U=False, use_kmer=1,pad_token_id=0):

        super(SeqClsCollator, self).__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.label2id = label2id
        # only replace T or U
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.use_kmer = use_kmer
        self.pad_token_id = pad_token_id

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
            # print(input_text)
            input_ids = self.tokenizer(input_text)["input_ids"]
            # exit()
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
                input_ids = input_ids[:self.max_seq_len] # truncate

            if len(input_ids) < self.max_seq_len:
                input_ids += [self.pad_token_id] * (self.max_seq_len - len(input_ids))
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
        metrics_dataset = self.compute_metrics(outputs_dataset, labels_dataset,
                                               epoch=epoch)

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

    def extract_embedding(self, epoch):
        self.model.eval()
        time_st = time.time()
        num_total = 0
        with tqdm(total=len(self.eval_dataset)) as pbar:
            outputs_dataset, labels_dataset = [], []
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                with torch.no_grad():
                    logits = self.model(input_ids, return_embedding=True)

                num_total += self.args.batch_size
                outputs_dataset.append(logits.cpu().detach())
                labels_dataset.append(labels.cpu().detach())

                if num_total >= self.args.logging_steps:
                    pbar.update(num_total)
                    num_total = 0

        outputs_dataset = torch.concat(outputs_dataset, axis=0)
        labels_dataset = torch.concat(labels_dataset, axis=0)

        # save emb
        self.compute_metrics.emb(outputs_dataset, labels_dataset, epoch)


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

    def buildTrainer(self, args):
        self._loss_fn = SeqClsLoss().to(self.device)
        self._collate_fn = SeqClsCollator(
            max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
            label2id=LABEL2ID['nRC'], 
            replace_T=args.replace_T, 
            replace_U=args.replace_U, 
            use_kmer=args.use_kmer,
            pad_token_id=args.pad_token_id)
        self._optimizer = AdamW(params=self.model.parameters(), lr=args.lr)
        self._metric = SeqClsMetrics(metrics=args.metrics,
                                     save_path=f'{args.output_dir}/{args.method}')

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Trainable parameters: {}".format(trainable_params))

    def run(self, args, train_data, eval_data):
        self.buildTrainer(args)

        args.device = self.device
        # ========== Create the trainer
        seq_cls_trainer = SeqClsTrainer(
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
            if args.extract_emb:
                seq_cls_trainer.extract_embedding(i_epoch)
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


class RNAFMEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
        self.model = RNAFmForSeqCls(self.model)
        self.model.to(self.device)


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
        super().__init__(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num
        )
        self.model = RNAErnieForSeqCls(self.model).to(self.device)


class DNABERT2Evaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(tokenizer)
        # config = BertConfig.from_pretrained(args.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num, trust_remote_code=True,
        )
        self.model = DNABERT2ForSeqCls(self.model).to(self.device)
