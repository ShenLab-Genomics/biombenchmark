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
# from model.wrap_for_cls import RNABertForSeqCls, RNAMsmForSeqCls, RNAFmForSeqCls, SeqClsLoss, DNABERTForSeqCls, DNABERT2ForSeqCls, RNAErnieForSeqCls, NTForSeqCls
from model.wrap_for_cls import SeqClsLoss
from model import wrap_models
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model.GENA import modeling_bert as GENA
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
    "mix": {
        "CD-box": 0,
        "HACA-box": 1,
        "scaRNA": 2,
        "Y_RNA": 3,
        "tRNA": 4,
        "Intron_gpI": 5,
        "Intron_gpII": 6,
        "5S_rRNA": 7,
        "5.8S_rRNA": 8,
        "miRNA": 9,
        "riboswitch": 10,
        "leader": 11,
        "IRES": 12,
        "lncRNA": 13,
        "piRNA": 14,
        "circRNA": 15
    },
    "index": {
        "0": 0,
        "1": 1,
        "2": 2
    }
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
                if (func == self.auc) or (func == self.pr_auc) or func == self.classwise_prauc:
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

    @staticmethod
    def classwise_acc(preds, labels):
        """
        For multi-class classification, calculate the class-wise accuracy. 
        # Actually it calculates the recall of each class.
        """
        num_classes = len(np.unique(labels))
        classwise_acc = []
        for i in range(num_classes):
            classwise_acc.append(
                np.sum(preds[labels == i] == i) / np.sum(labels == i))
        classwise_acc = [str(round(x, 4)) for x in classwise_acc]
        classwise_acc = "|".join(classwise_acc)
        return classwise_acc

    @staticmethod
    def classwise_prauc(preds, labels):
        """
        For multi-class classification, calculate the class-wise prauc.
        """
        num_classes = len(np.unique(labels))
        classwise_prauc = []
        for i in range(num_classes):
            # Calculate precision and recall for each class
            precision, recall, _ = precision_recall_curve(
                labels == i, preds[:, i])
            prauc = auc(recall, precision)
            classwise_prauc.append(prauc)
        classwise_prauc = [str(round(x, 4)) for x in classwise_prauc]
        classwise_prauc = "|".join(classwise_prauc)
        return classwise_prauc

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
                 replace_T=True, replace_U=False, use_kmer=1, pad_token_id=0):

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
                input_ids = input_ids[:self.max_seq_len]  # truncate

            if len(input_ids) < self.max_seq_len:
                input_ids += [self.pad_token_id] * \
                    (self.max_seq_len - len(input_ids))
            input_ids_stack.append(input_ids)
            labels_stack.append(label)

        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)),
            "labels": torch.from_numpy(self.stack_fn(labels_stack))}


class SeqClsOneHotCollator(BaseCollator):
    IN_MAP = np.asarray([[0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    # One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
    # to A, C, G, T respectively.

    IN_MAP_SS = np.asarray([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]])

    char_map = {
        'N': 0,
        'A': 1,
        'C': 2,
        'G': 3,
        'T': 4,
        '.': 0,
        '(': 1,
        ')': 2
    }

    @staticmethod
    def one_hot_encode(X, use_map):
        return use_map[X.astype('int8')]

    def __init__(self, max_seq_len, tokenizer, label2id, replace_T=True, replace_U=False, use_kmer=True, overflow=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.label2id = label2id
        assert replace_U, "Only use ACGT."

    def __call__(self, raw_data):
        input_ids_stack = []
        labels_stack = []
        for data in raw_data:
            seq = data['seq']
            seq = seq.upper()
            seq = seq.replace("U", "T")
            seq = np.array([self.char_map[x] for x in seq])
            input_ids = self.one_hot_encode(seq, self.IN_MAP)

            if len(data) == 3:
                # one-hot encoding of the secondary structure
                ss = data['struct']
                ss = np.array([self.char_map[x] for x in ss])
                ss = self.one_hot_encode(ss, self.IN_MAP_SS)
                input_ids = np.concatenate([input_ids, ss], axis=1)

            # padding to max_seq_len with zero
            if len(input_ids) < self.max_seq_len:
                input_ids = np.concatenate([input_ids, np.zeros(
                    (self.max_seq_len - len(input_ids), input_ids.shape[1]))], axis=0)
            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]

            input_ids_stack.append(input_ids)
            labels_stack.append(self.label2id[data['label']])
        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)).transpose(1, 2),
            "labels": torch.from_numpy(self.stack_fn(labels_stack))}


class SeqClsTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), mininterval=5) as pbar:
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

    def eval(self, epoch, info="Test_set"):
        self.model.eval()
        time_st = time.time()
        num_total = 0

        target_dataloader = self.eval_dataloader
        if info == "Train_set":
            target_dataloader = self.train_dataloader
        if info == "Extra_set" and self.extra_dataloader is not None:
            target_dataloader = self.extra_dataloader

        with tqdm(total=len(target_dataloader.dataset), mininterval=5) as pbar:
            outputs_dataset, labels_dataset = [], []
            for i, data in enumerate(target_dataloader):
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
        log = 'Test\t' + self.args.method + "\t" + info + "\t"
        # extract results
        for k, v in metrics_dataset.items():
            if k == "classwise_acc" or k == "classwise_prauc":
                log += k + ": {" + k + "}\t"
            else:
                log += k + ": {" + k + ":.4f}\t"
            results[k] = v

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
            label2id=LABEL2ID[args.labelset],
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
            # record performance on train set to check overfitting
            seq_cls_trainer.eval(i_epoch, info="Train_set")
            seq_cls_trainer.eval(i_epoch)
            if (i_epoch == 0) or ((i_epoch+1) % 5 == 0):
                try:
                    seq_cls_trainer.save_model(
                        f'{args.output_dir}/{args.method}', i_epoch)
                except Exception as e:
                    print(e)
                    print("Failed to save model.")


class RNABertEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        # ========== Build tokenizer, model, criterion
        model_config = get_config(args.model_config)
        self.model = BertModel(model_config)
        self.model = wrap_models.RNABertForSeqCls(
            self.model, class_num=args.class_num)
        self.model._load_pretrained_bert(args.model_path)
        self.model.to(self.device)

    def run(self, args, train_data, eval_data):
        super().run(args, train_data, eval_data)


class RNAMsmEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        model_config = get_config(args.model_config)
        self.model = MSATransformer(**model_config)
        self.model = wrap_models.RNAMsmForSeqCls(
            self.model, class_num=args.class_num)
        self.model._load_pretrained_bert(
            args.model_path)
        self.model.to(self.device)


class RNAFMEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
        self.model = wrap_models.RNAFmForSeqCls(
            self.model, class_num=args.class_num)
        self.model.to(self.device)


class DNABERTEvaluatorSeqCls(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num).to(self.device)
        self.model = wrap_models.DNABERTForSeqCls(self.model)


class SpliceBERTEvaluatorSeqCls(DNABERTEvaluatorSeqCls):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer=tokenizer)


class RNAErnieEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num
        )
        self.model = wrap_models.RNAErnieForSeqCls(self.model).to(self.device)


class DNABERT2Evaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(tokenizer)
        # config = BertConfig.from_pretrained(args.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num, trust_remote_code=True,
        )
        self.model = wrap_models.DNABERT2ForSeqCls(self.model).to(self.device)


class NTEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer=None):
        from peft import LoraConfig, TaskType, get_peft_model
        super().__init__(tokenizer)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num, trust_remote_code=True,
        )
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Trainable parameters: {}".format(trainable_params))

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        self.model = wrap_models.NTForSeqCls(self.model).to(self.device)


class GENAEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer=None):
        super().__init__(tokenizer)
        # a special hack
        self.model = GENA.BertForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num,
        )
        self.model = wrap_models.GENAForSeqCls(self.model).to(self.device)


class UTRLMEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        from model.UTRLM.utrlm import UTRLM
        super().__init__(tokenizer=tokenizer)
        self.model = UTRLM()
        # load model weights
        model_weights = torch.load(
            args.model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(
            {k.replace('module.', ''): v for k, v in model_weights.items()}, strict=True)

        self.model = wrap_models.UTRLMForSeqCls(
            self.model, class_num=args.class_num).to(self.device)


class ncRDenseEvaluator(SeqClsEvaluator):
    from model.ncRDense.model import ncRDense

    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=None)
        self.model = self.ncRDense(
            num_classes=args.class_num).float().to(self.device)

    def buildTrainer(self, args):
        self._loss_fn = SeqClsLoss().to(self.device)
        self._collate_fn = SeqClsOneHotCollator(
            max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
            replace_T=args.replace_T,
            replace_U=args.replace_U,
            label2id=LABEL2ID[args.labelset],
            use_kmer=args.use_kmer)
        self._optimizer = AdamW(params=self.model.parameters(), lr=args.lr)
        self._metric = SeqClsMetrics(metrics=args.metrics,
                                     save_path=f'{args.output_dir}/{args.method}')

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Trainable parameters: {}".format(trainable_params))

class HyenaDNAEvaluator(SeqClsEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=args.class_num, trust_remote_code=True)
        self.model = wrap_models.HyenaDNAForSeqCls(self.model).to(self.device)