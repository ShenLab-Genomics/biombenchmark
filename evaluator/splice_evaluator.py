from tqdm import tqdm
import time
from collections import defaultdict
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from model.BERT_like import RNATokenizer
from model.RNABERT.bert import get_config
from model.RNABERT.rnabert import BertModel
from model.RNAMSM.model import MSATransformer
import model.RNAFM.fm as fm
from model.BERT_like import SeqClsLoss
from model.wrap_for_splice import SpliceBERTForTokenCls, DNABERTForTokenCls, RNAFmForTokenCls, RNAErnieForTokenCls, RNAMsmForTokenCls
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification
from evaluator.base_evaluator import BaseMetrics, BaseCollator, BaseTrainer
from sklearn.metrics import average_precision_score, roc_auc_score


class SpliceTokenClsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        pred_dim = outputs.shape[1]
        res = {}
        if pred_dim == 3:
            # neither / acceptor / donor
            # calculate class 1 and 2
            outputs = outputs[:, :3]
            labels = labels[:, :3]
            loss = F.cross_entropy(outputs, labels)

        elif pred_dim == 18 or pred_dim == 56:

            if pred_dim == 18:
                # 3 classes and 15 tissues
                # calculate 15 tissues separately
                outputs = outputs[:, 3:]
                labels = labels[:, 3:]
                pass

            if pred_dim == 56:
                # 3 classes and 53 tissues
                # calculate 53 tissues separately
                outputs = outputs[:, 3:]
                labels = labels[:, 3:]
                pass

            assert torch.isnan(labels).sum() == 0, "Labels is NaN."
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            assert torch.isnan(loss).sum() == 0, "Loss is NaN."
        else:
            raise ValueError("Invalid prediction dimension.")
        return loss


class SpliceMetrics(BaseMetrics):
    def __call__(self, outputs, labels):
        """
        Args:
            outputs: logits in tensor
            labels: labels in tensor
        Returns:
            metrics in dict
        """
        # outputs = outputs.cpu().detach().numpy()
        # labels = labels.cpu().detach().numpy()
        pred_dim = outputs.shape[-1]
        res = {}
        if pred_dim == 3:
            # neither / acceptor / donor
            # calculate class 1 and 2
            outputs = torch.softmax(outputs, dim=1)[:, 1:]
            labels = labels[:, 1:]

        elif pred_dim == 18 or pred_dim == 56:

            if pred_dim == 18:
                # 3 classes and 15 tissues
                # calculate 15 tissues separately
                outputs = torch.sigmoid(outputs[:, 3:])
                labels = labels[:, 3:]
                pass

            if pred_dim == 56:
                # 3 classes and 51 tissues
                # calculate 51 tissues separately
                outputs = torch.sigmoid(outputs[:, 3:])
                labels = labels[:, 3:]
                pass

        else:
            raise ValueError("Invalid prediction dimension.")

        for name in self.metrics:
            func = getattr(self, name)
            if func:
                for i in range(outputs.shape[1]):
                    m = func(outputs[:, i], labels[:, i])
                    res[name + '_' + str(i)] = m
                # m = func(outputs, labels)
            else:
                raise NotImplementedError
        return res

    def topk(self, preds, labels):
        '''
        Parameters
        ---
            y_true:     1-dim array, label for one of the tissues
            y_pred:     1-dim array, model output for the tissue

        Returns
        ---
            topkl_accuracy: float, the calculated topkl accuracy
        '''
        idx_true = np.nonzero(labels >= 0.5)
        argsorted_y_pred = np.argsort(preds)
        sorted_y_pred = preds[argsorted_y_pred]

        # Get top-1L index
        idx_pred = argsorted_y_pred[-int(1 * len(idx_true)):]
        if len(idx_true) <= 0:
            raise ValueError("No positive data!")

        topkl_accuracy = np.size(np.intersect1d(
            idx_true, idx_pred)) / float(min(len(idx_pred), len(idx_true)))

        threshold = sorted_y_pred[-int(1 * len(idx_true))]

        return {
            'topkl': topkl_accuracy,
            'threshold': threshold
        }

    def pr_auc(self, preds, labels):
        '''
        Parameters
        ---
            y_true:     1-dim array, label for one of the tissues
            y_pred:     1-dim array, model output for the tissue

        Returns
        ---
            auprc:      float , the calculated auprc value
        '''
        auprc = average_precision_score(labels >= 0.5, preds)
        return auprc

    def roc_auc(self, preds, labels):
        auroc = roc_auc_score(labels >= 0.5, preds)
        return auroc


class SpliceTrainer(BaseTrainer):
    def train(self, epoch):
        self.model.train()
        time_st = time.time()
        num_total, loss_total = 0, 0

        with tqdm(total=len(self.train_dataset), mininterval=2) as pbar:
            for i, data in enumerate(self.train_dataloader):
                # for non-language model, the "input_ids" represents the one-hot encoding of the sequence
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                logits = self.model(input_ids)
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

                if i > 1000:
                    break

        time_ed = time.time() - time_st
        print('Train\tLoss: {:.6f}; Time: {:.4f}s'.format(
            loss.item(), time_ed))

    def eval(self, epoch):
        self.model.eval()
        time_st = time.time()
        num_total = 0
        with tqdm(total=len(self.eval_dataset), mininterval=2) as pbar:
            outputs_dataset, labels_dataset = [], []
            for i, data in enumerate(self.eval_dataloader):
                input_ids = data["input_ids"].to(self.args.device)
                labels = data["labels"].to(self.args.device)

                with torch.no_grad():
                    logits = self.model(input_ids)

                outputs_dataset.append(logits)
                labels_dataset.append(labels)
                num_total += self.args.batch_size

                if num_total >= self.args.logging_steps:
                    pbar.update(num_total)
                    num_total = 0

        outputs_dataset = torch.concat(outputs_dataset, axis=0)
        labels_dataset = torch.concat(labels_dataset, axis=0)

        # Now the shape of output is (total_batch, num_labels, seq_len)
        # We should reshape it to (total_batch*seq_len, num_labels)
        outputs_dataset = outputs_dataset.transpose(1, 2)
        outputs_dataset = outputs_dataset.reshape(-1, outputs_dataset.shape[2])
        outputs_dataset = outputs_dataset.cpu().detach()
        labels_dataset = labels_dataset.transpose(1, 2)
        labels_dataset = labels_dataset.reshape(-1, labels_dataset.shape[2])
        labels_dataset = labels_dataset.cpu().detach()

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
        print(results, "; Time: {:.4f}s".format(time_ed))


def seq2kmer(seq, kmer=1):
    kmer_text = ""
    i = 0
    while i < len(seq):
        kmer_text += (seq[i: i + 1] + " ")
        i += 1
    kmer_text = kmer_text.strip()
    return kmer_text


class SpliceTokenCollator(BaseCollator):
    def __init__(self, max_seq_len, tokenizer, replace_T=True, replace_U=False, use_kmer=True, overflow=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        assert replace_T ^ replace_U, "Only replace T or U."
        self.replace_T = replace_T
        self.replace_U = replace_U
        self.use_kmer = int(use_kmer)
        self.overflow = overflow

    def __call__(self, raw_data):
        input_ids_stack = []
        labels_stack = []

        for data in raw_data:
            seq = data[0]
            seq = seq.upper()
            seq = seq.replace(
                "T", "U") if self.replace_T else seq.replace("U", "T")

            start = (len(seq) - self.max_seq_len)//2
            if self.use_kmer > 0:
                seq = seq[start-self.overflow:start +
                          self.max_seq_len+self.overflow]
                seq = seq2kmer(seq)
            else:
                seq = seq[start:start+self.max_seq_len]

            input_text = "[CLS]" + seq
            # print(input_text)
            input_ids = self.tokenizer(input_text)["input_ids"]
            # print(input_ids)
            input_ids_stack.append(input_ids)

            label = data[1]
            labels_stack.append(label)

        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)),
            "labels": torch.from_numpy(self.stack_fn(labels_stack))}


class SpliceOneHotCollator(BaseCollator):
    IN_MAP = np.asarray([[0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    # One-hot encoding of the inputs: 0 is for padding, and 1, 2, 3, 4 correspond
    # to A, C, G, T respectively.

    OUT_MAP = np.asarray([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1],
                          [0, 0, 0]])
    # One-hot encoding of the outputs: 0 is for no splice, 1 is for acceptor,
    # 2 is for donor and -1 is for padding.

    @staticmethod
    def one_hot_encode(X, use_map):
        return use_map[X.astype('int8')]

    def __init__(self, max_seq_len, tokenizer, replace_T=True, replace_U=False, use_kmer=True, overflow=0):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        assert replace_U, "Only use ACGT."

    def __call__(self, raw_data):
        input_ids_stack = []
        labels_stack = []
        for data in raw_data:
            seq = data[0]
            seq = seq.upper()
            seq = seq.replace("U", "T")
            start = (len(seq) - self.max_seq_len)//2
            seq = seq[start:start+self.max_seq_len]
            input_ids = self.one_hot_encode(seq, self.IN_MAP)
            input_ids_stack.append(input_ids)
            labels_stack.append(data[1])
        return {
            "input_ids": torch.from_numpy(self.stack_fn(input_ids_stack)).transpose(1, 2),
            "labels": torch.from_numpy(self.stack_fn(labels_stack))}


class SpliceEvaluator:
    def __init__(self, args, tokenizer=None) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = tokenizer
        self.token_cls_trainer = None
        self.mode = 'token'

    def buildTrainer(self, args):
        mode = self.mode
        if mode == 'token':
            self._loss_fn = SpliceTokenClsLoss().to(self.device)
            self._collate_fn = SpliceTokenCollator(
                max_seq_len=args.max_seq_len, tokenizer=self.tokenizer, replace_T=args.replace_T, replace_U=args.replace_U, use_kmer=args.use_kmer)
            self._optimizer = AdamW(params=self.model.parameters(), lr=args.lr)
            self._metric = SpliceMetrics(metrics=args.metrics)
        elif mode == 'onehot':
            self._loss_fn = SeqClsLoss().to(self.device)
            self._collate_fn = SpliceOneHotCollator(
                max_seq_len=args.max_seq_len, tokenizer=self.tokenizer, replace_T=args.replace_T, replace_U=args.replace_U, use_kmer=args.use_kmer)
            self._optimizer = AdamW(params=self.model.parameters(), lr=args.lr)
            self._metric = SpliceMetrics(metrics=args.metrics)
        else:
            raise ValueError("Invalid mode.")

    def run(self, args, train_data, eval_data):
        self.buildTrainer(args)
        args.device = self.device
        self.token_cls_trainer = SpliceTrainer(
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
            self.token_cls_trainer.train(i_epoch)
            self.token_cls_trainer.eval(i_epoch)
            if (i_epoch == 0) or ((i_epoch+1) % 5 == 0):
                self.token_cls_trainer.save_model(
                    args.output_dir, i_epoch)


class SpliceBERTEvaluator(SpliceEvaluator):
    def __init__(self, args, tokenizer=None):
        super().__init__(args, tokenizer)
        # set the path to the folder of pre-trained SpliceBERT
        self.SPLICEBERT_PATH = args.model_path
        # load tokenizer
        self.tokenizer = tokenizer
        self.class_num = args.class_num
        self.model = AutoModelForTokenClassification.from_pretrained(
            args.model_path, num_labels=args.class_num).to(self.device)
        self.model = SpliceBERTForTokenCls(self.model)
        self.mode = 'token'

    pass


class DNABERTEvaluator(SpliceBERTEvaluator):
    def __init__(self, args, tokenizer=None):
        super().__init__(args, tokenizer)
        self.model = AutoModelForTokenClassification.from_pretrained(
            args.model_path, num_labels=args.class_num).to(self.device)
        self.model = DNABERTForTokenCls(self.model)


class RNAFMEvaluator(SpliceEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer=tokenizer)
        self.model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
        self.model = RNAFmForTokenCls(self.model, num_labels=args.class_num)
        self.model.to(self.device)
        self.mode = 'token'


class RNAMSMEvaluator(SpliceEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(args, tokenizer)
        model_config = get_config(args.model_config)
        self.model = MSATransformer(**model_config)
        self.model = RNAMsmForTokenCls(self.model, num_labels=args.class_num)
        self.model._load_pretrained_bert(
            args.model_path)
        self.model.to(self.device)
        self.mode = 'token'


class RNAErnieEvaluator(SpliceEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(args, tokenizer)
        self.model = AutoModelForTokenClassification.from_pretrained(
            args.model_path, num_labels=args.class_num
        )
        self.model = RNAErnieForTokenCls(
            self.model, num_labels=args.class_num).to(self.device)
        self.mode = 'token'
