import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import time
from model.RNABERT.bert import get_config
from model.RNABERT.rnabert import BertModel
from model.RNAMSM.model import MSATransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict
from evaluator.base_evaluator import BaseMetrics
from evaluator.seq_cls_evaluator import SeqClsTrainer, SeqClsCollator, SeqClsEvaluator, seq2kmer
import model.RNAFM.fm as fm
from model.BERT_like import RNABertForSeqCls, RNAMsmForSeqCls, RNAFmForSeqCls
from model.wrap_for_cls import DNABERTForSeqCls
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from model.wrap_for_cls import DNABERTForSeqCls, RNAErnieForSeqCls, RNABertForM6ACls, DNABERT2ForSeqCls
from model.wrap_for_mrl import RNAFmForReg, PureReg, UTRLMForReg
from model.UTRlm import utrlm
import scipy.stats as stats
from sklearn import preprocessing


class MRLMetrics(BaseMetrics):
    def __call__(self, outputs, labels):
        """
        Args:
            outputs: logits in tensor
            labels: labels in tensor
        Returns:
            metrics in dict
        """
        # regression model
        logits = outputs.cpu().numpy().astype('float')
        labels = labels.cpu().numpy().astype('float')

        res = {}
        for name in self.metrics:
            func = getattr(self, name)
            if func:
                m = func(logits, labels)
                res[name] = m
            else:
                raise NotImplementedError
        return res

    @staticmethod
    def r2(logits, labels):
        """
        All args have same shapes.
        Args:
            logits: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            R2
        """
        return r2_score(labels, logits)

    @staticmethod
    def mse(logits, labels):
        """
        All args have same shapes.
        Args:
            logits: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            MSE
        """
        return mean_squared_error(labels, logits)

    @staticmethod
    def mae(logits, labels):
        """
        All args have same shapes.
        Args:
            logits: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            MAE
        """

        return mean_absolute_error(labels, logits)

    @staticmethod
    def pearson(logits, labels):
        """
        All args have same shapes.
        Args:
            logits: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            Pearson
        """
        return stats.pearsonr(labels, logits)[0]

    @staticmethod
    def spearman(logits, labels):
        """
        All args have same shapes.
        Args:
            logits: predictions of model, (batch_size, 1)
            labels: ground truth, (batch_size, 1)

        Returns:
            Spearman
        """
        return stats.spearmanr(labels, logits)[0]


class MRLCollator(SeqClsCollator):
    def __init__(self, max_seq_len, tokenizer,
                 replace_T=True, replace_U=False, use_kmer=True):
        super(MRLCollator, self).__init__(max_seq_len, tokenizer, None,
                                          replace_T, replace_U, use_kmer)

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
            label_b.append(label)

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


class MRLTrainer(SeqClsTrainer):
    pass


class MRLLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, outputs, labels):
        """
            outputs: [Batch, 1]
            labels: [Batch,]
        """
        outputs = outputs.float()
        labels = labels.float()

        return self.loss_fn(outputs, labels)


class MRLEvaluator():
    # regression task
    def __init__(self, tokenizer=None) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = tokenizer

    def buildTrainer(self, args):
        self._loss_fn = MRLLoss().to(self.device)
        self._collate_fn = MRLCollator(
            max_seq_len=args.max_seq_len, tokenizer=self.tokenizer, replace_T=args.replace_T, replace_U=args.replace_U, use_kmer=args.use_kmer)
        self._optimizer = AdamW(params=self.model.parameters(), lr=args.lr)
        self._metric = MRLMetrics(metrics=args.metrics)

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Trainable parameters: {}".format(trainable_params))

    def run(self, args, train_data, eval_data):
        self.buildTrainer(args)
        args.device = self.device
        self.seq_cls_trainer = MRLTrainer(
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


class RNAFMEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
        # self.model = RNAFmForReg(self.model)
        self.model = RNAFmForReg(self.model)
        # self.model = PureReg()
        self.model.to(self.device)


class DNABERTEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=1).to(self.device)

        self.model = DNABERTForSeqCls(self.model)


class RNAMsmEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        model_config = get_config(args.model_config)
        self.model = MSATransformer(**model_config)
        self.model = RNAMsmForSeqCls(self.model, class_num=1)
        self.model._load_pretrained_bert(
            args.model_path)
        self.model.to(self.device)


class RNABertEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        # ========== Build tokenizer, model, criterion
        model_config = get_config(args.model_config)
        self.model = BertModel(model_config)
        self.model = RNABertForM6ACls(self.model, class_num=1)
        self.model._load_pretrained_bert(args.model_path)
        self.model.to(self.device)


class UTRLMEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        from multimolecule import RnaTokenizer, UtrLmForSequencePrediction
        tokenizer = RnaTokenizer.from_pretrained('model/UTRLM')
        super().__init__(tokenizer=tokenizer)
        model = UtrLmForSequencePrediction.from_pretrained(
            'model/UTRLM').to(self.device)

        self.model = UTRLMForReg(model)
        # self.model = AutoModelForSequenceClassification.from_pretrained(
        #     args.model_path, num_labels=1).to(self.device)
        # self.model = DNABERT2ForSeqCls(self.model)


class UTRLMoriginalEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        modelfile = '/public/home/shenninggroup/yny/code/biombenchmark/model/UTRlm/model.pt'

        model = utrlm.CNN_linear()
        # st.write(model.state_dict().keys())
        # st.write({k.replace('module.', ''):v for k,v in torch.load(modelfile, map_location=torch.device('cpu')).items()}.keys())
        model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(
            modelfile, map_location=torch.device('cpu')).items()}, strict=True)
        self.model = model.to(self.device)

    def run(self, args, train_data, eval_data):
        self.buildTrainer(args)
        args.device = self.device
        self.seq_cls_trainer = MRLTrainer(
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
            self.seq_cls_trainer.eval(i_epoch)