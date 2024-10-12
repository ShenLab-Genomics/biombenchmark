import os
from tqdm import tqdm
import torch
import numpy as np
from evaluator.seq_cls_evaluator import SeqClsTrainer, SeqClsCollator
import model.DeepM6ASeq.model
from model.RNABERT.bert import get_config
from model.RNABERT.rnabert import BertModel
from model.RNAMSM.model import MSATransformer
import model.RNAFM.fm as fm
from model.BERT_like import RNAMsmForSeqCls, RNAFmForSeqCls, SeqClsLoss
from model.wrap_for_cls import DNABERTForSeqCls, RNAErnieForSeqCls, RNABertForM6ACls, DNABERT2ForSeqCls
import model.DeepM6ASeq
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from evaluator.base_evaluator import BaseMetrics, BaseCollator, BaseTrainer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc)


LABEL2ID = {
    '0': 0,
    '1': 1,
}


class M6APredMetrics(BaseMetrics):
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

        # preds = 1 / (1+np.exp(-preds))  # sigmoid
        labels = labels.cpu().numpy().astype('int32')
        # pred_score = torch.log_softmax(
        #     outputs, dim=-1).cpu().numpy()
        pred_score = torch.softmax(outputs, dim=-1).cpu().numpy()
        pred_class = torch.argmax(
            outputs, axis=-1).cpu().numpy()

        res = {}
        for name in self.metrics:
            func = getattr(self, name)
            if func:
                if (func == self.auc) or (func == self.pr_auc):
                    # given two neural outputs, calculate their logits
                    # and then calculate auc
                    m = func(pred_score, labels)
                else:
                    m = func(pred_class, labels)

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


class M6APredCollator(SeqClsCollator):
    def __init__(self, max_seq_len, tokenizer, label2id,
                 replace_T=True, replace_U=False, use_kmer=True):
        super(M6APredCollator, self).__init__(max_seq_len, tokenizer, label2id,
                                              replace_T, replace_U, use_kmer)


class M6APredTrainer(SeqClsTrainer):
    pass


class M6ALoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        outputs = outputs
        labels = labels
        return self.loss_fn(outputs, labels)


class M6APredEvaluator():
    def __init__(self, tokenizer=None) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = tokenizer

    def buildTrainer(self, args):
        self._loss_fn = M6ALoss().to(self.device)
        self._collate_fn = M6APredCollator(
            max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
            label2id=LABEL2ID, replace_T=args.replace_T, replace_U=args.replace_U, use_kmer=args.use_kmer)
        self._optimizer = AdamW(params=self.model.parameters(), lr=args.lr)
        self._metric = M6APredMetrics(metrics=args.metrics,
                                      save_path=f'{args.output_dir}/{args.method}')

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Trainable parameters: {}".format(trainable_params))

    def run(self, args, train_data, eval_data):
        self.buildTrainer(args)
        args.device = self.device
        self.seq_cls_trainer = M6APredTrainer(
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


#### Evaluator for models ####

class RNAFMEvaluator(M6APredEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
        self.model = RNAFmForSeqCls(self.model, class_num=2)
        self.model.to(self.device)


class RNAMsmEvaluator(M6APredEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        model_config = get_config(args.model_config)
        self.model = MSATransformer(**model_config)
        self.model = RNAMsmForSeqCls(self.model, class_num=2)
        self.model._load_pretrained_bert(
            args.model_path)
        self.model.to(self.device)


class RNABertEvaluator(M6APredEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        # ========== Build tokenizer, model, criterion
        model_config = get_config(args.model_config)
        self.model = BertModel(model_config)
        self.model = RNABertForM6ACls(self.model, class_num=2)
        if args.model_path:
            self.model._load_pretrained_bert(args.model_path)
        else:
            print('Use un-pretrained model')
        self.model.to(self.device)


class DNABERTEvaluator(M6APredEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=2).to(self.device)
        self.model = DNABERTForSeqCls(self.model)


class DNABERT2Evaluator(M6APredEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(tokenizer)
        # config = BertConfig.from_pretrained(args.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=2, trust_remote_code=True,
        )
        self.model = DNABERT2ForSeqCls(self.model).to(self.device)


class SpliceBERTEvaluator(DNABERTEvaluator):
    # SpliceBERT和DNABERT结构相同，权重不同
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer=tokenizer)


class RNAErnieEvaluator(M6APredEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=args.class_num
        )
        self.model = RNAErnieForSeqCls(
            self.model).to(self.device)


class DeepM6ASeqEvaluator(M6APredEvaluator):

    def __init__(self, args, tokenizer) -> None:
        from dataset.splice_data.data_maker import IN_MAP, one_hot_encode

        def one_hot_embed(x):
            str_map = {
                'A': 1,
                'C': 2,
                'G': 3,
                'T': 4
            }
            x = [str_map[a] for a in x]
            x = one_hot_encode(np.array(x), IN_MAP)
            return {
                'input_ids': x
            }

        super().__init__(tokenizer=one_hot_embed)
        self.model = model.DeepM6ASeq.model.ConvNet_BiLSTM(
            output_dim=2, args=args, wordvec_len=4).to(self.device)
