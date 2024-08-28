import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from evaluator.base_evaluator import BaseMetrics
from evaluator.seq_cls_evaluator import SeqClsTrainer, SeqClsCollator
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


class RBPPredMetrics(BaseMetrics):
    def __call__(self, outputs, labels):
        return super().__call__(outputs, labels)


class RBPPredCollator(SeqClsCollator):
    def __init__(self, max_seq_len, tokenizer, label2id,
                 replace_T=True, replace_U=False, use_kmer=True):
        super(RBPPredCollator, self).__init__(max_seq_len, tokenizer, label2id,
                                              replace_T, replace_U, use_kmer)


class RBPPredTrainer(SeqClsTrainer):
    pass


class RBPPredEvaluator():
    def __init__(self, tokenizer=None) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = tokenizer

    def buildTrainer(self, args):
        self._loss_fn = SeqClsLoss().to(self.device)
        self._collate_fn = RBPPredCollator(
            max_seq_len=args.max_seq_len, tokenizer=self.tokenizer,
            label2id=LABEL2ID, replace_T=args.replace_T, replace_U=args.replace_U)
        self._optimizer = AdamW(params=self.model.parameters(), lr=args.lr)
        self._metric = RBPPredMetrics(metrics=args.metrics)

    def run(self, args, train_data, eval_data):
        self.buildTrainer(args)
        args.device = self.device
        self.seq_cls_trainer = RBPPredTrainer(
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


class RNAFMEvaluator(RBPPredEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
        self.model = RNAFmForSeqCls(self.model, class_num=2)  # 二分类
        self.model.to(self.device)


class RNAMsmEvaluator(RBPPredEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(tokenizer=tokenizer)
        model_config = get_config(args.model_config)
        self.model = MSATransformer(**model_config)
        self.model = RNAMsmForSeqCls(self.model, class_num=2)
        self.model._load_pretrained_bert(
            args.model_path)
        self.model.to(self.device)


class DNABERTEvaluator(RBPPredEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, num_labels=13).to(self.device)
        self.model = DNABERTForSeqCls(self.model)


class SpliceBERTEvaluator(DNABERTEvaluator):
    # SpliceBERT和DNABERT结构相同，权重不同
    def __init__(self, args, tokenizer) -> None:
        super().__init__(args, tokenizer=tokenizer)
