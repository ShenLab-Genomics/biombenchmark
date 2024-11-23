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
from model.BERT_like import RNABertForSeqCls, RNAMsmForSeqCls, RNAFmForSeqCls, SeqClsLoss
from model.wrap_for_cls import DNABERTForSeqCls, DNABERT2ForSeqCls, RNAErnieForSeqCls
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from seq_cls_evaluator import SeqClsMetrics, SeqClsCollator
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


class EmbMetrics(SeqClsMetrics):
    def __init__(self, metrics, save_path=None):
        super().__init__(metrics, save_path)

    # used this function independently
    def emb(self, preds, labels, epoch=0):
        preds = preds.cpu().numpy()
        fsave = os.path.join(
            self.save_path, f'epoch_{epoch}_{name}_{idx}')
        np.save(fsave, preds)


class EmbCollator(SeqClsCollator):
    pass


class EmbTrainer(BaseTrainer):
    def train(self, epoch):
        return super().train(epoch)

    def eval(self, epoch):
        return super().eval(epoch)

    def extract_emb(self, epoch):
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

        emb = EmbMetrics.emb(outputs_dataset, labels_dataset)
