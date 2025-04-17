import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
import numpy as np
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from evaluator.base_evaluator import BaseMetrics
from evaluator.seq_cls_evaluator import SeqClsTrainer, SeqClsCollator, SeqClsEvaluator, seq2kmer
# from model.wrap_for_mrl import RNAFmForReg, RNAErnieForReg, PureReg, RNAMsmForReg, RNABERTForReg, weights_init, DNABERT2ForReg, DNABERTForReg, Optimus, NTForReg, RNAErnieForRegAB
import model.RNAFM.fm as fm
from model.RNABERT.bert import get_config
from model.RNABERT.rnabert import BertModel
from model.RNAMSM.model import MSATransformer
from model.GENA import modeling_bert as GENA
from model import wrap_models, wrap_for_mrl
import scipy.stats as stats


class MRLMetrics(BaseMetrics):
    def __call__(self, outputs, labels, epoch=0):
        """
        Args:
            outputs: logits in tensor
            labels: labels in tensor
        Returns:
            metrics in dict
        """
        # regression model
        logits = outputs.cpu().numpy().reshape(-1).astype('float')
        labels = labels.cpu().numpy().reshape(-1).astype('float')

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
        return root_mean_squared_error(labels, logits)**2

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
                 replace_T=True, replace_U=False, use_kmer=True, pad_token_id=0):
        super(MRLCollator, self).__init__(max_seq_len, tokenizer, None,
                                          replace_T, replace_U, use_kmer, pad_token_id)

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
            input_ids = self.tokenizer(input_text)["input_ids"]
            if None in input_ids:
                raise ValueError("None in input_ids")
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

            input_ids = [self.pad_token_id] * self.max_seq_len + input_ids
            input_ids = input_ids[-self.max_seq_len:]

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
            max_seq_len=args.max_seq_len,
            tokenizer=self.tokenizer,
            replace_T=args.replace_T,
            replace_U=args.replace_U,
            use_kmer=args.use_kmer,
            pad_token_id=args.pad_token_id)
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
            # record performance on train set to check overfitting
            self.seq_cls_trainer.eval(i_epoch, info="Train_set")
            self.seq_cls_trainer.eval(i_epoch)
            if (i_epoch == 0) or ((i_epoch+1) % 5 == 0):
                try:
                    self.seq_cls_trainer.save_model(
                        f'{args.output_dir}/{args.method}', i_epoch)
                except Exception as e:
                    print(e)
                    print("Failed to save model.")


class RNAFMEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model, alphabet = fm.pretrained.rna_fm_t12(args.model_path)
        self.model = wrap_models.RNAFmForReg(
            self.model, freeze_base=args.freeze_base)
        self.model.to(self.device)


class RNAErnieEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer=None) -> None:
        super().__init__(tokenizer)
        self.model = AutoModel.from_pretrained(args.model_path)
        self.model = wrap_models.RNAErnieForReg(
            self.model, freeze_base=args.freeze_base).to(self.device)


# class RNAErnieEvaluatorAB(MRLEvaluator):
#     def __init__(self, args, tokenizer=None) -> None:
#         super().__init__(tokenizer)
#         self.model = AutoModel.from_pretrained(args.model_path)
#         self.model = wrap_models.RNAErnieForRegAB(self.model).to(self.device)


class RNAMsmEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        model_config = get_config(args.model_config)
        self.model = MSATransformer(**model_config)
        self.model = wrap_models.RNAMsmForReg(
            self.model, class_num=1, freeze_base=args.freeze_base)
        self.model._load_pretrained_bert(
            args.model_path)
        self.model.to(self.device)


class RNABertEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        # ========== Build tokenizer, model, criterion
        model_config = get_config(args.model_config)
        self.model = BertModel(model_config)
        self.model = wrap_models.RNABERTForReg(
            self.model, class_num=1, freeze_base=args.freeze_base)
        self.model._load_pretrained_bert(args.model_path)
        self.model.to(self.device)


class DNABERTEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = AutoModel.from_pretrained(
            args.model_path)
        self.model = wrap_models.DNABERTForReg(
            self.model, freeze_base=args.freeze_base).to(self.device)


class DNABERT2Evaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        config = BertConfig.from_pretrained(args.model_path)
        self.model = AutoModel.from_pretrained(
            args.model_path, trust_remote_code=True, config=config)
        self.model = wrap_models.DNABERT2ForReg(
            self.model, freeze_base=args.freeze_base).to(self.device)


class NTEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer=None):
        from peft import LoraConfig, TaskType, get_peft_model
        super().__init__(tokenizer)
        self.model = AutoModelForMaskedLM.from_pretrained(
            args.model_path, trust_remote_code=True)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, inference_mode=False, r=1, lora_alpha=32, lora_dropout=0.1,
            target_modules=["query", "value"]
        )
        if not args.freeze_base:
            # when not freeze_base, we need to set the base model with lora
            self.model = get_peft_model(self.model, peft_config)
        self.model = wrap_models.NTForReg(
            self.model, freeze_base=args.freeze_base).to(self.device)


class GENAEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = GENA.BertForMaskedLM.from_pretrained(
            args.model_path)

        self.model = wrap_models.GENAForReg(
            self.model, freeze_base=args.freeze_base).to(self.device)


class RNABertEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        # ========== Build tokenizer, model, criterion
        model_config = get_config(args.model_config)
        self.model = BertModel(model_config)
        self.model = wrap_models.RNABERTForReg(
            self.model, freeze_base=args.freeze_base)
        self.model._load_pretrained_bert(args.model_path)
        self.model.to(self.device)


class ResNetEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        super().__init__(tokenizer=tokenizer)
        self.model = wrap_for_mrl.PureReg().to(self.device)


class OptimusEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer=None):
        super().__init__(tokenizer=tokenizer)
        self.model = wrap_for_mrl.Optimus(inp_len=100).to(self.device)


class UTRLMEvaluator(MRLEvaluator):
    def __init__(self, args, tokenizer) -> None:
        from model.UTRLM.utrlm import UTRLM
        super().__init__(tokenizer=tokenizer)
        self.model = UTRLM()
        # load model weights
        model_weights = torch.load(
            args.model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(
            {k.replace('module.', ''): v for k, v in model_weights.items()}, strict=True)

        self.model = wrap_models.UTRLMForReg(
            self.model, freeze_base=args.freeze_base).to(self.device)
