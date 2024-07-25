import os
import tqdm
import torch
import numpy as np
import shutil
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification
from model.SpTransformer.sptransformer import Ex2
from model.utils import make_logger
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split


class SpliceEvaluator:
    """
    Splicing任务需要设置tissue_num，以决定在包含多少个tissue数据的数据集中测试
    暂定分三个任务： tissue_num = 0, 15, 53, 即不考虑组织特异性，考虑15个主要组织的特异性，以及考虑全部GTEx组织的特异性
    """

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        pass

    def get_topl_statistics(self, y_pred, y_true):
        '''
        Parameters
        ---
            y_true:     1-dim array, label for one of the tissues
            y_pred:     1-dim array, model output for the tissue

        Returns
        ---
            topkl_accuracy: float, the calculated topkl accuracy
            auprc:      float , the calculated pearson value
            pearson:    tuple , the calculated pearson correlation
        '''
        idx_true = np.nonzero(y_true >= 0.5)[0]
        argsorted_y_pred = np.argsort(y_pred)
        sorted_y_pred = y_pred[argsorted_y_pred]

        threshold = []
        # Get top-1L index
        idx_pred = argsorted_y_pred[-int(1 * len(idx_true)):]
        if len(idx_true) <= 0:
            raise ValueError("No positive data!")

        topkl_accuracy = np.size(np.intersect1d(
            idx_true, idx_pred)) / float(min(len(idx_pred), len(idx_true)))
        threshold = sorted_y_pred[-int(1 * len(idx_true))]

        auprc = average_precision_score(y_true >= 0.5, y_pred)
        pearson = pearsonr(y_true, y_pred)

        return topkl_accuracy, auprc, pearson

    def _evaluate(self, y_pred, y_true, tissue_num=15):
        topkl = []
        auprc = []
        pearson = []

        if tissue_num == 0:
            # No tissue, just distinguish acceptor/donor
            pass
        elif tissue_num == 15:
            # Usage of 15 tissues
            for idx in range(15):
                r1, r2, r3 = self.get_topl_statistics(
                    y_pred[idx, :], y_true[idx, :])
                topkl.append(r1)
                auprc.append(r2)
                pearson.append(r3)

        elif tissue_num == 54:
            # Usage of 54 tissues
            pass

        return topkl, auprc, pearson


class SpTransformerEvaluator(SpliceEvaluator):

    def __init__(self, model_weight='', tissue_num=15) -> None:
        super().__init__()
        self.tissue_num = tissue_num
        if tissue_num == 15:
            self.model = Ex2(128, context_len=4750, tissue_num=tissue_num,
                             max_seq_len=8192, attn_depth=8, training=False)
        elif tissue_num == 0:
            raise NotImplementedError
        elif tissue_num == 54:
            raise NotImplementedError

        save_dict = torch.load(
            model_weight, map_location='cpu')
        self.model.load_state_dict(save_dict["state_dict"])
        self.model.to(self.device).eval()

    def run(self, data_loader: DataLoader):

        with torch.no_grad():
            y_true = []
            y_pred = []
            cnt = 0
            for inputs, labels in tqdm.tqdm(data_loader, mininterval=5):
                cnt += 1
                if cnt > 200:
                    break
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                outputs[:, :3, :] = torch.softmax(outputs[:, :3, :], dim=1)
                outputs[:, 3:, :] = torch.sigmoid(outputs[:, 3:, :])
                is_expr = (labels[:, 1:3, :].sum(axis=(1, 2)) >= 1)
                y_true.append(labels[is_expr, :, :].cpu())
                y_pred.append(outputs[is_expr, :, :].cpu())

                # index = torch.where(torch.sum(labels[:, 1:3, :], dim=1) > 0)
                # y_true.append(labels[index[0], :, index[1]].cpu().detach())
                # y_pred.append(outputs[index[0], :, index[1]].cpu().detach())

        y_pred = torch.concat(y_pred, dim=0)
        y_true = torch.concat(y_true, dim=0)
        y_pred = y_pred.transpose(1, 2).reshape(-1, y_pred.shape[-1]).numpy()
        y_true = y_true.transpose(1, 2).reshape(-1, y_true.shape[-1]).numpy()
        # TODO: evaluate
        perform = self.evaluate(y_pred, y_true)
        return perform

    def evaluate(self, y_pred, y_true):
        if self.tissue_num == 15:
            y_pred = y_pred[:, 3:]
            y_true = y_true[:, 3:]
        topkl, auprc, pearson = self._evaluate(
            y_pred, y_true, tissue_num=self.tissue_num)
        # auprc, pearson, precision, recall, _ = None
        performance_dict = {
            'TopL': topkl,
            'PRAUC': auprc,
            'pearsonr': pearson
        }
        return performance_dict


class SpliceBERTEvaluator(SpliceEvaluator):
    """
    The code was adapted from https://github.com/chenkenbio/SpliceBERT?tab=readme-ov-file#how-to-use-splicebert
    """

    def __init__(self, train=False, tokenizer=None, model_path='', tissue_num=15) -> None:
        super().__init__()
        # set the path to the folder of pre-trained SpliceBERT
        self.SPLICEBERT_PATH = model_path
        # load tokenizer
        self.tokenizer = tokenizer
        self.tissue_num = tissue_num

    def run(self):

        # prepare input sequence
        seq = "ACGUACGuacguaCGu"  # WARNING: this is just a demo. SpliceBERT may not work on sequences shorter than 64nt as it was trained on sequences of 64-1024nt in length
        # U -> T and add whitespace
        seq = ' '.join(list(seq.upper().replace("U", "T")))
        # N -> 5, A -> 6, C -> 7, G -> 8, T(U) -> 9. NOTE: a [CLS] and a [SEP] token will be added to the start and the end of seq
        input_ids = self.tokenizer.encode(seq)
        input_ids = torch.as_tensor(input_ids)  # convert python list to Tensor
        # add batch dimension, shape: (batch_size, sequence_length)
        input_ids = input_ids.unsqueeze(0)

        # use huggerface's official API to use SpliceBERT
        # get nucleotide embeddings (hidden states)
        model = AutoModel.from_pretrained(self.SPLICEBERT_PATH)  # load model
        # get hidden states from last layer
        last_hidden_state = model(input_ids).last_hidden_state
        # hidden states from the embedding layer (nn.Embedding) and the 6 transformer encoder layers
        hiddens_states = model(
            input_ids, output_hidden_states=True).hidden_states

        # get nucleotide type logits in masked language modeling
        model = AutoModelForMaskedLM.from_pretrained(
            self.SPLICEBERT_PATH)  # load model
        # shape: (batch_size, sequence_length, vocab_size)
        logits = model(input_ids).logits

        # finetuning SpliceBERT for token classification tasks
        # assume the class number is 3, shape: (batch_size, sequence_length, num_labels)
        model = AutoModelForTokenClassification.from_pretrained(
            self.SPLICEBERT_PATH, num_labels=3)

    def train(self, args, train_data, test_data):
        # adapted from https://github.com/chenkenbio/SpliceBERT/blob/main/examples/04-splicesite-prediction/train_splicebert_cv.py
        """
        args : resume, debug, model_path, patience
        """
        OUT_DIR = 'model/fine_tuned/Splicing/SpliceBERT'
        best_auc = -1
        best_epoch = -1
        fold_ckpt = dict()
        logger = make_logger(log_file=os.path.join(
            OUT_DIR, "train.log"), level="DEBUG" if args.debug else "INFO")

        # Split train test
        train_inds = np.arange(len(train_data))
        train_inds, val_inds = train_test_split(
            train_inds, test_size=0.05, random_state=0)
        test_inds = np.arange(len(test_data))
        if args.debug:
            train_inds = train_inds[:100]
            val_inds = val_inds[:100]
            test_inds = np.random.permutation(test_inds)[:100]
        ##
        for epoch in range(200):
            fold = 0
            epoch_val_auc = list()
            epoch_val_f1 = list()
            epoch_test_auc = list()
            epoch_test_f1 = list()
            # setup dataset
            fold_outdir = os.path.join(OUT_DIR, "fold{}".format(fold))
            os.makedirs(fold_outdir, exist_ok=True)
            ckpt = os.path.join(fold_outdir, "checkpoint.pt")

            train_loader = DataLoader(
                Subset(train_data, indices=train_inds),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers
            )
            val_loader = DataLoader(
                Subset(train_data, indices=val_inds),
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            test_loader = DataLoader(
                test_data,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            # setup model, optimizer & scaler
            if epoch > 0 or (args.resume and os.path.exists(ckpt)):
                if epoch > 0:
                    del model, optimizer, scaler
                d = torch.load(ckpt)
                model = AutoModelForTokenClassification.from_pretrained(
                    args.model_path, num_labels=self.tissue_num).to(self.device)
                model.load_state_dict(d["model"])
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=1E-6
                )
                optimizer.load_state_dict(d["optimizer"])
                scaler = GradScaler()
                scaler.load_state_dict(d["scaler"])
                if epoch == 0:
                    trained_epochs = d.get("epoch", -1) + 1
            else:
                model = AutoModelForTokenClassification.from_pretrained(
                    args.model_path, num_labels=self.tissue_num).to(self.device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=1E-6
                )
                torch.save((train_inds, val_inds, test_inds),
                           "{}/split.pt".format(OUT_DIR))
                scaler = GradScaler()
                trained_epochs = 0

            model.train()
            # train
            pbar = tqdm.tqdm(train_loader,
                             total=len(train_loader),
                             desc="Epoch{}-{}".format(epoch +
                                                      trained_epochs, fold)
                             )
            epoch_loss = 0
            for it, (ids, mask, label) in tqdm.tqdm(enumerate(pbar)):
                ids, mask, label = ids.to(self.device), mask.to(
                    self.device), label.to(self.device).float()
                label = label.transpose(1, 2)[:, :, 3:]  # discard acc/don
                optimizer.zero_grad()
                with autocast():
                    logits = model.forward(
                        ids, attention_mask=mask).logits.squeeze(1)
                    # print(logits.shape)
                    logits = logits[:, 1:-1, :]  # discard [CLS] and [SEP]
                    if torch.isnan(logits).sum() > 0:
                        raise ValueError("NaN in logits: {}".format(
                            torch.isnan(logits).sum()))
                    loss = F.binary_cross_entropy_with_logits(
                        logits, label).mean()
                    if torch.isnan(loss).sum() > 0:
                        raise ValueError("NaN in loss: {}".format(
                            torch.isnan(loss).sum()))

                scaler.scale(loss).backward()
                scaler.step(optimizer)  # 0.step()
                scaler.update()

                epoch_loss += loss.item()

                pbar.set_postfix_str("loss/lr={:.4f}/{:.2e}".format(
                    epoch_loss / (it + 1), optimizer.param_groups[-1]["lr"]
                ))

            # validate
            val_auc, val_f1, val_score, val_label = self.test_model(
                model, val_loader)
            # torch.save((val_score, val_label),
            #            os.path.join(fold_outdir,  "val.pt"))
            epoch_val_auc.append(val_auc)
            epoch_val_f1.append(val_f1)
            test_auc, test_f1, test_score, test_label = self.test_model(
                model, test_loader)
            # torch.save((test_score, test_label),
            #            os.path.join(fold_outdir,  "test.pt"))
            epoch_test_auc.append(test_auc)
            epoch_test_f1.append(test_f1)
            logger.info("validate/test({}-{})AUC/F1: {:.4f} {:.4f} {:.4f} {:.4f}".format(
                epoch, fold, val_auc, val_f1, test_auc, test_f1))

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch
            }, ckpt)

            logger.info("Epoch{}validation&test(AUC/F1): {:.4f} {:.4f} {:.4f} {:.4f}".format(
                epoch,
                np.mean(epoch_val_auc),
                np.mean(epoch_val_f1),
                np.mean(epoch_test_auc),
                np.mean(epoch_test_f1)
            ))

            if np.mean(epoch_val_auc) > best_auc:
                best_auc = np.mean(epoch_val_auc)
                ckpt = fold_ckpt[fold]
                shutil.copy2(ckpt, "{}.best_model.pt".format(ckpt))
                wait = 0
                logger.info(f"model saved, best={epoch}\n")
            else:
                wait += 1
                logger.info("wait{}\n".format(wait))
                if wait >= args.patience:
                    break

    def test_model(self, model: AutoModelForTokenClassification, loader: DataLoader):
        r"""
        Return:
        auc : float
        f1 : float
        pred : list
        true : list
        """
        model.eval()
        pred, true = list(), list()
        for it, (ids, mask, label) in enumerate(tqdm.tqdm(loader, desc="predicting", total=len(loader))):
            ids = ids.to(self.device)
            mask = mask.to(self.device)
            score = torch.softmax(model.forward(
                ids, attention_mask=mask).logits, dim=1)

            score = score[:, 1:-1, :].detach().cpu().numpy()
            del ids
            label = label.transpose(1, 2).numpy()  # discard acc/don
            pred.append(score.astype(np.float16))
            true.append(label.astype(np.float16))
        y_pred = np.concatenate(pred)
        y_true = np.concatenate(true)

        print(y_pred.shape)
        print(y_true.shape)

        if self.tissue_num == 15:
            y_pred = y_pred[:, :, :].reshape(-1)
            y_true = y_true[:, :, 3:].reshape(-1)

        auc_list = roc_auc_score(y_true.T >= 0.5, y_pred.T)
        f1 = f1_score(y_true.T >= 0.5, y_pred.T >= 0.5)

        return auc_list, f1, pred, true
