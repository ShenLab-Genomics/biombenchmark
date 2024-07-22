import tqdm
import torch
from torch.utils.data import DataLoader
from model.SpTransformer.sptransformer import Ex2


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


class SpTransformerEvaluator(SpliceEvaluator):

    def __init__(self, model_weight='', tissue_num=15) -> None:
        super().__init__()
        self.model = Ex2(128, context_len=4000, tissue_num=tissue_num,
                         max_seq_len=8192, attn_depth=8, training=False)
        save_dict = torch.load(
            model_weight, map_location='cpu')
        self.model.load_state_dict(save_dict["state_dict"])
        self.model.to(self.device).eval()

    def run(self, data_loader: DataLoader):

        with torch.no_grad():
            y_true = []
            y_pred = []

            for inputs, labels, source in tqdm.tqdm(data_loader, mininterval=5):

                outputs = self.model(inputs)
                index = torch.where(torch.sum(labels[:, 1:3, :], dim=1) > 0)
                print(index[0], index[1])
                y_true.append(labels[index[0], :, index[1]].cpu().detach())
                y_pred.append(outputs[index[0], :, index[1]].cpu().detach())

        # TODO: evaluate
        pass

    def evaluate(self):
        raise NotImplementedError
        auprc, pearson, precision, recall, _ = None
        return auprc, pearson, precision, recall, _
