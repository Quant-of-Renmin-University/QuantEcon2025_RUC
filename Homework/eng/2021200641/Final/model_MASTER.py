# MASTER模型复现by lyl
import gc
import pandas as pd
import numpy as np
import json
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchmetrics import PearsonCorrCoef
from argparse import ArgumentParser
import warnings
import shutil
import copy
import pickle
import math
from datetime import datetime
import time
from dateutil.relativedelta import relativedelta
import pytorch_lightning as pl

from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                            ModelCheckpoint,
                                            StochasticWeightAveraging)
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler
from merged_sector import merge_sector

warnings.filterwarnings("ignore")

cpu_num = 10
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
torch.autograd.set_detect_anomaly(True)


# 超参数设置
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=30)  # 最大轮数
    parser.add_argument('--batch_size', type=int, default=2)  # batch中样本含几天
    parser.add_argument('--gpus', default=[0])
    parser.add_argument('--strategy', default='ddp')
    parser.add_argument('--find_unused_parameters', default=False)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--check_test_every_n_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00003)  # 学习率
    parser.add_argument('--weight_decay', type=float, default=1e-2)  # weight_decay
    parser.add_argument('--seed', type=int, default=42)  # 随机种子
    parser.add_argument('--optimizer', default='adam',
                        choices=['adam', 'adamw'])
    parser.add_argument('--log_every_n_steps', type=int, default=1)
    parser.add_argument('--loss', default='wpcc')  # 损失函数
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--swa', action='store_true')  # swa设置
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint', help='path to checkpoints (for test)')
    args, unknown = parser.parse_known_args()
    return args

args = parse_args()

# 模型文件存储目录
root_path = rf'/home/leanon'
# 数据存储目录
data_path = rf'/project'
# 因子
fac_path = rf'{data_path}/factor_data'
fac_name = rf'fac20250212'
fac_set = "factorV1"
# 标签
label_path = rf'{data_path}/label_data'
label_name = rf'label1'
# 流动性数据
liquid_path = rf'{data_path}/label_data'
liquid_name = rf'can_trade_amt1'


class params:
    # 模型前缀
    model_prefix = "MASTER"
    model_path = rf'{root_path}/{model_prefix}'
    profiler_path = rf'{root_path}/logs'
    # 模型前缀
    liquid_data = pd.read_feather(rf"{liquid_path}/{liquid_name}.fea").set_index("index")
    ret_data = pd.read_feather(rf"{label_path}/{label_name}.fea").set_index("index")
    # 是否使用dropout
    dropout = True
    # dropout率
    dropout_rate = 0.2
    # 因子标准化方法
    normed_method = 'zscore'
    # time window
    n_days = 7 # 不要超过10，不然有data leakage
    
def get_basic_name():
    """Basic Name 
    [将model_prefix、fac_name、label_name连接起来, 中间用--连接]

    Returns:
        str: Basic Name
        
    Example:
    >>> get_basic_name()
    'nn--fac20240819--label1'
    """
    name = rf'{params.model_prefix}--{fac_set}--{label_name}'
    if params.dropout:
        name += rf'--dropout{params.dropout_rate}'
    return name

def get_name(test_time:str, valid_period:str, market_name:str):
    """ Model Name [将market_name、test_time、valid_period连接起来, 中间用-连接]
    
    Args:
        test_time (str): test_time
        valid_period (str): valid_period
        market_name (str): market_name

    Returns:
        str: Model Name
        
    Example:
    >>> get_name("2023q1", "validperiod1", "ALL")
    'ALL-2023q1-validperiod1-Deep_Wide_V1'
    """

    return '-'.join(filter(None, [
        f'{market_name}{test_time}',
        f'{valid_period}',
        f'{get_basic_name()}',
    ])).replace(' ', '')


def normed_data(data, date, stage, normed_method=params.normed_method):
    """
    对当天(截面)的因子数据进行标准化
    """

    # 重置索引，将原有索引变为普通列
    data = data.reset_index()
    # 只保留因子列非空值数量大于10%（即有效数据较多）的行
    data = data.reindex(data[params.factor_list].dropna(thresh=int(0.1 * len(params.factor_list))).index)
    # 将'date'列设为索引
    data = data.set_index('date')
    # 获取当前日期的流动性数据
    liquid_data = params.liquid_data.loc[date]
    # 将流动性数据按股票代码对齐，添加到data中
    data['liquid'] = liquid_data.reindex(data["Code"]).values
    # 获取当前日期的标签（收益）数据
    ret_data = params.ret_data.loc[date]
    # 将标签数据按股票代码对齐，添加到data中
    data['Label'] = ret_data.reindex(data["Code"]).values
    # 如果是训练阶段，对标签做z-score标准化
    if stage == "train":
        data['Label'] = (data['Label'] - data['Label'].mean()) / data['Label'].std()
    # 标签中的缺失值用0填充
    data['Label'] = data['Label'].fillna(0)
    # 提取股票代码
    code_value = data['Code'].values
    # 获取股票代码对应的index
    code_index = torch.tensor([params.code2idx[code] for code in code_value], dtype=torch.long)
    # 提取中信行业(ind1, ind2)
    sector_value1 = data['ind1'].values
    sector_index1 = torch.tensor([params.sector1idx[sector] for sector in sector_value1], dtype=torch.long)
    sector_value2 = data['ind2'].values
    sector_index2 = torch.tensor([params.sector2idx[sector] for sector in sector_value2], dtype=torch.long)
    # 提取特征（去除'Code'、'Sector'、'Label'、'liquid'列）
    data_X = data.drop(['Code', 'ind1', 'ind2', 'Label', 'liquid'], axis=1)
    # 提取标签
    data_y = data['Label']
    # 提取流动性
    data_liquid = data['liquid']

    # 如果标准化方法为zscore
    if normed_method == 'zscore':
        # 对特征做去极值处理（按列分布的0.5%~99.5%分位数裁剪）
        data_X = np.clip(data_X, data_X.quantile(0.005), data_X.quantile(0.995), axis=1)
        # 对特征做z-score标准化
        data_X = (data_X - data_X.mean()) / data_X.std()
        # 标准化后缺失值用0填充
        data_X = data_X.fillna(0)
    else:
        # 其他标准化方法暂未实现
        raise NotImplementedError

    # 返回：特征张量、标签张量、股票代码、流动性张量
    return torch.tensor(data_X.values, dtype=torch.float32), \
        torch.tensor(data_y.values, dtype=torch.float32), \
        code_value, code_index, sector_index1, sector_index2, torch.tensor(data_liquid.values, dtype=torch.float32)

# 组装batch
def collate_fn(datas):
    # datas: list of batch_size个样本，每个样本是 (tsdata_list, label_list, ...)
    # 现在 tsdata_list 是 [n_days, stock, feature]
    # 目标：输出 (batch, n_days, stock, feature)

    # 拆包
    tsdata_lists, label_list, date_lists, code_value_lists, code_index_lists, sector_index1_lists, sector_index2_lists, liquid_list, mask_lists = zip(*datas)

    # tsdata_lists: tuple of batch_size个 [n_days, stock, feature] 的list
    # 先把每个样本的 tsdata_list 堆叠成 tensor: (n_days, stock, feature)
    tsdata = torch.stack([torch.stack(tsdata_list) for tsdata_list in tsdata_lists])  # (batch, n_days, stock, feature)

    # 其它同理
    labels = torch.stack([label for label in label_list]) # (batch, stock)
    code_index = torch.stack([torch.stack(code_index_list) for code_index_list in code_index_lists])
    sector_index1 = torch.stack([torch.stack(sector_index1_list) for sector_index1_list in sector_index1_lists])
    sector_index2 = torch.stack([torch.stack(sector_index2_list) for sector_index2_list in sector_index2_lists])
    liquids = torch.stack([liquid for liquid in liquid_list])
    mask = torch.stack([torch.stack(mask_list) for mask_list in mask_lists])

    # date和sorted_codes是list或str，直接合并成list
    date = list(date_lists)
    code_value = list(code_value_lists)

    return tsdata, labels, date, code_value, code_index, sector_index1, sector_index2, liquids, mask


class DLDataset(torch.utils.data.Dataset):

    def __init__(self, date_list, stage='train'):
        self.date_list = date_list
        self.stage = stage

    def __getitem__(self, index):
        """
        这里的index就相当于是第几天
        取出包括index当天的前n天数据
        """
        date = self.date_list[index]
        idx_in_all = params.date_list.index(date)
        # 取前n_days天的数据(包括当天)
        start_idx = idx_in_all - params.n_days + 1
        dates = params.date_list[start_idx:idx_in_all+1]
        
        tsdata_list = []           # 用于存放特征张量
        label_list = []           # 用于存放标签张量
        time_list = []        # 用于存放日期信息
        code_value_list = []  # 用于存放股票代码
        code_index_list = []  # 用于存放股票代码对应的index
        sector_index1_list = []  # 用于存放股票代码对应的sector index
        sector_index2_list = []  # 用于存放股票代码对应的sector index
        liquid_list = []      # 用于存放流动性张量
        mask_list = []      # 用于存放mask张量
        
        # 遍历每一天
        for d in dates:
            data = params.all_data.loc[d].copy()  # 取出日期对应的全部因子数据，并复制一份防止原数据被修改
            # 对该日期的数据进行标准化处理，返回特征、标签、股票代码、流动性
            data_X, data_y, code_value, code_index_valid, sector_index1_valid, sector_index2_valid, data_liquid = normed_data(data, date, stage=self.stage)

            # 初始化全集股票池的全0张量
            tsdata_full = torch.zeros((params.num_codes, params.factor_num), dtype=torch.float32)

            code_index_full = torch.arange(params.num_codes, dtype=torch.long)
            sector_index1_full = torch.zeros(params.num_codes, dtype=torch.long)
            sector_index2_full = torch.zeros(params.num_codes, dtype=torch.long)
            mask_full = torch.zeros(params.num_codes, dtype=torch.bool) # 掩码矩阵
            
            # 只在最后一天算label和liquid
            if d == date:
                label_full = torch.zeros(params.num_codes, dtype=torch.float32)
                liquid_full = torch.zeros(params.num_codes, dtype=torch.float32)
                
            # 填充有效股票
            for i, code in enumerate(code_value):
                idx = params.code2idx[code]
                tsdata_full[idx] = data_X[i]
                code_index_full[idx] = code_index_valid[i]
                sector_index1_full[idx] = sector_index1_valid[i]
                sector_index2_full[idx] = sector_index2_valid[i]
                mask_full[idx] = True # 有效的掩码为True!!!!!!!
                if d == date:
                    label_full[idx] = data_y[i]
                    liquid_full[idx] = data_liquid[i]

            tsdata_list.append(torch.nan_to_num(tsdata_full, nan=0, posinf=0, neginf=0))
            
            time_list.append(date)
            code_value_list.append(code_value)
            code_index_list.append(code_index_full)
            sector_index1_list.append(sector_index1_full)
            sector_index2_list.append(sector_index2_full)
            mask_list.append(mask_full)
            if d == date:
                label_list.append(torch.nan_to_num(label_full, nan=0, posinf=0, neginf=0))
                liquid_list.append(torch.nan_to_num(liquid_full, nan=0, posinf=0, neginf=0))

        # 返回n_days天的数据，每个元素shape和原来一样
        return tsdata_list, label_list[-1], time_list, code_value_list, code_index_list, sector_index1_list, sector_index2_list, liquid_list[-1], mask_list
    
    def __len__(self):
        return len(self.date_list)


class DLDataModule(pl.LightningDataModule):
    def __init__(self, args, train_date_list, valid_date_list, test_date_list):
        super().__init__()
        self.args = args
        self.tr = DLDataset(train_date_list, stage='train')
        self.val = DLDataset(valid_date_list, stage='valid')
        self.test = DLDataset(test_date_list, stage='test')

    def train_dataloader(self):
        return DataLoader(self.tr, batch_size=self.args.batch_size, collate_fn=collate_fn,
                            num_workers=8, 
                            shuffle= True,
                            persistent_workers=False,
                            drop_last=False,
                            pin_memory=False)

    def _val_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                            num_workers=0, persistent_workers=False,
                            pin_memory=False, drop_last=False)

    def val_dataloader(self):
        return self._val_dataloader(self.val)

    def test_dataloader(self):
        return self._val_dataloader(self.test)


# 损失函数设置
def get_loss_fn(loss):
    def wpcc(preds, y, mask):
        """
        preds, y: (batch, stock, 1) or (batch, stock)
        返回：batch内每个样本的wpcc，最后取均值
        """
        if preds.dim() == 3:
            preds = preds.squeeze(-1)  # (batch, stock)
        if y.dim() == 3:
            y = y.squeeze(-1)          # (batch, stock)
        if mask.dim() == 3:
            mask = mask.squeeze(1)    # (batch, stock)
        batch_size = preds.shape[0]
        losses = []
        for i in range(batch_size):
            pred = preds[i].flatten()  # (stock,)
            label = y[i].flatten()     # (stock,)
            valid_mask = mask[i].flatten()
            pred = pred[valid_mask]
            label = label[valid_mask]
            # 原来的wpcc逻辑
            _, argsort = torch.sort(label, descending=True, dim=0)
            weight = torch.zeros_like(pred)
            weight_new = torch.tensor([0.5 ** ((j - 1) / (pred.shape[0] - 1)) for j in range(1, pred.shape[0] + 1)],
                                    device=pred.device).unsqueeze(dim=1 if pred.ndim > 1 else 0)
            weight[argsort] = weight_new.squeeze()
            wcov = (pred * label * weight).sum() / weight.sum() - \
                (pred * weight).sum() / weight.sum() * (label * weight).sum() / weight.sum()
            pred_std = torch.sqrt(((pred - pred.mean()) ** 2 * weight).sum() / weight.sum() + 1e-8)
            label_std = torch.sqrt(((label - label.mean()) ** 2 * weight).sum() / weight.sum() + 1e-8)
            loss = -(wcov / (pred_std * label_std))

            losses.append(loss)
        return torch.stack(losses).mean()
    
    def liq_wpcc(preds, y, liquid, alpha=40.0):
        """
        preds, y, liquid: (batch, stock, 1) 或 (batch, stock)
        返回：batch内每个样本的liq_wpcc，最后取均值
        """
        if preds.dim() == 3:
            preds = preds.squeeze(-1)  # (batch, stock)
        if y.dim() == 3:
            y = y.squeeze(-1)
        if liquid.dim() == 3:
            liquid = liquid.squeeze(-1)
        batch_size = preds.shape[0]
        losses = []
        for i in range(batch_size):
            pred = preds[i]    # (stock,)
            label = y[i]       # (stock,)
            liq = liquid[i]    # (stock,)

            # 先对liquid取log1p
            liquid_log = torch.log1p(liq)
            # 再做min-max归一化
            liquid_min = liquid_log.min()
            liquid_max = liquid_log.max()
            # 防止分母为0
            liquid_norm = (liquid_log - liquid_min) / (liquid_max - liquid_min + 1e-8)
            # 权重限制在0.5~1
            weight = 0.5 * torch.sigmoid(alpha * liquid_norm * label) + 0.5

            wcov = (pred * label * weight).sum() / weight.sum() - \
                   (pred * weight).sum() / weight.sum() * (label * weight).sum() / weight.sum()
            pred_std = torch.sqrt(((pred - pred.mean()) ** 2 * weight).sum() / weight.sum())
            label_std = torch.sqrt(((label - label.mean()) ** 2 * weight).sum() / weight.sum())
            loss = -(wcov / (pred_std * label_std))
            losses.append(loss)
        return torch.stack(losses).mean()
    
    def output(loss):
        return {
            'wpcc': wpcc,
            'liq_wpcc': liq_wpcc,
        }[loss]

    return output(loss)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, feature_dim, time_window):
        super().__init__()
        pe = torch.zeros(time_window, feature_dim)  # (T, D)
        position = torch.arange(0, time_window, dtype=torch.float).unsqueeze(1)  # (T, 1)
        div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(10000.0) / feature_dim))  # (D/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        pe = pe.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (B, T, N, D)
        return: positional encoding of shape (B, T, N, D)
        """
        T = x.size(1)
        return self.pe[:, :T, :, :].expand_as(x)

class PredictModel(nn.Module):
    """
    tsdata: 特征张量，shape: (batch_size, n_days, stock, feature)
    code_index: 股票代码对应的index，shape: (batch_size, n_days, stock)
    sector_index1: 股票代码对应的sector index，shape: (batch_size, n_days, stock)
    sector_index2: 股票代码对应的sector index，shape: (batch_size, n_days, stock)
    mask: 掩码矩阵，shape: (batch_size, n_days, stock)
    """
    def __init__(self, args):
        super(PredictModel, self).__init__()
        input_dim = params.factor_num
        self.code_embedding = nn.Embedding(params.num_codes, 64)
        self.sector1_embedding = nn.Embedding(params.num_sectors1+1, 8, padding_idx=0) #加1！！！因为0是padding_idx
        self.sector2_embedding = nn.Embedding(params.num_sectors2+1, 16, padding_idx=0)
        embedding_dim = 64 + 8 + 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        # positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(feature_dim=512, time_window=params.n_days)
        # layer norm
        self.layer_norm = nn.LayerNorm(512)
        # intro-stock attention
        self.intra_stock_attn = nn.MultiheadAttention(embed_dim=512, num_heads=16, batch_first=True)
        self.FFN = nn.Sequential(
             nn.Linear(512, 512),
             nn.ReLU(),
             nn.Dropout(0.3),
             )
        self.inter_stock_attn = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)
        self.temporal_aggregation = nn.Sequential(
            nn.Linear(512, 512, bias=False),
        )
        self.final_fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        # He初始化, Var(w) = 2/n，适合activation function为ReLU的情况
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, tsdata, code_index, sector_index1, sector_index2, mask):
        tsdata = tsdata.float() # tsdata: (batch_size, n_days, stock, feature)
        
        B, T, N = mask.shape  # batch_size, n_days, num_codes
        code_embedding = self.code_embedding(code_index)  # (B, T, N, 64)
        sector1_embedding = self.sector1_embedding(sector_index1)  # (B, T, N, 8)
        sector2_embedding = self.sector2_embedding(sector_index2)  # (B, T, N, 16)
        panel_feature = torch.cat([tsdata, code_embedding, sector1_embedding, sector2_embedding], dim=-1) # (B, T, N, feature + 64 + 8 + 16)
        # Module1.Macro Gating for feature selection
        
        # Module2. intra-stock aggregation
        panel_feature = self.encoder(panel_feature) # (B, T, N, 512)
        panel_feature = panel_feature + self.pos_encoding(panel_feature)
        panel_feature = self.layer_norm(panel_feature) # (B, T, N, 512)
        # mask掉padding的股票
        padding_mask_infra = (mask==0).permute(0, 2, 1).reshape(B*N, T)
        # intra-stock aggregation
        x_ = panel_feature.permute(0, 2, 1, 3).reshape(B*N, T, 512)
        x_, _ = self.intra_stock_attn(x_, x_, x_, key_padding_mask=padding_mask_infra)
        x_ = x_.reshape(B, N, T, 512)
        x_ = x_.permute(0, 2, 1, 3)
        intra_feature = panel_feature + x_ # 残差连接
        intra_feature = self.FFN(intra_feature) + intra_feature
        intra_feature = self.FFN(intra_feature) + intra_feature # (B, T, N, 512)
        
        # Module3. inter-stock aggregation
        inter_x = intra_feature.reshape(B*T, N, 512)
        padding_mask_inter = (mask==0).reshape(B*T, N)
        inter_x, _ = self.inter_stock_attn(inter_x, inter_x, inter_x, key_padding_mask=padding_mask_inter)
        inter_x = inter_x.reshape(B, T, N, 512)
        inter_feature = intra_feature + inter_x # 残差连接
        inter_feature = self.FFN(inter_feature) + inter_feature
        inter_feature = self.FFN(inter_feature) + inter_feature # (B, T, N, 512)
        
        # Module4. temporal aggregation
        temporal_feature = inter_feature.permute(0, 2, 1, 3).reshape(B*N, T, 512)
        query = temporal_feature[:, -1:, :] # (B*N, 1, 512)
        key_value = temporal_feature # (B*N, T, 512)
        proj = self.temporal_aggregation(key_value)    # (B*N, T, 512)
        attn_score = torch.matmul(proj, query.transpose(-1, -2)).squeeze(-1)  # (B*N, T)
        attn_score = torch.softmax(attn_score, dim=1)  # (B*N, T)
        eu = torch.sum(attn_score.unsqueeze(-1) * temporal_feature, dim=1)  # (B*N, 512)
        eu = eu.reshape(B, N, 512) # (B, N, 512)
        
        # Module5. prediction
        pred = self.final_fc(eu)
        pred = pred.squeeze(-1) # (B, N)
        return pred
        
# 训练，验证和测试步骤
class DLLitModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = PredictModel(args)
        self.test_pearson = PearsonCorrCoef()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        loss_fn = get_loss_fn(self.args.loss)
        tsdatas, labels, times, code_values, code_indexes, sector_indexes1, sector_indexes2, liquids, masks = batch
        preds = self.forward(tsdatas, code_indexes, sector_indexes1, sector_indexes2, masks)
        mask = masks[:, -1, :] # (batch_size, 1, stock)
        print("preds nan:", torch.isnan(preds).any())
        print("preds all nan:", torch.isnan(preds).all())
        if self.args.loss in ['liq_wpcc']:
            loss = loss_fn(preds, labels, liquids,mask)
        else:
            loss = loss_fn(preds, labels,mask)

        self.log('train_loss', loss, prog_bar=True)       
        return loss

    def _evaluate_step(self, batch, batch_idx, stage):
        """对验证集或测试集的一个batch进行评估，计算超额收益和IC，并在测试阶段保存预测结果。

        Args:
            batch (tuple): 一个batch的数据，包含特征、标签、时间、股票代码、流动性等信息。
            batch_idx (int): 当前batch的索引。
            stage (str): 当前阶段，'val'表示验证，'test'表示测试。
            
        Returns:
            list: [平均超额收益, 平均IC]，分别为当前batch内所有天的超额收益均值和IC均值
        """
        # 其实不是超额收益，只是流动性加权收益
        def get_excess_return(preds, ret, liquid, money):
            # 计算超额收益：按预测值排序，依次买入流动性允许的前500只股票，计算总收益/总资金
            _, sort = preds.sort(dim=0, descending=True, stable=True)
            total_hold = torch.tensor(0.0, device=preds.device) # 累计已买入的资金
            total_earned = torch.tensor(0.0, device=preds.device) # 累计获得的收益
            for num, idx in enumerate(sort):
                if num >= 500:
                    break
                if (money - total_hold) < 1:
                    break
                hold_money = min(money - total_hold, liquid[idx])
                total_hold += hold_money
                total_earned += ret[idx] * hold_money
            total_ret = total_earned / money
            return total_ret
        
        excess_return_list = [] # 存放每一天的超额收益
        ic_list = [] # 存放每一天的IC
        model_cpu = copy.deepcopy(self.model)
        model_cpu = model_cpu.to('cpu')
        model_cpu.eval() # 评估模式（不启用dropout等）
        save_model = False
        tsdatas, rets, times, code_values, code_indexes, sector_indexes1, sector_indexes2, liquids, masks = batch # 解包batch，得到特征、标签、时间、股票代码、股票代码对应的index、股票代码对应的sector index、流动性
        # 第一次遇到"out_sample"标识后，保存模型。目前来说，不会有"out_sample"出现，所以保存模型的行为都是下面else段产生的
        if tsdatas == 'out_sample':
            if not save_model:
                try:
                    torch.jit.script(model_cpu).save(f"{params.model_name}.pth")
                    save_model = True
                except:
                    pass
        else:
            preds = self.forward(tsdatas, code_indexes, sector_indexes1, sector_indexes2, masks)
            if stage == "test":
                # 测试阶段，保存预测结果为pkl文件
                data_cpu = tsdatas.to('cpu')
                code_index_cpu = code_indexes.to('cpu')
                sector_index1_cpu = sector_indexes1.to('cpu')
                sector_index2_cpu = sector_indexes2.to('cpu')
                mask_cpu = masks.to('cpu')
                preds_cpu = model_cpu.forward(data_cpu, code_index_cpu, sector_index1_cpu, sector_index2_cpu, mask_cpu)
                preds_cpu = preds_cpu.squeeze(0) # (stock,) # batch是1，所以squeeze(0)
                # res(result)模型的预测结果
                valid_codes = np.array(code_values)[mask_cpu.cpu().numpy()]
                res = pd.DataFrame(preds_cpu.cpu().detach().numpy(), index=valid_codes, columns=['value'])
                res.index.name = 'Code'
                res.to_pickle(f'{params.test_save_path}/{time}.pkl')
                    # 只保存一次
                if not save_model:
                    try:
                        torch.jit.script(model_cpu).save(f"{params.model_name}.pth")
                        save_model = True
                    except:
                        pass
            # 获得流动性加权收益和IC
            preds = preds.squeeze(0)
            valid_mask = masks[:, -1, :].squeeze(0) # (stock,)
            liquid = liquids.squeeze(0) # (stock,)
            ret = rets.squeeze(0) # (stock,)
            preds = preds[valid_mask]
            ret = ret[valid_mask]
            liquid = liquid[valid_mask]
            
            excess_return = get_excess_return(preds, ret, liquid, money=1.5e9)
            excess_return_list.append(excess_return)
            ic_list.append(self.test_pearson(preds, ret))
        try:
            res_list = [sum(excess_return_list) / len(excess_return_list), sum(ic_list) / len(ic_list)]
        except:
            res_list = [np.nan, np.nan]
        if stage == 'val':
            self.validation_step_outputs.append(res_list)
        if stage == "test":
            self.test_step_outputs.append(res_list)
        # 返回list[平均超额收益, 平均IC]
        return res_list

    def test_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'test')

    def validation_step(self, batch, batch_idx):
        return self._evaluate_step(batch, batch_idx, 'val')

    def on_validation_epoch_end(self):
        val_step_outputs = self.validation_step_outputs # 拿到本轮所有batch的评估结果
        num_batch = len(val_step_outputs) # batch数量
        # 自定义的指标val_wei： 流动性加权收益 * 50 + IC 的均值
        avg_excess_return = sum([data[0] for data in val_step_outputs]) / num_batch
        avg_ic = sum([data[1] for data in val_step_outputs]) / num_batch
        self.log('val_wei', avg_excess_return * 50 + avg_ic, prog_bar=True, sync_dist=True)
        print(f"\nAvg_excess_return={avg_excess_return:.6f}, Avg_IC={avg_ic:.6f}")
    
        self.validation_step_outputs.clear() # 清空本轮的评估结果
        gc.collect() # 释放内存
    

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        num_batch = len(test_step_outputs)
        # test_wei： 仅仅是流动性加权收益
        self.log('test_wei', sum([data[0] for data in test_step_outputs]) / num_batch, prog_bar=True, sync_dist=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        kwargs = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay,
        }

        optimizer = {
            'adam': torch.optim.Adam(self.model.parameters(), **kwargs),
            'adamw': torch.optim.AdamW(self.model.parameters(), **kwargs),
        }[self.args.optimizer]

        optim_config = {
            'optimizer': optimizer,
        }

        return optim_config

    def configure_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            ModelCheckpoint(monitor='val_wei', mode='max', save_top_k=10, save_last=False,
                            filename='{epoch}-{val_wei:.4f}')
        ]
        if self.args.swa:
            callbacks.append(StochasticWeightAveraging(swa_epoch_start=0.7,
                                                       device='gpu'))
        if self.args.early_stop:
            callbacks.append(EarlyStopping(monitor='val_wei',
                                           mode='max', patience=7))
        return callbacks


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(path, data):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


# 模型训练
def train_single(args, name, seed, train_date_list, valid_date_list, test_date_list):
    # 设置PyTorch使用的线程数
    torch.set_num_threads(args.threads)
    # 固定随机种子，保证实验可复现
    seed_everything(seed)
    # 创建TensorBoard日志记录器，用于记录训练过程
    logger = TensorBoardLogger(save_dir=params.model_path, name=name)
    # 创建性能分析器，用于记录训练过程中的性能数据
    profiler = SimpleProfiler(dirpath=params.profiler_path, filename=name)
    # 构建Trainer参数字典，只保留Trainer支持的参数
    args_for_trainer = dict()
    for key, value in vars(args).items():
        try:
            Trainer(**{key: value})  # 测试该参数是否被Trainer支持
            args_for_trainer[key] = value
        except:
            pass
    # 实例化PyTorch Lightning的Trainer，传入所有参数、性能分析器、日志记录器，并设置可复现
    trainer = Trainer(**args_for_trainer,
                        num_sanity_val_steps=1,  # 训练前先跑1步验证，检查流程是否正常
                        profiler=profiler,
                        logger=logger,
                        deterministic=True)

    # 实例化自定义的Lightning模型
    litmodel = DLLitModule(args)
    # 实例化数据模块，传入训练、验证、测试日期
    dm = DLDataModule(args, train_date_list, valid_date_list, test_date_list)
    # 开始训练模型
    trainer.fit(litmodel, dm)
    # 获取训练过程中表现最好的模型权重路径
    best_ckpt = trainer.checkpoint_callback.best_model_path
    # 用最佳模型在测试集上评估
    test_result = trainer.test(ckpt_path=best_ckpt, datamodule=dm)
    # 打印测试结果
    print(test_result)


# 训练函数
def train(args, name, market, season, fac_data, state='train'):
    # ------------------------------------------------------------
    # 保存最终模型的路径
    save_path = rf"{root_path}/model_test/{get_basic_name()}" # example: /home/user78/model_test/nn--fac20240819--label1--dropout0.2
    try:
        os.makedirs(save_path, exist_ok=True)
        # 备份模型脚本——将当前脚本复制到model_test/get_basic_name()路径下做备份
        current_file_path = os.path.abspath(__file__)
        shutil.copy(current_file_path, save_path)
    except Exception as e:
        print(e)

    params.model_name = f"{save_path}/{name[:len(market) + 19]}"
    params.test_save_path = f"{save_path}/{name[:len(market)] + name[len(market) + 6:len(market) + 19]}"
    os.makedirs(params.test_save_path, exist_ok=True)
    
    # --------------------------------------------------------------
    print("【1】Loading Factor Data")
    # 读取因子
    with open(rf'{root_path}/FactorSelection/{fac_set}/fac_sel.json', "r") as f:
        fac_list = json.load(f)[season]
    print(f"[{datetime.now()}] 开始加载因子数据...")
    start_time = time.time() 
    params.all_data = fac_data[["Code", "date"] + fac_list]
    params.sorted_codes = sorted(params.all_data["Code"].unique().tolist()) # 对code排序
    params.all_data = merge_sector(params.all_data)

    # code embedding
    params.code2idx = {code: idx for idx, code in enumerate(params.sorted_codes)}
    params.num_codes = len(params.sorted_codes)
    
    # sector embedding
    sectors1 = params.all_data["ind1"].unique().tolist()
    sectors2 = params.all_data["ind2"].unique().tolist()
    params.sector1idx = {sector: idx+1 for idx, sector in enumerate(sectors1)}
    params.sector2idx = {sector: idx+1 for idx, sector in enumerate(sectors2)}
    params.num_sectors1 = len(sectors1)
    params.num_sectors2 = len(sectors2)
    
    end_time = time.time()
    print(f"[{datetime.now()}] 因子数据加载完成，shape={params.all_data.shape}，耗时 {end_time - start_time:.2f} 秒")
    params.date_list = list(params.all_data["date"].unique()) # 获取因子文件中的日期
    params.date_list = [x for x in params.date_list if x in params.ret_data.index and x in params.liquid_data.index] # 筛选出有标签和流动性的日期
    params.date_list.sort() # 对日期进行排序
    params.all_data = params.all_data.set_index("date").sort_index() # 将日期设置为索引，并进行排序
    params.factor_num = params.all_data.shape[1] - 3 # 2级sector embedding 要再减去2个！

    def get_train_date_split(season, period):
        """
        测试集长度为一个季度
        Example:
        >>> get_train_date_split('2023q2', 2)
        >>> ('202104', '202110', '202204', '202304', '202307')
        # 分别对应训练集开始时间，验证集开始时间，验证集结束时间，测试集开始时间，测试集结束时间
        """
        # 计算测试集的起始年月（如 season='2023q2'，则 test_start='202304'）
        test_start = season[:4] + str(int(season.split("q")[1]) * 3 - 2).zfill(2)
        # 将 test_start 转换为 datetime 对象
        start_date = datetime.strptime(test_start, "%Y%m")
        valid_date_split = []  # 用于存放一系列分割点的年月字符串
        
        # 生成一组以 test_start 为基准，分别往前推 -3、0、6、12、18、24 个月的年月字符串
        for i in [-3, 0, 6, 12, 18, 24]:
            valid_date_split.append((start_date - relativedelta(months=i)).strftime("%Y%m"))
        valid_date_split.reverse()  # 反转，使时间从早到晚排列

        train_start = valid_date_split[0]         # 训练集起始时间
        valid_start = valid_date_split[period - 1]# 验证集起始时间（根据 period 参数选择）
        valid_end = valid_date_split[period]      # 验证集结束时间
        test_end = valid_date_split[-1]           # 测试集结束时间（最晚的时间点）

        # 返回训练集起始、验证集起始、验证集结束、测试集起始、测试集结束时间
        return train_start, valid_start, valid_end, test_start, test_end
    print("【2】 Splitting Train, Valid, Test Date")
    train_start, valid_start, valid_end, test_start, test_end = get_train_date_split(season, period=int(
        params.test_save_path[-1]))
    print(f"period: {params.test_save_path[-1]}, train_start: {train_start}, valid_start: {valid_start}, valid_end: {valid_end}, test_start: {test_start}, test_end: {test_end}")
    if train_start < "202101":
        train_start = "202101"

    # 获取训练集，验证集，测试集日期
    # 隔开10天以防泄露未来信息
    # hv_Blocked_K_Fold
    valid_date_list = [x for x in params.date_list if valid_start <= x < valid_end][params.n_days:-10] # 删去时序训练用不到的n_days-1天
    train_date_list1 = [x for x in params.date_list if train_start <= x < valid_start][params.n_days:-10] # train第一部分:train_start到valid_start
    train_date_list2 = [x for x in params.date_list if valid_end <= x < test_start][10:-10] # train第二部分:valid_end到test_start的空档期
    train_date_list = train_date_list1 + train_date_list2
    test_date_list = [x for x in params.date_list if test_start <= x < test_end]
    
    # 极端行情不参与训练
    not_train_date = [x for x in params.date_list if (x >= "202402") & (x <= "20240223")]
    train_date_list = [x for x in train_date_list if x not in not_train_date]

    if len(test_date_list) == 0:
        test_date_list = ['out_sample']
    elif market == 'ALL':
        # 在 market 为 'ALL' 时，把 params.all_data 里"全为 0 的列"全部删除，只保留至少有一个非 0 值的列
        params.all_data = params.all_data.loc[:, params.all_data.replace(0, np.nan).dropna(how="all", axis=1).columns]
    else:
        raise NotImplementedError

    # 将特征名（feature names）和它们的编号（索引）写入一个文件，用于后续特征映射或特征管理
    feature_map = list(params.all_data.columns[1:-2]) # ind1 2 在最后2个col，没有sector embedding的话取[1:]即可
    params.factor_list = feature_map[:]
    with open(rf'{save_path}/{market}{name[len(market):len(market) + 6]}-feature_map.fea', 'w') as file:
        for idx, factor_name in enumerate(feature_map):
            file.write(rf'{factor_name}={idx}')
            file.write('\n')


    if state == 'train':
        print("【3】Training Started")
        train_single(args, name, args.seed, train_date_list, valid_date_list, test_date_list)
    else:
        raise NotImplementedError