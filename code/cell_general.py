import os
import glob
import torch
import numpy as np
import pandas as pd
import pdb
import time
import pickle
import logging
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset,random_split, ConcatDataset,Dataset
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import interp
from sklearn.metrics import roc_auc_score, roc_curve,precision_recall_curve, auc
from rdkit.Chem import rdMolDescriptors
import random
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit.Chem import Descriptors
from torch_geometric.loader import DataLoader
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch.utils.data.dataloader import default_collate
from rdkit import DataStructs
from torch_geometric.data import Batch
import learn2learn as l2l
import gc

cuda=True
device = torch.device('cuda:2'if torch.cuda.is_available() else "cpu")
data_path = ''

DATA_DIR = "/data"
REFERENCE_OUTPUT_BASE_DIR = "/cell_adapt"
REFERENCE_DATA_OUTPUT_DIR = os.path.join(REFERENCE_OUTPUT_BASE_DIR, "data_outputs")
REFERENCE_RESULT_DIR = os.path.join(REFERENCE_OUTPUT_BASE_DIR, "result")
TARGET_SPLIT_RUN = 1
TARGET_SPLIT_SEEDS = [1, 2, 3]
GENERAL_OUTPUT_DIR = "/cell_general"

os.makedirs(GENERAL_OUTPUT_DIR, exist_ok=True)

#载入数据
drug_features = pd.read_csv("/data/4130_drug_smiles_cid.csv")
source_cell_features = pd.read_csv(
    os.path.join(DATA_DIR, "log2_tpm_rich_zscore.csv"),
    index_col=0
).astype('float')
target_cell_features = pd.read_csv(
    os.path.join(DATA_DIR, "log2_tpm_poor_zscorebyrich.csv"),
    index_col=0
).astype('float')
cell_features = pd.concat(
    [source_cell_features, target_cell_features[~target_cell_features.index.isin(source_cell_features.index)]],
    axis=0
)


def load_source_domain_data():
    source_path = os.path.join(REFERENCE_DATA_OUTPUT_DIR, "source_domain_data.csv")
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"未找到 source_data: {source_path}")
    return pd.read_csv(source_path), source_path


def resolve_seed_split_dir(split_seed, split_run=TARGET_SPLIT_RUN):
    candidates = []
    if split_run is not None:
        candidates.append(os.path.join(REFERENCE_RESULT_DIR, f"run_{split_run}", f"seed_{split_seed}_data_splits"))
    candidates.extend(sorted(glob.glob(os.path.join(REFERENCE_RESULT_DIR, "run_*", f"seed_{split_seed}_data_splits"))))

    for split_dir in candidates:
        if os.path.isdir(split_dir):
            return split_dir
    return None


def load_target_split_data(split_seed):
    split_dir = resolve_seed_split_dir(split_seed)
    if split_dir is None:
        raise FileNotFoundError(f"未找到 seed={split_seed} 的数据划分目录。")

    finetune_files = sorted(glob.glob(os.path.join(split_dir, "task_*_train_split.csv")))
    test_files = sorted(glob.glob(os.path.join(split_dir, "task_*_test_split.csv")))
    if not finetune_files or not test_files:
        raise FileNotFoundError(f"seed={split_seed} 的 finetune/test 拆分文件不完整：{split_dir}")

    finetune_data = pd.concat([pd.read_csv(path) for path in finetune_files], ignore_index=True)
    test_data = pd.concat([pd.read_csv(path) for path in test_files], ignore_index=True)
    return finetune_data, test_data, split_dir


def get_cell_expression(cell_feature_df, cell):
    if cell in cell_feature_df.index:
        return np.array(cell_feature_df.loc[cell][:], dtype=np.float32)

    cell_str = str(cell)
    if cell_str in cell_feature_df.index:
        return np.array(cell_feature_df.loc[cell_str][:], dtype=np.float32)

    try:
        cell_idx = int(cell)
        return np.array(cell_feature_df.iloc[cell_idx][:], dtype=np.float32)
    except (TypeError, ValueError, IndexError):
        raise KeyError(f"细胞系 {cell} 在特征表中不存在。")


def get_cell_id_tensor(cell, fallback_value):
    try:
        return torch.tensor(int(cell), dtype=torch.long)
    except (TypeError, ValueError):
        return torch.tensor(fallback_value, dtype=torch.long)


# region 模型各种自定义
# 导入模型
# Two layers of fully connected layers
class FC2(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(FC2, self).__init__()

        self.fc1 = nn.Linear(in_features, int(in_features / 2))
        self.fc2 = nn.Linear(int(in_features / 2), out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


# Two layers of fully connected layers
class COMBFC2(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(COMBFC2, self).__init__()

        self.fc1 = nn.Linear(in_features, int(in_features / 2))
        self.fc2 = nn.Linear(int(in_features / 2), int(in_features / 2))
        self.fc3 = nn.Linear(int(in_features / 2), out_features)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x


# ========== Step 2: 定义原子和键类型 ========== #
ATOM_LIST = ['C', 'O', 'N', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H', 'K', 'Pt', 'As']
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]
DEVICE = torch.device('cpu')


# ========== Step 3: 原子/键特征提取 ========== #
def atom_features(atom):
    features = [atom.GetSymbol() == a for a in ATOM_LIST]
    features += [
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),
        int(atom.IsInRing()),
        int(atom.GetIsAromatic()),
        atom.GetMass() / 100.0,
    ]
    hyb = atom.GetHybridization()
    features += [
        int(hyb == Chem.rdchem.HybridizationType.SP),
        int(hyb == Chem.rdchem.HybridizationType.SP2),
        int(hyb == Chem.rdchem.HybridizationType.SP3),
        int(hyb == Chem.rdchem.HybridizationType.SP3D),
        int(hyb == Chem.rdchem.HybridizationType.SP3D2),
    ]
    return torch.tensor(features, dtype=torch.float)


def bond_features(bond):
    bt = bond.GetBondType()
    return [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ]


# ========== Step 4: SMILES 转图函数 ========== #
fpgen = GetMorganGenerator(radius=2, fpSize=64)


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    try:
        fp = fpgen.GetFingerprint(mol)
        arr = np.zeros((64,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        mol_desc = arr
    except Exception as e:
        print(f"Fingerprint error for SMILES {smiles}: {e}")
        return None  # 转换失败，跳过

    node_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    edge_index = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        b_feat = bond_features(bond)
        edge_attrs += [b_feat, b_feat]

    if node_feats:
        x = torch.stack(node_feats)
    else:
        x = torch.empty((0, 25), dtype=torch.float)

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.mol_desc = torch.tensor(mol_desc, dtype=torch.float).unsqueeze(0)
    return data


# # 药物特征编码器
class DrugEncoder(nn.Module):
    def __init__(self, in_channels=25, hidden_channels=64, out_channels=256, edge_attr_dim=6, mol_desc_dim=64):
        super(DrugEncoder, self).__init__()
        self.FC2 = FC2(out_channels + hidden_channels, out_channels, dropout=0.3)

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, out_channels)

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        self.mol_desc_mlp = nn.Sequential(
            nn.Linear(mol_desc_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, data):
 
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index)
        x = F.relu(x)
        x = self.conv3(x, data.edge_index)

        graph_feat = global_mean_pool(x, data.batch)
        mol_desc_feat = self.mol_desc_mlp(data.mol_desc.squeeze(1))  # 维度修复
        combined_feat = torch.cat([graph_feat, mol_desc_feat], dim=1)

        x = self.FC2(combined_feat)
        return x




class CellEncoder(nn.Module):
    def __init__(self, in_features=14890, out_features=64, dropout=0.3):
        super(CellEncoder, self).__init__()
        layers = []
        layers.append(nn.Linear(in_features, 4096))
        layers.append(nn.LayerNorm(4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, 2048))
        layers.append(nn.LayerNorm(2048))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(2048, 512))
        layers.append(nn.LayerNorm(512))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(512, 128))
        layers.append(nn.LayerNorm(128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, out_features))
        self.net = nn.Sequential(*layers)
    def forward(self, c_list):
        id, ge = c_list
        return self.net(ge)




class Comb(nn.Module):
    def __init__(self,
                 out_size=64,
                 dropout=0.3):
        super(Comb, self).__init__()

        self.dropout = dropout
        # drug
        self.DrugEncoder = DrugEncoder()
        # cell
        self.CellEncoder = CellEncoder()
        # fc
        self.fc_response = COMBFC2(in_features=576, out_features=1, dropout=dropout)  # 重新写预测部分的全连接

    def forward(self, d1_list, d2_list, c_list):
        d1 = self.DrugEncoder(d1_list)
        d2 = self.DrugEncoder(d2_list)
        c = self.CellEncoder(c_list)
        alll = torch.cat((d1, d2, c), 1)
        y = self.fc_response(alll)

        return y


# 数据集任务划分
##取样本及其对应特征
class DrugCombDataset(Dataset):
    def __init__(self, df, drug_features, cell_features):
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.d1_col = 'drug_row_cid' if 'drug_row_cid' in df.columns else df.columns[0]
        self.d2_col = 'drug_col_cid' if 'drug_col_cid' in df.columns else df.columns[1]
        self.cell_col = 'cell_line_name' if 'cell_line_name' in df.columns else df.columns[2]
        self.label_col = 'label' if 'label' in df.columns else df.columns[3]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        d1 = row[self.d1_col]
        d2 = row[self.d2_col]
        cell = row[self.cell_col]
        label = row[self.label_col]

        # 取出 SMILES
        d1_sm = self.drug_features.loc[d1, 'smiles']
        d2_sm = self.drug_features.loc[d2, 'smiles']

        # 将 SMILES 转为图结构
        d1_graph = mol_to_graph(d1_sm)

        if d1_graph is None:
            print(f"Invalid SMILES at index {idx}: {d1_sm}")

        d2_graph = mol_to_graph(d2_sm)

        if d1_graph is None or d2_graph is None:
            # 出现非法 SMILES（None），可返回空图或抛出异常；这里抛出异常方便调试
            return None

        # 为 batch 添加 batch index（单图视为 batch=0）
        d1_graph.batch = torch.zeros(d1_graph.x.size(0), dtype=torch.long)
        d2_graph.batch = torch.zeros(d2_graph.x.size(0), dtype=torch.long)

        # 细胞表达信息
        c_ge = torch.tensor(get_cell_expression(self.cell_features, cell), dtype=torch.float)

        sample = {
            'd1': d1_graph,
            'd2': d2_graph,
            'cell': get_cell_id_tensor(cell, idx),
            'c_ge': c_ge,
            'label': torch.tensor(label, dtype=torch.float)
        }

        return sample


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {'skip': True}

    d1_list = [b['d1'] for b in batch]
    d2_list = [b['d2'] for b in batch]

    batch_d1 = Batch.from_data_list(d1_list)
    batch_d2 = Batch.from_data_list(d2_list)

    cells = torch.tensor([b['cell'] for b in batch], dtype=torch.long)
    c_ge = torch.stack([b['c_ge'] for b in batch])
    labels = torch.tensor([b['label'] for b in batch], dtype=torch.float)

    return {
        'd1': batch_d1,
        'd2': batch_d2,
        'cell': cells,
        'c_ge': c_ge,
        'label': labels
    }




# endregion函数

# region 源域、目标域、微调、测试数据
support_data, support_data_path = load_source_domain_data()
print(f"source_data loaded from: {support_data_path}")
cl_support = DrugCombDataset(support_data, drug_features, source_cell_features)
support_loader = DataLoader(cl_support, batch_size=1280, num_workers=4, shuffle= True, pin_memory=True,collate_fn=collate_skip_none)


# endregion





def train(dataloader):
    model.train()
    running_loss = 0.0
    for iteration, sample in enumerate(dataloader):

        d1 = sample['d1'].to(device)
        d2 = sample['d2'].to(device)
        cell = sample['cell'].to(device)
        c_ge = sample['c_ge'].to(device).float()
        labels = sample['label'].to(device).float()

        optimizer.zero_grad()
        y_pred = model(d1, d2, (cell, c_ge))
        loss = criterion(y_pred, labels.view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('| epoch {:3d} | train_loss {:8.5f}'.format(i, running_loss / len(dataloader)))
    return running_loss / len(dataloader)



def new_evaluate(dataloader):
    model.eval()  ##固定模型参
    all_outputs = []
    all_labels = []
    with torch.no_grad():  # 不计算梯度
        for iteration, sample in enumerate(dataloader):
            d1 = sample['d1'].to(device)
            d2 = sample['d2'].to(device)
            cell = sample['cell'].to(device)
            c_ge = sample['c_ge'].to(device).float()
            labels = sample['label'].to(device).float()


            y_pred = model(d1, d2, (cell, c_ge))
            all_outputs.extend(y_pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_outputs, all_labels


# ============== 早停 + 三层重复（外部数据读取） ==============
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 外部准备好的三套随机划分数据路径（请确保存在）
f_path = REFERENCE_DATA_OUTPUT_DIR
t_path = REFERENCE_DATA_OUTPUT_DIR

# 若上述文件不存在，可退回到默认文件以确保代码可运行（但三次读取将相同）
for split_seed in TARGET_SPLIT_SEEDS:
    split_dir = resolve_seed_split_dir(split_seed)
    if split_dir is None:
        print(f"Warning: seed={split_seed} 的目标域划分目录不存在。")

train_seeds = [1, 2, 3]
ft_seeds = [1, 2, 3]

results_counter = 0

for split_seed in TARGET_SPLIT_SEEDS:
    # 从外部文件读取本次划分
    finetune_data, test_data, split_data_path = load_target_split_data(split_seed)
    print(f"target split seed={split_seed} loaded from: {split_data_path}")

    # 构建 DataLoader（本轮使用）
    cl_finetune = DrugCombDataset(finetune_data, drug_features, target_cell_features)
    finetune_loader = DataLoader(cl_finetune, batch_size=len(cl_finetune), num_workers=4, pin_memory=True, collate_fn=collate_skip_none)

    cl_test = DrugCombDataset(test_data, drug_features, target_cell_features)
    test_loader = DataLoader(cl_test, batch_size=len(cl_test), num_workers=4, pin_memory=True, collate_fn=collate_skip_none)

    r_train = 0
    # ========== 预训练（源域）含早停 ==========
    set_seed(train_seeds[r_train])
    model = Comb().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    patience = 20
    no_improve_count = 0
    pretrain_path = os.path.join(
        GENERAL_OUTPUT_DIR,
        f"general_model_run_{TARGET_SPLIT_RUN}_seed{split_seed}_train{r_train}_pre.pth"
    )

    for i in range(1, 301):
        l = train(support_loader)
        if l < best_loss:
            best_loss = l
            no_improve_count = 0
            torch.save(model.state_dict(), pretrain_path)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"[预训练早停] 连续 {patience} 轮 loss 未下降，停止预训练。")
                break


    for r_ft in range(3):
        set_seed(ft_seeds[r_ft])

        # 载入最佳预训练权重后进行微调（含早停）
        model = Comb().to(device)
        model.load_state_dict(torch.load(pretrain_path, map_location=device))
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        best_ft_loss = float('inf')
        no_improve_count = 0
        finetune_path = os.path.join(
            GENERAL_OUTPUT_DIR,
            f"general_model_run_{TARGET_SPLIT_RUN}_seed{split_seed}_train{r_train}_ft{r_ft}.pth"
        )

        for i in range(1, 201):  # 原代码微调50轮
            l = train(finetune_loader)
            if l < best_ft_loss:
                best_ft_loss = l
                no_improve_count = 0
                torch.save(model.state_dict(), finetune_path)
            else:
                no_improve_count += 1
                if no_improve_count >= patience:
                    print(f"[微调早停] 连续 {patience} 轮 loss 未下降，停止微调。")
                    break

        ### 下面针对每一次的微调结果进行模型评估
        model = Comb().to(device)
        model.load_state_dict(torch.load(finetune_path, map_location=device))
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.002)

        cl_pred, cl_label = new_evaluate(test_loader)

        results_df = pd.DataFrame({
            'prediction': np.array(cl_pred).flatten(),
            'label': np.array(cl_label).flatten()
        })
        out_csv = os.path.join(
            GENERAL_OUTPUT_DIR,
            f"general_finetune_results_run_{TARGET_SPLIT_RUN}_seed{split_seed}_train{r_train}_eval{r_ft}.csv"
        )
        results_df.to_csv(out_csv, index=False)
        results_counter += 1
        print(f"测试结果已保存到 {out_csv} ({results_counter}/27)")
        print("\n评估指标:")
        print("AUC-ROC:", roc_auc_score(results_df['label'], results_df['prediction']))
        print("AUPR:", metrics.average_precision_score(results_df['label'], results_df['prediction']))



