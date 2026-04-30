#    meta_lr = 0.0008, fast_lr = 0.08
import os
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
import json

# ========== 路径配置 ========== #
DATA_DIR = "/data"
OUTPUT_BASE_DIR = "/cell_adapt_result/"
CHECKPOINT_DIR = os.path.join(OUTPUT_BASE_DIR, "checkpoints")
RESULT_DIR = os.path.join(OUTPUT_BASE_DIR, "result")
DATA_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "data_outputs")

# ========== 检查点配置 ========== #
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

def save_checkpoint(state, filename):
    """保存检查点"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    torch.save(state, filepath)
    print(f"检查点已保存: {filepath}")

def load_checkpoint(filename, model, optimizer=None):
    """加载检查点"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"检查点已加载: {filepath}")
        return checkpoint.get('epoch', 0), checkpoint.get('best_loss', float('inf'))
    else:
        print(f"检查点不存在: {filepath}")
        return 0, float('inf')

def save_training_state(run, epoch, model, optimizer, best_loss, filename):
    """保存训练状态"""
    state = {
        'run': run,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    save_checkpoint(state, filename)

def save_testing_state(run, seed, iteration, results, filename):
    """保存测试状态"""
    state = {
        'run': run,
        'random_seed': seed,
        'test_iteration': iteration,
        'results': results
    }
    save_checkpoint(state, filename)

def save_progress(run, current_seed, current_iteration, status):
    """保存整体进度"""
    progress = {
        'current_run': run,
        'current_seed': current_seed,
        'current_iteration': current_iteration,
        'status': status,
        'timestamp': time.time()
    }
    with open(os.path.join(CHECKPOINT_DIR, 'training_progress.json'), 'w') as f:
        json.dump(progress, f, indent=2)

def load_progress():
    """加载整体进度"""
    progress_file = os.path.join(CHECKPOINT_DIR, 'training_progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return None

#载入数据
#源域：细胞系的源域数据(datarich)
df = pd.read_pickle("/data/cl_label_data_onlysmiles_1016.pickle")
df = df.drop(columns=['css_ri', 'S_sum'])  # 删除不需要的列

drug_features =  pd.read_csv("/data/4130_drug_smiles_cid.csv")

source_cell_features = pd.read_csv(
    os.path.join(DATA_DIR, "log2_tpm_rich_zscore.csv"),
    index_col=0
)
target_cell_features = pd.read_csv(
    os.path.join(DATA_DIR, "log2_tpm_poor_zscorebyrich.csv"),
    index_col=0
)
cuda=True
device = torch.device('cuda:0'if torch.cuda.is_available() else "cpu")

#导入模型
#Two layers of fully connected layers
class FC2(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(FC2, self).__init__()
        
        self.fc1 = nn.Linear(in_features, int(in_features/2))
        self.fc2 = nn.Linear(int(in_features/2),out_features)
        self.dropout= nn.Dropout(dropout)
                
    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
#Two layers of fully connected layers
class COMBFC2(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(COMBFC2, self).__init__()
        
        self.fc1 = nn.Linear(in_features, int(in_features/2))
        self.fc2 = nn.Linear(int(in_features/2), int(in_features/2))
        self.fc3= nn.Linear(int(in_features/2),out_features)
        self.dropout= nn.Dropout(dropout)
        self.sigmoid= nn.Sigmoid()
                
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
SMALL_TASK_THRESHOLD = 256
DRUG_GRAPH_CACHE = {}
CELL_EXPRESSION_CACHE = {}

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


def _normalize_cache_key(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_cached_drug_graph(smiles):
    cache_key = _normalize_cache_key(smiles)
    cached_graph = DRUG_GRAPH_CACHE.get(cache_key)
    if cached_graph is None:
        cached_graph = mol_to_graph(smiles)
        if cached_graph is None:
            return None
        DRUG_GRAPH_CACHE[cache_key] = cached_graph
    return cached_graph.clone()


def get_cached_cell_expression(cell_features, cell):
    cell_key = _normalize_cache_key(cell)
    cache_key = (id(cell_features), cell_key)
    cached_tensor = CELL_EXPRESSION_CACHE.get(cache_key)
    if cached_tensor is None:
        cell_row = np.asarray(cell_features.loc[cell_key], dtype=np.float32)
        cached_tensor = torch.from_numpy(cell_row)
        CELL_EXPRESSION_CACHE[cache_key] = cached_tensor
    return cached_tensor.clone()


def load_model_state(filepath, map_location=None):
    load_kwargs = {}
    if map_location is not None:
        load_kwargs["map_location"] = map_location
    try:
        return torch.load(filepath, weights_only=True, **load_kwargs)
    except TypeError:
        return torch.load(filepath, **load_kwargs)

#药物特征编码器
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

        # 第1层：in_features -> 5000
        layers.append(nn.Linear(in_features, 4096))
        layers.append(nn.LayerNorm(4096))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # 第2层：5000 -> 2000
        layers.append(nn.Linear(4096, 2048))
        layers.append(nn.LayerNorm(2048))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # 第3层：2000 -> 500
        layers.append(nn.Linear(2048, 512))
        layers.append(nn.LayerNorm(512))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # 第4层：500 -> 128
        layers.append(nn.Linear(512, 128))
        layers.append(nn.LayerNorm(128))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # 第5层：128 -> out_features
        layers.append(nn.Linear(128, out_features))
        self.net = nn.Sequential(*layers)
    def forward(self, c_list):
        id, ge = c_list
        return self.net(ge)



class Comb(nn.Module):
    def __init__(self,
              out_size = 64,
              dropout = 0.3):
        super(Comb, self).__init__()
        
        self.dropout = dropout
        #drug 
        self.DrugEncoder = DrugEncoder()
        #cell
        self.CellEncoder = CellEncoder()
        #fc
        self.fc_response = COMBFC2(in_features=576, out_features=1, dropout=dropout) #重新写预测部分的全连接
        
    def forward(self,d1_list,d2_list,c_list):
        d1 = self.DrugEncoder(d1_list)
        d2 = self.DrugEncoder(d2_list)
        c = self.CellEncoder(c_list)
        alll = torch.cat((d1, d2, c),1)
        y = self.fc_response(alll)
        
        return y
    


#数据集任务划分
##取样本及其对应特征
class DrugCombDataset(Dataset):
    def __init__(self, df, drug_features, cell_features):
        self.df = df.reset_index(drop=True)
        self.drug_features = drug_features
        self.cell_features = cell_features
        self.d1_values = self.df.iloc[:, 0].to_numpy()
        self.d2_values = self.df.iloc[:, 1].to_numpy()
        self.cells = self.df.iloc[:, 2].to_numpy()
        self.labels = self.df.iloc[:, 3].to_numpy()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        d1 = self.d1_values[idx]
        d2 = self.d2_values[idx]
        cell = self.cells[idx]
        label = self.labels[idx]
    
        # 取出 SMILES
        d1_sm = self.drug_features.loc[d1, 'smiles']
        d2_sm = self.drug_features.loc[d2, 'smiles']
    
        # 将 SMILES 转为图结构
        d1_graph = get_cached_drug_graph(d1_sm)
        
        if d1_graph is None:
            print(f"Invalid SMILES at index {idx}: {d1_sm}")
            
        d2_graph = get_cached_drug_graph(d2_sm)

        if d1_graph is None or d2_graph is None:
            # 出现非法 SMILES（None），可返回空图或抛出异常；这里抛出异常方便调试
            return None

        # 细胞表达信息
        c_ge = get_cached_cell_expression(self.cell_features, cell)

        sample = {
            'd1': d1_graph,
            'd2': d2_graph,
            'cell': torch.tensor(_normalize_cache_key(cell)),
            'c_ge': c_ge,
            'label': torch.tensor(label, dtype=torch.float)
        }

        return sample
        
        
        
# 统计每个细胞系出现的次数，并重置索引
singledrug = df['cell_line_name'].value_counts().reset_index()
# 重命名列名为 cell_line_name 和 frequency
singledrug.columns = ['cell_line_name','frequency']
# 选出样本数大于50的细胞系，作为源域
singledrug_s = singledrug[singledrug['frequency'] > 50]
# 选出样本数在10到50之间的细胞系，作为目标域
singledrug_t = singledrug[(singledrug['frequency'] <= 50)&(singledrug['frequency'] >= 10) ]
# 从原始数据中筛选出属于源域细胞系的数据
source_data = df[df['cell_line_name'].isin(singledrug_s['cell_line_name'])]
# 从原始数据中筛选出属于目标域细胞系的数据
target_data = df[df['cell_line_name'].isin(singledrug_t['cell_line_name'])]

# 初始化训练任务列表
train_tasks = []
# 获取源域所有唯一的细胞系名称
cell_lines = source_data['cell_line_name'].unique()
# 遍历每个细胞系
for cell_line in cell_lines:
    # 取出该细胞系对应的所有数据，作为一个任务
    task = source_data[source_data['cell_line_name'] == cell_line]
    # 加入训练任务列表
    train_tasks.append(task)

# 初始化测试任务列表
test_tasks = []
# 获取目标域所有唯一的细胞系名称
cell_lines = target_data['cell_line_name'].unique()
# 遍历每个细胞系
for cell_line in cell_lines:
    # 取出该细胞系对应的所有数据，作为一个任务
    task = target_data[target_data['cell_line_name'] == cell_line]
    # 加入测试任务列表
    test_tasks.append(task)
     

# 保存源域数据
# 创建保存目录
output_dir = DATA_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)

# 保存源域数据
source_data.to_csv(f"{output_dir}/source_domain_data.csv", index=False)
print(f"源域数据已保存，共 {len(source_data)} 条样本")

# 保存源域任务信息
source_tasks_info = []
for i, task in enumerate(train_tasks):
    source_tasks_info.append({
        'task_id': i,
        'cell_line': task['cell_line_name'].iloc[0],
        'num_samples': len(task),
        'num_positive': len(task[task['label'] == 1]),
        'num_negative': len(task[task['label'] == 0])
    })

source_tasks_df = pd.DataFrame(source_tasks_info)
source_tasks_df.to_csv(f"{output_dir}/source_tasks_info.csv", index=False)
print(f"源域任务信息已保存，共 {len(source_tasks_info)} 个任务")


# 保存目标域数据
target_data.to_csv(f"{output_dir}/target_domain_data.csv", index=False)
print(f"目标域数据已保存，共 {len(target_data)} 条样本")

# 保存目标域任务信息
target_tasks_info = []
for i, task in enumerate(test_tasks):
    target_tasks_info.append({
        'task_id': i,
        'cell_line': task['cell_line_name'].iloc[0],
        'num_samples': len(task),
        'num_positive': len(task[task['label'] == 1]),
        'num_negative': len(task[task['label'] == 0])
    })

target_tasks_df = pd.DataFrame(target_tasks_info)
target_tasks_df.to_csv(f"{output_dir}/target_tasks_info.csv", index=False)
print(f"目标域任务信息已保存，共 {len(target_tasks_info)} 个任务")


    
# 定义一个函数，用于从数据集中按标签采样
def label_sampling(df,k):
    # 取出所有正样本（label为1）
    df_positive = df[df['label'] == 1]
    # 取出所有负样本（label为0）
    df_negative = df[df['label'] == 0]

    # 统计正负样本的数量
    num_positive = df_positive.shape[0]
    num_negative = df_negative.shape[0]

    # 计算要采样的正负样本数，最多不超过k
    num_positive_sample = min(num_positive, k)
    num_negative_sample = min(num_negative, k)

    # 如果正样本不足k，则负样本补足2k
    if num_positive_sample < k:
        num_negative_sample = min(num_negative, 2*k - num_positive_sample)
    # 如果负样本不足k，则正样本补足2k
    elif num_negative_sample < k:
        num_positive_sample = min(num_positive, 2*k - num_negative_sample)

    # 随机采样正样本
    sample_positive = df_positive.sample(n=num_positive_sample)
    # 随机采样负样本
    sample_negative = df_negative.sample(n=num_negative_sample)

    # 合并采样的正负样本
    sample_df = pd.concat([sample_positive, sample_negative])
    
    # 返回采样后的数据
    return sample_df
    
    
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


def build_task_batch(dataset):
    if len(dataset) == 0:
        return {'skip': True}
    return collate_skip_none([dataset[idx] for idx in range(len(dataset))])


def get_task_batch_iter(dataset, batch_size, num_workers=8, pin_memory=True):
    if len(dataset) == 0:
        return [{'skip': True}]
    if len(dataset) <= SMALL_TASK_THRESHOLD:
        return [build_task_batch(dataset)]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_skip_none,
    )


def maml_adapt_on_support(learner, support_batches, criterion, device, adapt_steps=3):
    valid_batches = 0

    for _ in range(adapt_steps):
        for sample in support_batches:
            if isinstance(sample, dict) and sample.get('skip', False):
                continue

            valid_batches += 1

            support_d1 = sample['d1'].to(device)
            support_d2 = sample['d2'].to(device)
            support_cell = sample['cell'].to(device)
            support_c_ge = sample['c_ge'].float().to(device)
            support_label = sample['label'].float().to(device)

            support_pred = learner(support_d1, support_d2, (support_cell, support_c_ge))
            if torch.isnan(support_pred).any() or torch.isinf(support_pred).any():
                print("检测到NaN或Inf值在support_pred中")
                continue
            if (support_pred < 0).any() or (support_pred > 1).any():
                print(f"support_pred值超出范围: min={support_pred.min().item()}, max={support_pred.max().item()}")
                support_pred = torch.clamp(support_pred, 1e-7, 1-1e-7)

            support_loss = criterion(support_pred, support_label.view(-1, 1))
            learner.adapt(support_loss)

    return valid_batches      

# ========== 主要修改：为每次运行定义不同的随机种子组合 ========== #
num_runs = 3  # 运行3次
# 定义每次运行的随机种子组合：第一次[1,2,3]，第二次[4,5,6]，第三次[7,8,9]
run_seeds = {
    1: [1, 2, 3],
    2: [4, 5, 6], 
    3: [7, 8, 9]
}

all_run_results = []  # 保存每次运行的结果

# ========== 加载进度检查点 ========== #
progress = load_progress()
if progress:
    print(f"发现之前的进度: 运行 {progress['current_run']}, 种子 {progress['current_seed']}, 迭代 {progress['current_iteration']}")
    resume_run = progress['current_run']
    resume_seed = progress['current_seed'] 
    resume_iteration = progress['current_iteration']
else:
    resume_run = 1
    resume_seed = None
    resume_iteration = None

for run in range(resume_run, num_runs + 1):
    print(f"\n{'='*60}")
    print(f"开始第 {run}/{num_runs} 次完整运行")
    print(f"{'='*60}")
    
    # 为每次运行创建独立的输出目录
    run_output_dir = os.path.join(RESULT_DIR, f"run_{run}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    # 获取当前运行的随机种子组合
    current_run_seeds = run_seeds[run]
    print(f"第 {run} 次运行使用的随机种子: {current_run_seeds}")
    
    # ========== 检查点：训练阶段 ========== #
    train_checkpoint_file = f"run_{run}_training_checkpoint.pth"
    
    #源域 MAML 训练 - 添加早停机制
    net_meta = Comb().to(device)
    meta_lr = 0.0008
    fast_lr = 0.08
    # ========== 修改：使用带数值稳定性的BCEWithLogitsLoss ========== #
    # 注意：由于我们在模型内部已经使用了sigmoid，所以这里仍然使用BCELoss
    criterion = nn.BCELoss()
    maml = l2l.algorithms.MAML(net_meta, lr=fast_lr, allow_unused=True)  # 内循环大，外循环小
    meta_optimizer = optim.Adam(maml.parameters(), lr=meta_lr)

    task_batch_size = 4  # 每批处理多少个任务
    num_batches = len(train_tasks) // task_batch_size + int(len(train_tasks) % task_batch_size > 0)

    # ========== 添加早停机制 ========== #
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    early_stop = False
    
    # ========== 尝试加载训练检查点 ========== #
    start_epoch, checkpoint_best_loss = load_checkpoint(train_checkpoint_file, net_meta, meta_optimizer)
    if start_epoch > 0:
        print(f"从第 {start_epoch} 轮恢复训练")
        best_loss = checkpoint_best_loss
    
    for counter in range(start_epoch, 200):
        if early_stop:
            print(f"早停触发！在第 {counter} 轮停止训练")
            break
            
        t0 = time.time()
        total_loss = 0.0
        
        for batch_id in range(num_batches):
            meta_optimizer.zero_grad()
            batch_loss = 0.0

            # 获取该批次的任务索引
            task_indices = list(range(batch_id * task_batch_size, min((batch_id + 1) * task_batch_size, len(train_tasks))))
            
            for i in task_indices:
                try:
                    cell_task = train_tasks[i].copy()
                    cell_task = label_sampling(cell_task, 25)
                    pos = cell_task[cell_task['label'] == 1]
                    neg = cell_task[cell_task['label'] == 0]
                    
                    # 确保正负样本都存在
                    if len(pos) == 0 or len(neg) == 0:
                        continue
                        
                    pos_train, pos_test = train_test_split(pos, test_size=0.5)
                    neg_train, neg_test = train_test_split(neg, test_size=0.5)
                    train = pd.concat([pos_train, neg_train], axis=0)
                    test = pd.concat([pos_test, neg_test], axis=0)

                    net_copy = maml.clone()
                    support_set = DrugCombDataset(train, drug_features, source_cell_features)
                    query_set = DrugCombDataset(test, drug_features, source_cell_features)
                    support_batches = get_task_batch_iter(support_set, batch_size=1280, num_workers=8, pin_memory=True)
                    query_batches = get_task_batch_iter(query_set, batch_size=1280, num_workers=8, pin_memory=True)

                    # Support set 训练
                    valid_batches = 0
                    for _ in range(3):
                        for sample in support_batches:
                            if isinstance(sample, dict) and sample.get('skip', False):
                                continue
                            valid_batches += 1
                            
                            support_d1 = sample['d1'].to(device)
                            support_d2 = sample['d2'].to(device)
                            support_cell = sample['cell'].to(device)
                            support_c_ge = sample['c_ge'].float().to(device)
                            support_label = sample['label'].float().to(device)

                            support_pred = net_copy(support_d1, support_d2, (support_cell, support_c_ge))
                            # ========== 添加数值检查 ========== #
                            if torch.isnan(support_pred).any() or torch.isinf(support_pred).any():
                                print("检测到NaN或Inf值在support_pred中")
                                continue
                            if (support_pred < 0).any() or (support_pred > 1).any():
                                print(f"support_pred值超出范围: min={support_pred.min().item()}, max={support_pred.max().item()}")
                                # 进行数值裁剪
                                support_pred = torch.clamp(support_pred, 1e-7, 1-1e-7)
                            
                            support_loss = criterion(support_pred, support_label.view(-1, 1))
                            net_copy.adapt(support_loss)
                        
                        if valid_batches == 0:
                            continue

                    # Query set 验证
                    valid_batches = 0
                    for sample in query_batches:
                        if isinstance(sample, dict) and sample.get('skip', False):
                            continue
                        valid_batches += 1
                            
                        query_d1 = sample['d1'].to(device)
                        query_d2 = sample['d2'].to(device)
                        query_cell = sample['cell'].to(device)
                        query_c_ge = sample['c_ge'].float().to(device)
                        query_label = sample['label'].float().to(device)

                        query_pred = net_copy(query_d1, query_d2, (query_cell, query_c_ge))
                        # ========== 添加数值检查 ========== #
                        if torch.isnan(query_pred).any() or torch.isinf(query_pred).any():
                            print("检测到NaN或Inf值在query_pred中")
                            continue
                        if (query_pred < 0).any() or (query_pred > 1).any():
                            print(f"query_pred值超出范围: min={query_pred.min().item()}, max={query_pred.max().item()}")
                            # 进行数值裁剪
                            query_pred = torch.clamp(query_pred, 1e-7, 1-1e-7)
                        
                        query_loss = criterion(query_pred, query_label.view(-1, 1))
                        batch_loss += query_loss

                    del net_copy
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing task {i}: {str(e)}")
                    continue

            if len(task_indices) > 0:
                meta_loss = batch_loss / len(task_indices) 
                #print(f"Epoch {counter+1} - Batch {batch_id+1}/{num_batches} - Meta Loss: {meta_loss.item():.4f}")
                meta_loss.backward()
                meta_optimizer.step()
                total_loss += meta_loss.item()

        avg_loss = total_loss / num_batches if num_batches > 0 else total_loss
        print(f"Run {run} - Epoch {counter+1} finished - Average Meta Loss: {avg_loss:.4f} - Time: {time.time()-t0:.2f}s")
        
        # ========== 早停检查 ========== #
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_path = f"{run_output_dir}/gcn_maml_run_{run}_best.pth"
            torch.save(net_meta.state_dict(), best_model_path)
            print(f"新的最佳模型已保存，损失: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"损失未改善，耐心计数: {patience_counter}/{patience}")
            
        # ========== 保存训练检查点 ========== #
        save_training_state(run, counter + 1, net_meta, meta_optimizer, best_loss, train_checkpoint_file)
        save_progress(run, None, None, 'training')
            
        if patience_counter >= patience:
            early_stop = True
            print(f"早停触发！最佳损失: {best_loss:.4f}")

    # 保存最终模型（每次运行保存独立的模型）
    final_model_path = f"{run_output_dir}/gcn_maml_run_{run}_final.pth"
    torch.save(net_meta.state_dict(), final_model_path)
    print(f"第 {run} 次运行的最终模型已保存: {final_model_path}")

    # 目标域测试
    test_tasks = []
    cell_lines = target_data['cell_line_name'].unique()
    for cell_line in cell_lines:
        task = target_data[target_data['cell_line_name'] == cell_line]
        test_tasks.append(task)

    best_model_path = f"{run_output_dir}/gcn_maml_run_{run}_best.pth"
    best_model_state = load_model_state(best_model_path, map_location=device)
    test_base_model = Comb().to(device)
    test_base_model.load_state_dict(best_model_state)
    maml_test = l2l.algorithms.MAML(test_base_model, lr=fast_lr, allow_unused=True)
    test_criterion = nn.BCELoss()
        
    # ========== 主要修改：使用当前运行的随机种子组合 ========== #
    seed_results = []  # 保存每个随机种子的结果
    
    # 确定从哪个种子开始恢复
    start_seed_idx = 0
    if run == resume_run and resume_seed is not None:
        for idx, seed in enumerate(current_run_seeds):
            if seed == resume_seed:
                start_seed_idx = idx
                break
    
    for seed_idx in range(start_seed_idx, len(current_run_seeds)):
        random_seed = current_run_seeds[seed_idx]
        print(f"\n--- 第 {run} 次运行 - 随机种子 {seed_idx+1}/{len(current_run_seeds)} (seed={random_seed}) ---")
        
        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # 为当前随机种子创建数据划分目录
        seed_data_dir = f"{run_output_dir}/seed_{random_seed}_data_splits"
        os.makedirs(seed_data_dir, exist_ok=True)
        
        all_test_task = []
        task_split_info = []  # 保存每个任务的数据划分信息

        for i in range(len(test_tasks)):
            cell_task = test_tasks[i]
            # 确保正负样本都存在
            if len(cell_task[cell_task['label'] == 1]) == 0 or len(cell_task[cell_task['label'] == 0]) == 0:
                continue
            
            # 使用当前随机种子划分数据
            train, test = train_test_split(cell_task, test_size=0.5, random_state=random_seed)
            temp_dict = {
                'train': train,
                'test': test,
            }
            all_test_task.append(temp_dict)
            
            # 保存数据划分信息
            task_split_info.append({
                'task_id': i,
                'cell_line': cell_task['cell_line_name'].iloc[0],
                'random_seed': random_seed,
                'original_samples': len(cell_task),
                'original_positive': len(cell_task[cell_task['label'] == 1]),
                'original_negative': len(cell_task[cell_task['label'] == 0]),
                'train_samples': len(train),
                'train_positive': len(train[train['label'] == 1]),
                'train_negative': len(train[train['label'] == 0]),
                'test_samples': len(test),
                'test_positive': len(test[test['label'] == 1]),
                'test_negative': len(test[test['label'] == 0])
            })
            
            # 保存每个任务的具体数据划分
            train.to_csv(f"{seed_data_dir}/task_{i}_train_split.csv", index=False)
            test.to_csv(f"{seed_data_dir}/task_{i}_test_split.csv", index=False)
            cell_task.to_csv(f"{seed_data_dir}/task_{i}_original_data.csv", index=False)

        # 保存数据划分汇总信息
        task_split_df = pd.DataFrame(task_split_info)
        task_split_df.to_csv(f"{seed_data_dir}/data_split_summary.csv", index=False)
        print(f"随机种子 {random_seed} 的数据划分已保存到: {seed_data_dir}")
        print(f"共 {len(task_split_info)} 个任务的数据划分")

        # 保存所有测试任务的总信息
        test_tasks_summary = []
        for i, task_data in enumerate(all_test_task):
            test_tasks_summary.append({
                'task_id': i,
                'cell_line': task_data['train']['cell_line_name'].iloc[0],
                'random_seed': random_seed,
                'adapt_samples': len(task_data['train']),
                'adapt_positive': len(task_data['train'][task_data['train']['label'] == 1]),
                'adapt_negative': len(task_data['train'][task_data['train']['label'] == 0]),
                'test_samples': len(task_data['test']),
                'test_positive': len(task_data['test'][task_data['test']['label'] == 1]),
                'test_negative': len(task_data['test'][task_data['test']['label'] == 0])
            })

        test_tasks_summary_df = pd.DataFrame(test_tasks_summary)
        test_tasks_summary_df.to_csv(f"{run_output_dir}/target_test_tasks_seed_{random_seed}_summary.csv", index=False)
        print(f"第 {run} 次运行 - 随机种子 {random_seed} - 目标域测试任务汇总已保存，共 {len(test_tasks_summary)} 个任务")

        # ========== 主要修改：细分保存每次测试迭代的结果 ========== #
        all_roc = []
        all_pr = []
        iteration_results = []  # 保存每次迭代的详细结果

        # 确定从哪个迭代开始恢复
        start_iteration = 0
        if run == resume_run and seed_idx == start_seed_idx and resume_iteration is not None:
            start_iteration = resume_iteration - 1  # 转换为0-based索引
        
        for test_iter in range(start_iteration, 3):  # 运行3次取平均
            print(f"\n--- 第 {run} 次运行 - 随机种子 {random_seed} - 测试迭代 {test_iter+1}/3 ---")
            
            # 保存当前进度
            save_progress(run, random_seed, test_iter + 1, 'testing')
            
            # 为每次测试迭代创建独立的目录
            iteration_dir = f"{run_output_dir}/seed_{random_seed}/iteration_{test_iter+1}"
            os.makedirs(iteration_dir, exist_ok=True)
            
            preds = []
            labels = []
            task_details = []  # 保存每个任务的详细预测结果
            
            for j, d in enumerate(all_test_task):
                try:
                    learner = maml_test.clone()
                    
                    support_set = d['train']
                    query_set = d['test']
                    supportdata = DrugCombDataset(support_set, drug_features, target_cell_features)
                    querydata = DrugCombDataset(query_set, drug_features, target_cell_features)
                    support_batches = get_task_batch_iter(supportdata, batch_size=len(support_set), num_workers=8, pin_memory=True)
                    query_batches = get_task_batch_iter(querydata, batch_size=len(query_set), num_workers=8, pin_memory=True)
                    
                    # 使用 MAML adapt 在 support set 上做任务适配
                    learner.train()
                    valid_batches = maml_adapt_on_support(
                        learner,
                        support_batches,
                        test_criterion,
                        device,
                        adapt_steps=3,
                    )
                    
                    if valid_batches == 0:
                        del learner
                        continue
                        
                    # 测试模型
                    learner.eval()
                    with torch.no_grad():
                        for sample in query_batches:
                            if isinstance(sample, dict) and sample.get('skip', False):
                                continue
                                
                            query_d1 = sample['d1'].to(device)
                            query_d2 = sample['d2'].to(device)
                            query_cell = sample['cell'].to(device)
                            query_c_ge = sample['c_ge'].float().to(device)
                            query_label = sample['label'].float().to(device)
                            
                            query_pred = learner(query_d1, query_d2, (query_cell, query_c_ge))
                            
                            # 保存每个样本的预测结果
                            batch_preds = query_pred.cpu().detach().numpy()
                            batch_labels = query_label.cpu().detach().numpy()
                            
                            # 保存任务级别的详细信息
                            for idx, (pred, label_val) in enumerate(zip(batch_preds, batch_labels)):
                                task_details.append({
                                    'task_id': j,
                                    'cell_line': d['train']['cell_line_name'].iloc[0],
                                    'predicted_prob': pred[0] if isinstance(pred, np.ndarray) else pred,
                                    'true_label': label_val,
                                    'random_seed': random_seed,
                                    'test_iteration': test_iter + 1,
                                    'run': run
                                })
                            
                            preds.extend(batch_preds)
                            labels.extend(batch_labels)
                            
                    del learner
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing test task {j}: {str(e)}")
                    continue
            
            # 保存每次迭代的详细结果
            if len(task_details) > 0:
                iteration_df = pd.DataFrame(task_details)
                iteration_df.to_csv(f"{iteration_dir}/detailed_predictions.csv", index=False)
                print(f"第 {test_iter+1} 次迭代的详细预测结果已保存: {iteration_dir}/detailed_predictions.csv")
                print(f"共保存 {len(iteration_df)} 条记录")
            
            if len(preds) > 0 and len(labels) > 0:
                fpr, tpr, _ = roc_curve(labels, preds)
                roc_auc = auc(fpr, tpr)
                precision, recall, _ = precision_recall_curve(labels, preds)
                aupr = auc(recall, precision)
                
                all_roc.append(roc_auc)
                all_pr.append(aupr)
                
                # 保存本次迭代的汇总结果
                iteration_result = {
                    'run': run,
                    'random_seed': random_seed,
                    'test_iteration': test_iter + 1,
                    'auroc': roc_auc,
                    'aupr': aupr,
                    'num_samples': len(preds),
                    'num_positive': sum(labels),
                    'num_negative': len(labels) - sum(labels)
                }
                iteration_results.append(iteration_result)
                
                # 保存本次迭代的labels和preds
                iteration_labels_preds = pd.DataFrame({
                    'true_label': labels,
                    'predicted_prob': [p[0] if isinstance(p, np.ndarray) else p for p in preds],
                    'random_seed': random_seed,
                    'test_iteration': test_iter + 1,
                    'run': run
                })
                iteration_labels_preds.to_csv(f"{iteration_dir}/labels_preds.csv", index=False)
                
                print(f"Run {run} - Seed {random_seed} - Iteration {test_iter+1}: AUROC = {roc_auc:.4f}, AUPR = {aupr:.4f}, Samples = {len(preds)}")
            else:
                print(f"Run {run} - Seed {random_seed} - Iteration {test_iter+1} skipped due to no valid predictions")

        # 保存当前随机种子的所有迭代结果汇总
        if iteration_results:
            iteration_summary_df = pd.DataFrame(iteration_results)
            iteration_summary_df.to_csv(f"{run_output_dir}/seed_{random_seed}_iterations_summary.csv", index=False)
            print(f"随机种子 {random_seed} 的所有迭代结果汇总已保存")

        # 保存当前随机种子的结果
        if len(all_roc) > 0:
            mean_roc = np.mean(all_roc)
            std_roc = np.std(all_roc)
            mean_pr = np.mean(all_pr)
            std_pr = np.std(all_pr)
            
            seed_result = {
                'run': run,
                'random_seed': random_seed,
                'mean_auroc': mean_roc,
                'std_auroc': std_roc,
                'mean_aupr': mean_pr,
                'std_aupr': std_pr,
                'num_iterations': len(all_roc)
            }
            seed_results.append(seed_result)
            
            print(f"第 {run} 次运行 - 随机种子 {random_seed} 结果:")
            print(f'MEAN AUROC: {mean_roc:.4f} ± {std_roc:.4f}')   
            print(f'MEAN AUPR: {mean_pr:.4f} ± {std_pr:.4f}')
        else:
            print(f"第 {run} 次运行 - 随机种子 {random_seed}: 没有获得有效结果")
            seed_results.append({
                'run': run,
                'random_seed': random_seed,
                'mean_auroc': None,
                'std_auroc': None,
                'mean_aupr': None,
                'std_aupr': None,
                'num_iterations': 0
            })
        
        # 保存当前运行的随机种子结果汇总
        seed_results_df = pd.DataFrame(seed_results)
        seed_results_df.to_csv(f"{run_output_dir}/seed_results_summary.csv", index=False)
        
        # 完成一个种子的测试后，清除进度（避免重复恢复）
        save_progress(run, None, None, 'completed_seed')
    
    # 保存本次运行的所有随机种子结果
    all_run_results.extend(seed_results)
    
    # 完成一次运行后，清除训练检查点
    if os.path.exists(os.path.join(CHECKPOINT_DIR, train_checkpoint_file)):
        os.remove(os.path.join(CHECKPOINT_DIR, train_checkpoint_file))
        print(f"训练检查点已清除: {train_checkpoint_file}")

# ========== 输出最终统计结果 ========== #
print(f"\n{'='*60}")
print(f"3次运行最终统计结果")
print(f"{'='*60}")

# 保存所有运行结果的汇总
results_summary_df = pd.DataFrame(all_run_results)
results_summary_df.to_csv(os.path.join(RESULT_DIR, "all_runs_summary.csv"), index=False)

# 计算总体平均值（排除无效结果）
valid_auroc = [r['mean_auroc'] for r in all_run_results if r['mean_auroc'] is not None]
valid_aupr = [r['mean_aupr'] for r in all_run_results if r['mean_aupr'] is not None]

if valid_auroc:
    overall_mean_auroc = np.mean(valid_auroc)
    overall_std_auroc = np.std(valid_auroc)
    overall_mean_aupr = np.mean(valid_aupr)
    overall_std_aupr = np.std(valid_aupr)
    
    print(f"总体平均 AUROC: {overall_mean_auroc:.4f} ± {overall_std_auroc:.4f}")
    print(f"总体平均 AUPR: {overall_mean_aupr:.4f} ± {overall_std_aupr:.4f}")
    
    # 保存总体结果
    overall_results = {
        'overall_mean_auroc': overall_mean_auroc,
        'overall_std_auroc': overall_std_auroc,
        'overall_mean_aupr': overall_mean_aupr,
        'overall_std_aupr': overall_std_aupr,
        'num_valid_runs': len(valid_auroc)
    }
    overall_df = pd.DataFrame([overall_results])
    overall_df.to_csv(os.path.join(RESULT_DIR, "overall_results.csv"), index=False)
else:
    print("没有有效的运行结果")

# 清除最终进度文件
if os.path.exists(os.path.join(CHECKPOINT_DIR, 'training_progress.json')):
    os.remove(os.path.join(CHECKPOINT_DIR, 'training_progress.json'))

print(f"\n所有运行完成！详细结果已保存到相应目录。")
print(f"总共保存了 {3 * 3 * 3} = 27 组labels和preds文件")
