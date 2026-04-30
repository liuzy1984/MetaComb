

import os
import copy
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
from torch.utils.data import DataLoader, Subset,random_split, ConcatDataset,Dataset
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
from rdkit. Chem import rdMolDescriptors
import random
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from rdkit. Chem import Descriptors
#from torch_geometric.loader import DataLoader as GeometricDataLoader
from rdkit. Chem.rdFingerprintGenerator import GetMorganGenerator
from torch.utils.data.dataloader import default_collate
from rdkit import DataStructs
from torch_geometric.data import Batch

#预定义
data_path = '/data'
cell_feature_path = 'data/log2_tpm_cell.csv'
save_path = '5fold_new/'
cuda=True
device = torch.device('cuda:2'if torch.cuda.is_available() else "cpu")

# 创建保存目录
os.makedirs(save_path, exist_ok=True)


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
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        d1 = self.df.iloc[idx, 0]
        d2 = self.df.iloc[idx, 1]
        cell = self.df.iloc[idx, 2]
        label = self.df.iloc[idx, 3]
    
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

        # 细胞表达信息：读取当前 fold 基于训练集统计量标准化后的特征
        c_ge_np = np.array(self.cell_features.loc[cell][:], dtype=np.float32)
        if not np.all(np.isfinite(c_ge_np)):
            print(f"检测到非有限细胞特征，cell index={cell}")
        c_ge = torch.tensor(c_ge_np, dtype=torch.float)

        sample = {
            'd1': d1_graph,
            'd2': d2_graph,
            'cell': torch.tensor(cell),
            'c_ge': c_ge,
            'label': torch.tensor(label, dtype=torch.float)
        }

        return sample


#预训练
#源域：细胞系的源于数据
df =  pd.read_pickle("/data/cl_label_data_onlysmiles_1016.pickle") 
df = df.drop(columns = ['css_ri','S_sum'])
drug_features =  pd.read_csv("/data/4130_drug_smiles_cid.csv")
cell_features = pd.read_csv(cell_feature_path, index_col=0).apply(pd.to_numeric, errors='coerce')
singledrug = df['cell_line_name'].value_counts().reset_index()
singledrug.columns = ['cell_line_name','frequency']
singledrug_s = singledrug[singledrug['frequency'] > 50]
singledrug_t = singledrug[(singledrug['frequency'] <= 50)&(singledrug['frequency'] >= 10) ]
source_data = df[df['cell_line_name'].isin(singledrug_s['cell_line_name'])].reset_index(drop=True)
target_data = df[df['cell_line_name'].isin(singledrug_t['cell_line_name'])].reset_index(drop=True)




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


def build_fold_scaled_cell_features(base_cell_features, train_df):
    train_cell_ids = pd.to_numeric(train_df.iloc[:, 2], errors='raise').astype(int).to_numpy()
    unique_train_cell_ids = np.unique(train_cell_ids)
    train_cell_expr = base_cell_features.loc[unique_train_cell_ids]

    train_mean = train_cell_expr.mean(axis=0)
    train_var = train_cell_expr.var(axis=0, ddof=1)
    train_std = np.sqrt(train_var)

    safe_std = train_std.copy()
    invalid_std = (~np.isfinite(safe_std)) | (safe_std == 0)
    safe_std[invalid_std] = 1.0

    scaled_features = base_cell_features.subtract(train_mean, axis=1).divide(safe_std, axis=1)

    if invalid_std.any():
        scaled_features.loc[:, invalid_std] = 0.0

    return scaled_features.fillna(0.0)


#五倍交叉验证检测源域模型的效果
def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    valid_batches = 0

    for iteration, sample in enumerate(dataloader):
        if 'skip' in sample:
            continue
        valid_batches += 1

        d1 = sample['d1'].to(device)
        d2 = sample['d2'].to(device)
        cell = sample['cell'].to(device)
        c_ge = sample['c_ge'].float().to(device)
        label = sample['label'].float().to(device)

        optimizer.zero_grad() 
        y_pred = model(d1, d2, (cell,c_ge))

        loss = criterion(y_pred, label.view(-1,1)) 

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if valid_batches == 0:
        return 0.0
    return running_loss / valid_batches



def validate(model, dataloader, criterion, optimizer):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    valid_batches = 0

    with torch.no_grad():
        for iteration, sample in enumerate(dataloader):
            if 'skip' in sample:
                continue
            valid_batches += 1

            d1 = sample['d1'].to(device)
            d2 = sample['d2'].to(device)
            cell = sample['cell'].to(device)
            c_ge = sample['c_ge'].float().to(device)
            label = sample['label'].float().to(device)

            y_pred = model(d1, d2, (cell, c_ge))
            loss = criterion(y_pred, label.view(-1,1))
            running_loss += loss.item()
            all_outputs.extend(y_pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    if valid_batches == 0:
        return 0.0, np.array([]), np.array([])

    return running_loss / valid_batches, np.array(all_outputs), np.array(all_labels)


def main():
    num_samples = len(source_data)

    # 存储所有重复实验的结果
    all_repeat_fpr = []
    all_repeat_tpr = []
    all_repeat_precision = []
    all_repeat_recall = []
    all_repeat_auc = []
    all_repeat_aupr = []
    
    # 存储每次重复的五折平均AUC和AUPR
    repeat_mean_auc_list = []
    repeat_mean_aupr_list = []
    
    # 存储所有重复和所有fold的详细结果
    all_fold_results = []

    # 重复 10 次五倍交叉验证
    num_repeats = 10
    for repeat in range(0,num_repeats):
        print(f"\n=== Repeat {repeat + 1}/{num_repeats} ===")

        kfold = KFold(n_splits=5, shuffle=True, random_state=42+repeat)  # 使用不同的随机种子
        fold = 0
        repeat_fpr = []
        repeat_tpr = []
        repeat_precision = []
        repeat_recall = []
        repeat_auc = []
        repeat_aupr = []
        
        # 存储当前重复中每一折的预测结果
        fold_results = []

        for train_indices, val_indices in kfold.split(np.arange(num_samples)):
            fold += 1
            print(f"Fold {fold}")

            train_df = source_data.iloc[train_indices].reset_index(drop=True)
            val_df = source_data.iloc[val_indices].reset_index(drop=True)

            fold_cell_features = build_fold_scaled_cell_features(cell_features, train_df)

            train_set = DrugCombDataset(train_df, drug_features, fold_cell_features)
            val_set = DrugCombDataset(val_df, drug_features, fold_cell_features)
            train_loader = DataLoader(train_set, batch_size=1280, shuffle=True,num_workers=16, collate_fn=collate_skip_none)
            val_loader = DataLoader(val_set, batch_size=1280, shuffle=False, num_workers=16,collate_fn=collate_skip_none)

            model = Comb()
            model = model.to(device)
            criterion = nn. BCELoss()
            optimizer = optim. Adam(model.parameters(), lr=0.001)

            num_epochs = 300
            # 早停机制参数
            patience = 20
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            for epoch in range(num_epochs):
                train_loss = train(model, train_loader, criterion, optimizer)
                val_loss, val_outputs, val_labels = validate(model, val_loader, criterion, optimizer)
                
                # 早停机制检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型状态
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f"Epoch {epoch + 1}: New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"Epoch {epoch + 1}: Validation loss did not improve. Patience: {patience_counter}/{patience}")
                
                if (epoch + 1) % 50 == 0:
                    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # 检查是否早停
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1} after {patience} epochs without improvement.")
                    # 恢复最佳模型
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            # 最终验证使用最佳模型
            val_loss, val_outputs, val_labels = validate(model, val_loader, criterion, optimizer)

            # 计算每个 fold 的指标
            auc_score = roc_auc_score(val_labels, val_outputs)
            precision, recall, _ = precision_recall_curve(val_labels, val_outputs)
            aupr_score = auc(recall, precision)

            fpr, tpr, _ = roc_curve(val_labels, val_outputs)

            repeat_fpr.append(fpr)
            repeat_tpr.append(tpr)
            repeat_precision.append(precision)
            repeat_recall.append(recall)
            repeat_auc.append(auc_score)
            repeat_aupr.append(aupr_score)
            
            # 保存当前折的预测结果和标签到DataFrame
            fold_df = pd.DataFrame({
                'repeat': repeat + 1,
                'fold': fold,
                'prediction': val_outputs.flatten(),
                'label': val_labels,
                'auc': auc_score,
                'aupr': aupr_score
            })
            fold_results.append(fold_df)
            all_fold_results.append(fold_df)

            print(f'Fold {fold} AUC: {auc_score:.4f}, AUPR: {aupr_score:.4f}')

        # 计算当前重复的五折平均AUC和AUPR
        mean_auc = np.mean(repeat_auc)
        mean_aupr = np.mean(repeat_aupr)
        repeat_mean_auc_list.append(mean_auc)
        repeat_mean_aupr_list.append(mean_aupr)
        
        print(f"Repeat {repeat + 1} completed.")
        print(f"Mean AUC over 5 folds: {mean_auc:.4f}")
        print(f"Mean AUPR over 5 folds: {mean_aupr:.4f}")

        # 保存当前重复的所有折结果为CSV
        repeat_df = pd.concat(fold_results, ignore_index=True)
        repeat_filename = os.path.join(save_path, f'repeat_{repeat+1}_results.csv')
        repeat_df.to_csv(repeat_filename, index=False)
        print(f"Saved fold results for repeat {repeat+1} to {repeat_filename}")

        # 存储每次重复实验的结果
        all_repeat_fpr.append(repeat_fpr)
        all_repeat_tpr.append(repeat_tpr)
        all_repeat_precision.append(repeat_precision)
        all_repeat_recall.append(repeat_recall)
        all_repeat_auc.append(mean_auc)
        all_repeat_aupr.append(mean_aupr)

    # 保存所有fold的汇总结果
    all_results_df = pd.concat(all_fold_results, ignore_index=True)
    all_results_filename = os.path.join(save_path, 'all_repeats_results.csv')
    all_results_df.to_csv(all_results_filename, index=False)
    print(f"Saved all fold results to {all_results_filename}")

    # 保存每次重复的汇总统计
    summary_df = pd.DataFrame({
        'repeat': range(1, num_repeats + 1),
        'mean_auc': repeat_mean_auc_list,
        'mean_aupr': repeat_mean_aupr_list
    })
    summary_filename = os.path.join(save_path, 'repeats_summary.csv')
    summary_df.to_csv(summary_filename, index=False)
    print(f"Saved repeats summary to {summary_filename}")

    # 输出每次重复的五折平均AUC和AUPR
    print(f"\n=== Summary of 10 Repeats ===")
    for i in range(num_repeats):
        print(f"Repeat {i+1}: Mean AUC = {repeat_mean_auc_list[i]:.4f}, Mean AUPR = {repeat_mean_aupr_list[i]:.4f}")

    # 计算平均曲线
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    # 计算平均 AUROC 曲线
    mean_tpr = np.zeros_like(mean_fpr)
    for repeat_fpr, repeat_tpr in zip(all_repeat_fpr, all_repeat_tpr):
        for fpr, tpr in zip(repeat_fpr, repeat_tpr):
            mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr /= (num_repeats * 5)  # 10 次重复 * 5 折

    # 计算平均 AUPR 曲线
    mean_precision = np.zeros_like(mean_recall)
    for repeat_recall, repeat_precision in zip(all_repeat_recall, all_repeat_precision):
        for recall, precision in zip(repeat_recall, repeat_precision):
            # 注意：precision_recall_curve 返回的 recall 是递减的，需要反转
            precision_interp = interp(mean_recall, recall[::-1], precision[::-1])
            mean_precision += precision_interp
    mean_precision /= (num_repeats * 5)

    # 计算总体平均 AUC 和 AUPR
    overall_mean_auc = np.mean(all_repeat_auc)
    overall_mean_aupr = np.mean(all_repeat_aupr)
    overall_std_auc = np.std(all_repeat_auc)
    overall_std_aupr = np.std(all_repeat_aupr)

    print(f"\n=== Final Results ===")
    print(f"Overall Mean AUC: {overall_mean_auc:.4f} ± {overall_std_auc:.4f}")
    print(f"Overall Mean AUPR: {overall_mean_aupr:.4f} ± {overall_std_aupr:.4f}")

    # 保存结果
    results = {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'mean_recall': mean_recall,
        'mean_precision': mean_precision,
        'overall_mean_auc': overall_mean_auc,
        'overall_std_auc': overall_std_auc,
        'overall_mean_aupr': overall_mean_aupr,
        'overall_std_aupr': overall_std_aupr,
        'all_repeat_auc': all_repeat_auc,
        'all_repeat_aupr': all_repeat_aupr,
        'repeat_mean_auc_list': repeat_mean_auc_list,
        'repeat_mean_aupr_list': repeat_mean_aupr_list
    }

    with open(os.path.join(save_path, 'cross_validation_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    return results

main()
