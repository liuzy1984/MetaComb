#考虑边属性、注释图batch，修改细胞系编码，gcn第三层不加relu

import os
import torch
import numpy as np
import pandas as pd
import pdb
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

#载入数据
drug_features =  pd.read_csv("data/4130_drug_smiles_cid.csv")
cuda=True
device = torch.device('cuda:1'if torch.cuda.is_available() else "cpu")
OUTPUT_BASE_DIR = "/patient_css10sum5_adapt"

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
        # edge_embedding = self.edge_mlp(data.edge_attr)

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

        # # 为 batch 添加 batch index（单图视为 batch=0）
        # d1_graph.batch = torch.zeros(d1_graph.x.size(0), dtype=torch.long)
        # d2_graph.batch = torch.zeros(d2_graph.x.size(0), dtype=torch.long)

        # 细胞表达信息
        c_ge = torch.tensor(np.array(self.cell_features.iloc[cell][:]), dtype=torch.float)

        sample = {
            'd1': d1_graph,
            'd2': d2_graph,
            'cell': torch.tensor(cell),
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



        

#3个病人few-show css和S的预测

patient_3_features = pd.read_csv("data/log2_tpm_patient_zscorebyrich.csv",index_col=0)  #表达谱

patient_3 = pd.read_pickle("data/patient3_CSS_Ssum_35_0518.pickle")  #金标准

result = patient_3.groupby('Sample ID')['label'].value_counts()

#print(patient_3_features.shape)
#print(patient_3.head())


pa_tasks = []
pas = patient_3['Sample ID'].unique()
for pa in pas:
    task = patient_3[patient_3['Sample ID'] == pa]
    pa_tasks.append(task)

    
def binarize_predictions(preds, best_threshold):
    new_preds = [1 if pred >= best_threshold else 0 for pred in preds]
    return new_preds

    
def prepare_pa_tasks(random_state):
    """根据给定的random_state准备任务划分"""
    all_pa_task = []
    for i in range(0, len(pa_tasks)):
        pa_task = pa_tasks[i]
        train, test = train_test_split(pa_task, test_size=0.5, random_state=random_state)
        temp_dict = {
            'train': train,
            'test': test,
        }
        all_pa_task.append(temp_dict)
    return all_pa_task


ADAPT_STEPS = 5  # few-shot adaptation只更新少量梯度步；需要调整步数时改这里
ADAPT_LR = 0.001


def few_shot_adapt(net_meta, support_loader, criterion, optimizer, adapt_steps=ADAPT_STEPS):
    """在support set上做固定步数的few-shot adaptation。"""
    net_meta.train()
    support_iter = iter(support_loader)

    for step in range(adapt_steps):
        try:
            sample = next(support_iter)
        except StopIteration:
            support_iter = iter(support_loader)
            sample = next(support_iter)

        if sample.get('skip', False):
            continue

        support_d1 = sample['d1'].to(device)
        support_d2 = sample['d2'].to(device)
        support_cell = sample['cell'].to(device)
        support_c_ge = sample['c_ge'].float().to(device)
        support_label = sample['label'].float().to(device)

        optimizer.zero_grad()
        support_pred = net_meta(support_d1, support_d2, (support_cell, support_c_ge))
        support_loss = criterion(support_pred, support_label.view(-1, 1))
        support_loss.backward()
        optimizer.step()


#### 输出预测值，用于画ROC
def new_output_print(task_list, model_path, run_id, random_state):  
    preds = []
    labels = []

    for j, d in enumerate(task_list):
        net_meta = Comb().to(device)
        net_meta.load_state_dict(torch.load(model_path, weights_only=True))
        criterion = nn.BCELoss()
        optimizer = optim.Adam(net_meta.parameters(), lr=ADAPT_LR)

        support_set = d['train']
        querry_set = d['test']

        # ✅ 保存目标域的adapt数据和测试数据（只在第一个权重run时保存）
        if run_id == 1:
            save_dir = os.path.join(OUTPUT_BASE_DIR, f"rd{random_state}_result", "data")
            os.makedirs(save_dir, exist_ok=True)

            support_path = os.path.join(save_dir, f"target_domain_support_task{j}.csv")
            query_path = os.path.join(save_dir, f"target_domain_query_task{j}.csv")

            support_set.to_csv(support_path, index=False)
            querry_set.to_csv(query_path, index=False)

            print(f"✅ 已保存目标域任务 {j} 的adapt数据到 {support_path}")
            print(f"✅ 已保存目标域任务 {j} 的测试数据到 {query_path}")

        supportdata = DrugCombDataset(support_set, drug_features, patient_3_features)
        querrydata = DrugCombDataset(querry_set, drug_features, patient_3_features)
        support_loader = DataLoader(supportdata, shuffle=True, batch_size=len(support_set), collate_fn=collate_skip_none)
        querry_loader = DataLoader(querrydata, shuffle=True, batch_size=len(querry_set), collate_fn=collate_skip_none)

        # few-shot adapt：每个病人任务只更新固定的少量梯度步
        few_shot_adapt(net_meta, support_loader, criterion, optimizer)

        # 测试
        net_meta.eval()
        with torch.no_grad():
            for iteration, sample in enumerate(querry_loader):
                if sample.get('skip', False):
                    continue

                querry_d1 = sample['d1'].to(device)
                querry_d2 = sample['d2'].to(device)
                querry_cell = sample['cell'].to(device)
                querry_c_ge = sample['c_ge'].float().to(device)
                querry_label = sample['label'].float().to(device)

                querry_pred = net_meta(querry_d1, querry_d2, (querry_cell, querry_c_ge))

                preds.extend(querry_pred.cpu().numpy().ravel().tolist())
                labels.extend(querry_label.cpu().numpy().ravel().tolist())

    # ✅ 返回few-shot adapt后的预测结果
    return np.array(labels, dtype=int), np.array(preds, dtype=float)
    

# 定义要测试的random_state列表
random_states = list(range(10))

# 定义三个权重文件的路径
model_paths = [
    "/cell_adapt_result/result/run_1/gcn_maml_run_1_best.pth",
    "/cell_adapt_result/result/run_2/gcn_maml_run_2_best.pth", 
    "/cell_adapt_result/result/run_3/gcn_maml_run_3_best.pth"
]

# 存储所有结果
all_results = []

# 对每个random_state进行测试
for random_state in random_states:
    print(f"\n{'='*60}")
    print(f"开始测试 random_state = {random_state}")
    print(f"{'='*60}")
    
    # 创建对应的结果目录
    result_dir = os.path.join(OUTPUT_BASE_DIR, f"rd{random_state}_result")
    os.makedirs(result_dir, exist_ok=True)
    
    # 准备任务划分
    all_pa_task = prepare_pa_tasks(random_state)
    
    # 依次使用三个权重文件进行测试
    results = []
    for run_id, model_path in enumerate(model_paths, 1):
        print(f"\n{'='*50}")
        print(f"使用第 {run_id} 次训练权重进行测试 (random_state={random_state})")
        print(f"权重文件: {model_path}")
        print(f"{'='*50}")
        
        # 检查权重文件是否存在
        if not os.path.exists(model_path):
            print(f"❌ 警告: 权重文件不存在: {model_path}")
            continue
            
        # 进行测试
        labels, preds = new_output_print(all_pa_task, model_path, run_id, random_state)
        
        # 计算评估指标
        fpr, tpr, thresholds = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(labels, preds)
        aupr = auc(recall, precision)
        
        print(f"第 {run_id} 次测试结果 (random_state={random_state}):")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"AUPR: {aupr:.4f}")
        
        # 保存结果到DataFrame
        result_df = pd.DataFrame({
            'label': labels, 
            'prediction': preds,
            'run_id': run_id
        })
        
        # 保存到文件
        output_path = f'{result_dir}/gcn_maml_run_{run_id}_3patient_results.csv'
        result_df.to_csv(output_path, index=False)
        print(f"✅ 第 {run_id} 次测试结果已保存到: {output_path}")
        
        # 保存评估指标
        metrics_df = pd.DataFrame({
            'run_id': [run_id],
            'roc_auc': [roc_auc],
            'aupr': [aupr],
            'model_path': [model_path],
            'random_state': [random_state]
        })
        
        metrics_path = f'{result_dir}/gcn_maml_run_{run_id}_3patient_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✅ 第 {run_id} 次评估指标已保存到: {metrics_path}")
        
        # 保存到结果列表
        results.append({
            'run_id': run_id,
            'labels': labels,
            'preds': preds,
            'roc_auc': roc_auc,
            'aupr': aupr,
            'result_df': result_df,
            'random_state': random_state
        })
    
    # 汇总当前random_state的所有结果
    print(f"\n{'='*50}")
    print(f"random_state = {random_state} 的所有测试结果汇总:")
    print(f"{'='*50}")
    
    # 创建汇总表格
    summary_data = []
    for result in results:
        summary_data.append({
            'Run ID': result['run_id'],
            'ROC AUC': f"{result['roc_auc']:.4f}",
            'AUPR': f"{result['aupr']:.4f}",
            'Random State': result['random_state']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    
    # 保存汇总结果
    summary_path = f'{result_dir}/gcn_maml_all_runs_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ random_state={random_state} 的所有测试结果汇总已保存到: {summary_path}")
    
    # 计算当前random_state的平均性能
    if results:
        avg_roc_auc = np.mean([r['roc_auc'] for r in results])
        avg_aupr = np.mean([r['aupr'] for r in results])
        print(f"\nrandom_state={random_state} 的平均性能:")
        print(f"平均 ROC AUC: {avg_roc_auc:.4f}")
        print(f"平均 AUPR: {avg_aupr:.4f}")
        
        # 保存平均性能
        avg_metrics_df = pd.DataFrame({
            'metric': ['avg_roc_auc', 'avg_aupr'],
            'value': [avg_roc_auc, avg_aupr],
            'random_state': [random_state, random_state]
        })
        avg_metrics_path = f'{result_dir}/gcn_maml_average_metrics.csv'
        avg_metrics_df.to_csv(avg_metrics_path, index=False)
        print(f"✅ random_state={random_state} 的平均性能指标已保存到: {avg_metrics_path}")
        
        # 保存到总结果列表
        all_results.extend(results)

# 计算所有random_state的总体平均性能
print(f"\n{'='*60}")
print("所有random_state的总体平均性能:")
print(f"{'='*60}")

if all_results:
    # 按random_state分组计算平均
    random_state_groups = {}
    for result in all_results:
        rs = result['random_state']
        if rs not in random_state_groups:
            random_state_groups[rs] = []
        random_state_groups[rs].append(result)
    
    # 计算每个random_state的平均值
    overall_summary = []
    for rs, results in random_state_groups.items():
        avg_roc_auc = np.mean([r['roc_auc'] for r in results])
        avg_aupr = np.mean([r['aupr'] for r in results])
        overall_summary.append({
            'random_state': rs,
            'avg_roc_auc': avg_roc_auc,
            'avg_aupr': avg_aupr
        })
    
    # 计算所有random_state的总平均
    total_avg_roc_auc = np.mean([s['avg_roc_auc'] for s in overall_summary])
    total_avg_aupr = np.mean([s['avg_aupr'] for s in overall_summary])
    
    print(f"\n所有random_state的总体平均性能:")
    print(f"总平均 ROC AUC: {total_avg_roc_auc:.4f}")
    print(f"总平均 AUPR: {total_avg_aupr:.4f}")
    
    # 保存总体汇总
    overall_summary_df = pd.DataFrame(overall_summary)
    overall_summary_path = os.path.join(OUTPUT_BASE_DIR, 'gcn_maml_overall_summary.csv')
    overall_summary_df.to_csv(overall_summary_path, index=False)
    print(f"✅ 所有random_state的汇总结果已保存到: {overall_summary_path}")
    
    # 保存总平均性能
    total_avg_df = pd.DataFrame({
        'metric': ['total_avg_roc_auc', 'total_avg_aupr'],
        'value': [total_avg_roc_auc, total_avg_aupr]
    })
    total_avg_path = os.path.join(OUTPUT_BASE_DIR, 'gcn_maml_total_average_metrics.csv')
    total_avg_df.to_csv(total_avg_path, index=False)
    print(f"✅ 总平均性能指标已保存到: {total_avg_path}")

print(f"\n{'='*60}")
print("所有测试完成!")
print(f"{'='*60}")
