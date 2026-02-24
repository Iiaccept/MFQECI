import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATConv
from torch_scatter import scatter_mean
import pennylane as qml
from pennylane import numpy as pnp  # 避免与numpy冲突
import math


# ========================
# 全局设置
# ========================
# 设置随机数种子
def set_random_seed(seed=48):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # 防止 Python hash 随机化
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ========================
# 解码器模块
# ========================
class decoder1(nn.Module):
    def __init__(self, dropout=0.5):
        super(decoder1, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_node, z_hyperedge):
        z = z_node.mm(z_hyperedge.t())
        return z


# ========================
# 多视图GNN编码器
# ========================
class MultiViewGNN_Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_views_rna=2, n_views_drug=3, num_heads=2, total_nodes=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_views_rna = n_views_rna
        self.n_views_drug = n_views_drug
        self.total_nodes = total_nodes

        # GAT layers for each view
        self.gat_rna = nn.ModuleList([
            GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
            for _ in range(n_views_rna)
        ])
        self.gat_drug = nn.ModuleList([
            GATConv(in_dim, hidden_dim // num_heads, heads=num_heads, concat=True)
            for _ in range(n_views_drug)
        ])

        # View-level attention
        self.rna_view_attn = nn.Linear(hidden_dim, 1)
        self.drug_view_attn = nn.Linear(hidden_dim, 1)

        # Optional node embedding
        if total_nodes is not None:
            self.node_emb = nn.Embedding(total_nodes, in_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)
        else:
            self.node_emb = None

    def forward(self, x, edge_indices_rna, edge_indices_drug, N_rna, return_views=True):
        if x is None:
            x = self.node_emb.weight

        # Encode RNA views
        rna_views = []
        for i, edge_index in enumerate(edge_indices_rna):
            out = self.gat_rna[i](x[:N_rna], edge_index)
            rna_views.append(out)

        # Encode Drug views
        drug_views = []
        for i, edge_index in enumerate(edge_indices_drug):
            out = self.gat_drug[i](x[N_rna:], edge_index - N_rna)
            drug_views.append(out)

        # Fuse RNA views
        rna_stack = torch.stack(rna_views, dim=1)
        rna_attn = F.softmax(self.rna_view_attn(rna_stack).squeeze(-1), dim=1).unsqueeze(-1)
        rna_final = (rna_stack * rna_attn).sum(dim=1)

        # Fuse Drug views
        drug_stack = torch.stack(drug_views, dim=1)
        drug_attn = F.softmax(self.drug_view_attn(drug_stack).squeeze(-1), dim=1).unsqueeze(-1)
        drug_final = (drug_stack * drug_attn).sum(dim=1)
        return rna_final, drug_final

# ========================
# 量子神经网络编码器
# ========================
class QNN_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_qubits=2, n_layers=2, dropout=0):
        super(QNN_Encoder, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 预处理网络
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, 2 * n_qubits),
            nn.ReLU(),
            nn.Linear(2 * n_qubits, n_qubits),
            nn.LayerNorm(n_qubits)
        )

        # 量子部分
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        def qnode_def(inputs, weights):
            # 角度嵌入
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")

            for j in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights[j, i, 0], wires=i)
                    qml.RY(weights[j, i, 1], wires=i)
                    qml.RZ(weights[j, i, 2], wires=i)

                # 环形纠缠
                if n_qubits > 1:
                    for i in range(n_qubits):
                        qml.CNOT(wires=[i, (i + 1) % n_qubits])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = qml.QNode(qnode_def, self.dev, interface="torch", diff_method="backprop")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.weight_shapes)

        # 后处理网络
        self.postprocess = nn.Sequential(
            nn.Linear(n_qubits, 2 * output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        w_init = torch.randn(self.weight_shapes["weights"]) * 0.1
        with torch.no_grad():
            self.qlayer.weights = torch.nn.Parameter(w_init)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {x.shape[1]}")

        # 预处理和缩放
        x = self.preprocess(x)
        x = torch.sigmoid(x) * math.pi

        # 量子层计算
        x = self.qlayer(x)

        # 后处理
        x = self.postprocess(x)
        return x


# ========================
# 量子跨域模块
# ========================
class QuantumCrossDomainModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_qubits=6, n_qnn_layers=3, dropout=0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 跨域更新网络
        self.rna_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.drug_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 量子增强模块
        self.qnn_rna = QNN_Encoder(
            input_dim=input_dim,
            output_dim=hidden_dim,
            n_qubits=n_qubits,
            n_layers=n_qnn_layers,
            dropout=dropout
        )
        self.qnn_drug = QNN_Encoder(
            input_dim=input_dim,
            output_dim=hidden_dim,
            n_qubits=n_qubits,
            n_layers=n_qnn_layers,
            dropout=dropout
        )

    def cross_domain_propagate(self, rna_emb, drug_emb, inter_edge):
        """跨域传播"""
        if inter_edge is None or inter_edge.numel() == 0:
            return rna_emb, drug_emb

        # Drug -> RNA
        drug_to_rna = scatter_mean(
            drug_emb[inter_edge[1]],
            inter_edge[0],
            dim=0,
            dim_size=rna_emb.size(0)
        )

        # RNA -> Drug
        rna_to_drug = scatter_mean(
            rna_emb[inter_edge[0]],
            inter_edge[1],
            dim=0,
            dim_size=drug_emb.size(0)
        )

        # 残差更新
        updated_rna = rna_emb + self.rna_update(drug_to_rna)
        updated_drug = drug_emb + self.drug_update(rna_to_drug)
        return updated_rna, updated_drug

    def forward(self, rna_emb, drug_emb, inter_edge):
        # 量子增强
        rna_refined = self.qnn_rna(rna_emb)
        drug_refined = self.qnn_drug(drug_emb)

        # 跨域消息传递
        rna_final, drug_final = self.cross_domain_propagate(
            rna_refined, drug_refined, inter_edge
        )

        return rna_final, drug_final


# ========================
# 主模型
# ========================
class Gai_HGNN(nn.Module):
    def __init__(self, num_in_node=129, num_in_edge=48, num_hidden1=128, num_out=128):
        super(Gai_HGNN, self).__init__()

        self.multiviewfusion = MultiViewGNN_Encoder(
            in_dim=36,
            hidden_dim=36,
            total_nodes=790
        )

        self.QuantumCross = QuantumCrossDomainModule(36, 24)

        self.decoder1 = decoder1()

    def forward(self, x, edge_indices_rna, edge_indices_drug, inter_edge, N_rna):
        # 多视图融合
        rna_final, drug_final = self.multiviewfusion(x, edge_indices_rna, edge_indices_drug, N_rna)

        # 量子跨域增强
        rna_final, drug_final = self.QuantumCross(rna_final, drug_final, inter_edge)

        # 解码重建
        reconstruction1 = self.decoder1(drug_final, rna_final)

        return rna_final, drug_final, reconstruction1


# ========================
# 辅助函数
# ========================
def create_resultlist(result, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow, Index_zeroCol,
                      test_length_p, zero_length, test_f):
    result_list = np.zeros((test_length_p + len(test_f), 1))
    for i in range(test_length_p):
        result_list[i, 0] = result[Index_PositiveRow[testset[i]], Index_PositiveCol[testset[i]]]
    for i in range(len(test_f)):
        result_list[i + test_length_p, 0] = result[Index_zeroRow[test_f[i]], Index_zeroCol[test_f[i]]]
    return result_list


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())