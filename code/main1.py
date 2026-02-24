from sklearn.metrics import matthews_corrcoef
import numpy as np
import copy
import math
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import average_precision_score
from numpy.core import multiarray
from torch.nn.parameter import Parameter
import random
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from model import *
from utils import f1_score_binary, precision_binary, recall_binary, accuracy_binary
import scipy.sparse as sp
# import plot_auc_curves
from scipy.stats import gaussian_kde
# 以下两句用来忽略版本错误信息
import warnings
from sklearn.manifold import TSNE
import seaborn as sns

import os
warnings.filterwarnings("ignore")
# 设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 48
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def multi_view_contrastive_loss(view_reprs, temperature=0.7):
    """
    view_reprs: list of tensors, each shape (N, D)
    Computes average pairwise NT-Xent loss across all view pairs.
    """
    if len(view_reprs) < 2:
        return torch.tensor(0.0, device=view_reprs[0].device)

    loss = 0.0
    n = len(view_reprs)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            z1 = F.normalize(view_reprs[i], dim=1)
            z2 = F.normalize(view_reprs[j], dim=1)
            logits = torch.mm(z1, z2.t()) / temperature
            labels = torch.arange(z1.size(0), device=z1.device)
            loss += F.cross_entropy(logits, labels)
            count += 1
    return loss / count



def laplacian_norm(adj):
    adj += np.eye(adj.shape[0])  # add self-loop
    degree = np.array(adj.sum(1))
    D = []
    for i in range(len(degree)):
        if degree[i] != 0:
            de = np.power(degree[i], -0.5)
            D.append(de)
        else:
            D.append(0)
    degree = np.diag(np.array(D))
    norm_A = degree.dot(adj).dot(degree)

    return norm_A


def cross_validation_5fold(k_folds):
    fold = int(totalassociation / k_folds)  # 1538

    auc = 0
    aupr = 0
    rec = 0
    pre = 0
    f1 = 0
    acc = 0
    mcc = 0
    tprs = []
    fprs = []
    aucs = []
    precisions = []
    recalls = []
    auprs = []
    loss_lists = []
    accuracy_lists = []
    mcc_lists = []

    # 五折交叉验证开始
    for f in range(1, k_folds + 1):
        print('%d fold:' % (f))
        if f == k_folds:
            testset = shuffle_data[((f - 1) * fold): totalassociation + 1]
        else:
            testset = shuffle_data[((f - 1) * fold): f * fold]


        auc1, aupr1, recall1, precision1, f11, acc1, mcc1, loss_list, accuracy_list, mcc_list, all_recall, all_precision, all_aupr, recall, precision = train(
            testset, epochs)
        precisions.append(precision1)
        recalls.append(recall1)
        aucs.append(auc1)
        auprs.append(aupr1)
        loss_lists.append(loss_list)
        accuracy_lists.append(accuracy_list)
        mcc_lists.append(mcc_list)

        auc = auc + auc1
        aupr = aupr + aupr1
        rec = rec + recall1
        pre = pre + precision1
        f1 = f1 + f11
        acc = acc + acc1
        mcc = mcc + mcc1

    auc2 = auc / k_folds
    aupr2 = aupr / k_folds
    pre2 = pre / k_folds
    rec2 = rec / k_folds
    f1_2 = f1 / k_folds
    acc2 = acc / k_folds
    mcc2 = mcc / k_folds
    print("cv_mean:")
    print('auc: {:.4f}, aupr: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1_score: {:.4f}, acc: {:.4f}, mcc: {:.4f}'
          .format(auc2, aupr2, pre2, rec2, f1_2, acc2, mcc2))

    metric = ["{:.4f}".format(v) for v in [auc2, aupr2, pre2, rec2, f1_2, acc2, mcc2]]

    return metric, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists, all_recall, all_precision, all_aupr, recall, precision


def train(testset, epochs):
    all_f = np.random.permutation(np.size(Index_zeroRow))
    test_p = list(testset)
    test_f = all_f[0:len(test_p)]
    difference_set_f = list(set(all_f).difference(set(test_f)))
    train_f = difference_set_f

    X = copy.deepcopy(MD)
    Xn = copy.deepcopy(X)
    zero_index = []
    for ii in range(len(train_f)):
        zero_index.append([Index_zeroRow[train_f[ii]], Index_zeroCol[train_f[ii]]])

    true_list = multiarray.zeros((len(test_p) + len(test_f), 1))
    for ii in range(len(test_p)):
        Xn[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
        true_list[ii, 0] = 1
    train_mask = np.ones(shape=Xn.shape)
    for ii in range(len(test_p)):
        train_mask[Index_PositiveRow[testset[ii]], Index_PositiveCol[testset[ii]]] = 0
        train_mask[Index_zeroRow[test_f[ii]], Index_zeroCol[test_f[ii]]] = 0
    train_mask_tensor = torch.from_numpy(train_mask).to(torch.bool)
    train_mask_tensor = train_mask_tensor.to(device)
    label = true_list

    model = Gai_HGNN()
    optimizer2 = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    def duo_hg():
        A = copy.deepcopy(Xn)
        N_rna = 308
        N_drug = 62


        def similarity_to_edge_index(sim_mat, threshold):
            rows, cols = np.where(sim_mat > threshold)
            edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
            return edge_index

        # RNA 视图边
        rna_edge_seq = similarity_to_edge_index(S_rna_seq, threshold=0.7)
        rna_edge_text = similarity_to_edge_index(S_rna_text, threshold=0.7)

        # Drug 视图边（注意偏移）
        drug_offset = N_rna
        drug_edge_seq = similarity_to_edge_index(S_drug_seq, threshold=0.7) + drug_offset
        drug_edge_text = similarity_to_edge_index(S_drug_text, threshold=0.7) + drug_offset
        drug_edge_img = similarity_to_edge_index(S_drug_img, threshold=0.7) + drug_offset

        # 整体
        edge_indices_rna = [rna_edge_seq, rna_edge_text]
        edge_indices_drug = [drug_edge_seq, drug_edge_text, drug_edge_img]


        # ✅ 修复：逐个移到 device
        edge_indices_rna = [e.to(device) for e in edge_indices_rna]
        edge_indices_drug = [e.to(device) for e in edge_indices_drug]

        rows, cols = np.where(A == 1)  # MD 是关联矩阵
        inter_edge = torch.tensor([rows, cols], dtype=torch.long).to(device)

        A = torch.tensor(A, dtype=torch.float32).to(device)  # 也别忘了 A 要转 tensor + to(device)
        x = np.concatenate([rna_features, drug_features], axis=0)
        x = torch.tensor(x, dtype=torch.float32).to(device)



        return edge_indices_rna, edge_indices_drug, A, N_rna,x,inter_edge  # 注意：返回 N_rna


    edge_indices_rna, edge_indices_drug, A, N_rna,x,inter_edge,S_RNA66_fusion,S_DRUG66_fusion = duo_hg()
    pos_weight = float(A.shape[0] * A.shape[1] - A.sum()) / A.sum()
    accuracy_list = []
    mcc_list = []
    loss_list = []
    for epoch in tqdm(range(epochs), desc='epochs'):
        model.train()
        # 消融-只保留属性
        rna, drug, recover = model(
            x=x,
            edge_indices_rna=edge_indices_rna,
            edge_indices_drug=edge_indices_drug,
            inter_edge=inter_edge,  # ← 关键：传入已知关联边
            N_rna=N_rna,
        S_RNA66_fusion= S_RNA66_fusion,
        S_DRUG66_fusion= S_DRUG66_fusion)


        outputs = recover.t().cpu().detach().numpy()
        test_predict = create_resultlist(outputs, testset, Index_PositiveRow, Index_PositiveCol, Index_zeroRow,
                                         Index_zeroCol, len(test_p), zero_length, test_f)
        MA = torch.masked_select(A, train_mask_tensor)

        rec = torch.masked_select(recover.t(), train_mask_tensor)

        loss_1 = F.binary_cross_entropy_with_logits(rec.t(), MA, pos_weight=pos_weight)
        loss = loss_1
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        auc_val = roc_auc_score(label, test_predict)
        aupr_val = average_precision_score(label, test_predict)
        print('Epoch: {:04d},loss: {:.5f},auc_val: {:.5f},aupr_val: {:.5f}'
              .format(epoch + 1, loss.data.item(), auc_val, aupr_val))
        loss_list.append(loss.data.item())
        max_f1_score, threshold = f1_score_binary(torch.from_numpy(label).float(),
                                                  torch.from_numpy(test_predict).float())
        print("max_f1_score", max_f1_score)
        precision = precision_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("precision:", precision)
        recall = recall_binary(torch.from_numpy(label).float(), torch.from_numpy(test_predict).float(), threshold)
        print("recall:", recall)
        test_predict = np.array(test_predict)  # Ensure it's a NumPy array
        # Convert test_predict to a tensor to match the type with threshold
        test_predict_tensor = torch.from_numpy(test_predict)
        # Accuracy calculation using tensors
        binary_predictions = (test_predict_tensor >= threshold).float()
        accuracy = (binary_predictions == torch.from_numpy(label).float()).float().mean().item()
        # Append metrics to lists
        accuracy_list.append(accuracy)
        # MCC calculation
        mcc = matthews_corrcoef(torch.from_numpy(label).numpy(), binary_predictions.numpy())
        mcc_list.append(mcc)
        print("Accuracy:", accuracy)
        print("MCC:", mcc)

    fpr, tpr = [], []
    print("train end!")

    auc1 = auc_val
    aupr1 = aupr_val
    recall1 = recall
    precision1 = precision
    f11 = max_f1_score
    acc1 = accuracy
    mcc1 = mcc



    precision, recall, thresholds = precision_recall_curve(label, test_predict)

    # Interpolate the recall-precision values for consistent comparison across folds
    interp_recall = np.linspace(0, 1, 100)  # Uniformly spaced recall values
    interp_precision = np.interp(interp_recall, recall[::-1],
                                 precision[::-1])  # Reverse recall and precision for interpolation
    interp_precision[0] = 1.0  # Ensure the starting precision is 1.0 for recall = 0

    # Store interpolated values
    all_precision.append(precision)
    all_recall.append(recall)

    # # Calculate AUPR for this fold
    aupr = auc(recall, precision)
    all_aupr.append(aupr)
    for i in range(len(recall)):
        if recall[i] == 1:
             precision[i] = 0
    # aupr = auc(recall, precision)
    return auc1, aupr1, recall1, precision1, f11, acc1, mcc1, loss_list, accuracy_list, mcc_list, all_recall, all_precision, all_aupr, recall, precision

# 主函数
if __name__ == '__main__':
    # 读数据
    #数据集1
    # 关联矩阵
    MD = np.loadtxt("dataset1/association.txt")
    # 药物文本相似性矩阵
    S_drug_text = np.loadtxt("dataset1/LLM_drug_sim.txt")
    # RNA文本相似性矩阵
    S_rna_text = np.loadtxt("dataset1/LLM_rna_sim.txt")
    # 药物序列相似性矩阵
    S_drug_seq = np.loadtxt("dataset1/Seq_drug_sim.txt")
    # RNA序列相似性矩阵
    S_rna_seq = np.loadtxt("dataset1/Seq_rna_sim.txt")
    # 药物序列相似性矩阵
    S_drug_img = np.loadtxt("dataset1/Img_drug_sim.txt")


    def kernel_to_node_features(K, target_dim):
        """
        从 N×N 高斯核矩阵 K 中提取 N×d 节点特征
        """
        N = K.shape[0]
        # 确保 target_dim <= N
        d = min(target_dim, N)

        # 1. 特征分解 (K = U @ diag(λ) @ U^T)
        # 注意: eigh 适用于对称矩阵，返回升序特征值
        eigvals, eigvecs = np.linalg.eigh(K)

        # 2. 取最大的 d 个特征值对应的特征向量
        # argsort 升序 -> 取最后 d 个（最大）
        idx = np.argsort(eigvals)[::-1][:d]  # 降序索引
        node_features = eigvecs[:, idx]  # (N, d)

        # 3. (可选) L2 归一化，使不同模态尺度一致
        node_features = node_features / np.linalg.norm(node_features, axis=1, keepdims=True)

        return node_features.astype(np.float32)




    [row, col] = np.shape(MD)  # 获取MD数组的形状，即行数和列数
    # 识别出MD数组中值为0和值为1的元素位置，代表不同的类别或状态（如正样本和负样本）。
    indexn = np.argwhere(MD == 0)  # 找出MD数组中所有值为0的元素的位置，返回一个数组indexn，其中包含这些元素的行和列索引。
    Index_zeroRow = indexn[:, 0]  # 从indexn中提取所有行索引
    Index_zeroCol = indexn[:, 1]  # 提取所有列索引
    indexp = np.argwhere(MD == 1)  # 找出MD数组中所有值为1的元素的位置，返回一个数组indexp，其中包含这些元素的行和列索引。
    Index_PositiveRow = indexp[:, 0]  # 提取所有行索引
    Index_PositiveCol = indexp[:, 1]  # 提取所有列索引
    zero_length = np.size(Index_zeroRow)  # 计算值为0的元素的数量（通过行索引的大小）。
    totalassociation = np.size(Index_PositiveRow)  # 计算值为1的元素的数量，即正样本的总数。
    shuffle_data = np.random.permutation(totalassociation)  # 对totalassociation（值为1的元素数量）进行随机排列。打乱正样本的顺序

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    all_fpr, all_tpr, all_auc = [], [], []
    all_precision, all_recall, all_aupr = [], [], []
    k_folds = 5
    # 定义模型超参数
    lr = 0.0023 # 可调
    p = 0.3  # 现在无了
    # k= 50
    weight_decay = 0.01# 可调
    temperature = 0.6
    epochs = 350
    # result, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists, all_fpr, all_tpr, all_auc, fpr, tpr = cross_validation_5fold(
    #     k_folds)
    result, aucs, precisions, recalls, auprs, loss_lists, accuracy_lists, mcc_lists, all_recall, all_precision, all_aupr, recall, precision = cross_validation_5fold(
        k_folds)
