import numpy as np
import pickle as pkl
#import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import matplotlib as plt
from pylab import *
import random
from inits import *
#import pandas as pd
from sklearn.preprocessing import normalize
h = 1663
l = 258
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(train_arr, test_arr):
    """Load data."""
    labels = np.loadtxt("data/adj.csv", delimiter=',')
    logits_test = sp.csr_matrix((labels[test_arr,2],(labels[test_arr,0]-1, labels[test_arr,1]-1)),shape=(h, l)).toarray()#简便表达大规模稀疏矩阵，构建邻接矩阵
    logits_test = logits_test.reshape([-1,1]) #指定列数为1，行数不定，将上述矩阵转换为一列数组

    logits_train = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(h, l)).toarray()#训练集N
    logits_train = logits_train.reshape([-1,1])

    train_mask = np.array(logits_train[:,0], dtype=np.bool).reshape([-1,1])#有关系的为1
    test_mask = np.array(logits_test[:,0], dtype=np.bool).reshape([-1,1])

    # drug->RNA drug:features
    M = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(h, l)).toarray()#训练集的邻接矩阵

    # lnc adj
    LS=np.loadtxt("data/lncRNAsimilarity.csv", delimiter=',')
    # RNA adj
    MS=np.loadtxt("data/miRNAsimilarity.csv", delimiter=',')
    adj = np.vstack((np.hstack((LS,M)),np.hstack((M.transpose(),MS))))#将矩阵上下左右放置

    F1 = np.loadtxt("E:\py\wwy\RWR\lncRNAfeaturei.txt")
    F2 = np.loadtxt("E:\py\wwy\RWR\miRNAfeaturei.txt")

    size_u = F1.shape #矩阵F1,F2的行列数
    size_v = F2.shape
    adj = preprocess_adj(adj)
    return adj, features, size_u, size_v, logits_train,  logits_test, train_mask, test_mask, labels

def generate_mask(labels,N):   #划分五次数据集，N为训练集正样本数量
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(h,l)).toarray()
    mask = np.zeros(A.shape)
    label_neg=np.zeros((N,2))  #5*N个作用对，负样本
    while(num<N):
        a = random.randint(0,h-1)
        b = random.randint(0,l-1)
        if A[a,b] != 1 and mask[a,b] != 1:#负样本作用对
            mask[a, b] = 1
            label_neg[num, 0] = a
            label_neg[num, 1] = b
            num += 1
    mask = np.reshape(mask,[-1,1])  #mask为一维列表
    #return mask
    return mask,label_neg#随机生成与正样本数量等同的负样本,5*N个负样本

def test_negative_sample(labels,N,negative_mask):  
    num = 0
    A = sp.csr_matrix((labels[:,2],(labels[:,0]-1, labels[:,1]-1)),shape=(h,l)).toarray()
    negative_mask = np.reshape(negative_mask, [h, l])
    mask = np.zeros(A.shape)
    test_neg=np.zeros((5*N,2))
    while(num<5*N):
        a = random.randint(0,h-1)
        b = random.randint(0,l-1)
        if A[a,b] != 1 and mask[a,b] != 1 and negative_mask[a,b] != 1:#除去训练集中的负样本
            mask[a,b] = 1
            test_neg[num,0]=a
            test_neg[num,1]=b
            num += 1
    mask = np.reshape(mask,[-1,1])
    return mask,test_neg

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized

def construct_feed_dict(adj, features, labels, labels_mask, negative_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['adjacency_matrix']: adj})
    feed_dict.update({placeholders['Feature_matrix']: features})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['negative_mask']: negative_mask})
    return feed_dict

def div_list(ls,n):
    ls_len=len(ls)  
    j = ls_len//n
    ls_return = []  
    for i in range(0,(n-1)*j,j):  
        ls_return.append(ls[i:i+j])  
    ls_return.append(ls[(n-1)*j:])  
    return ls_return

def adj_to_bias(sizes, nhood):#sizes什么情况不清楚 def adj_to_bias(sizes=[3973], nhood=1)
      
    #labels = np.loadtxt("adj.txt")
    labels = np.loadtxt("data/adj.csv", delimiter=',')
    reorder = np.arange(labels.shape[0])
    train_arr=reorder.tolist()
    M = sp.csr_matrix((labels[train_arr,2],(labels[train_arr,0]-1, labels[train_arr,1]-1)),shape=(h,l)).toarray()
    adj = np.vstack((np.hstack((np.zeros(shape=(h,h),dtype=int),M)),np.hstack((M.transpose(),np.zeros(shape=(l,l),dtype=int)))))
    adj=adj+np.eye(adj.shape[0])
    adj=np.reshape(adj,(1,adj.shape[0],adj.shape[1]))
    nb_graphs = adj.shape[0] 
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)  
    #return adj

