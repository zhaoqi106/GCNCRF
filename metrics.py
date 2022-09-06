import tensorflow as tf
import numpy as np
import sklearn
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix

from sklearn import metrics
h = 1663
l = 258
association_nam=15386

def masked_accuracy(preds, labels, mask, negative_mask): #preds为重构，labels为原始
    """Accuracy with masking."""
    preds = tf.cast(preds, tf.float32) #将张量里的数据类型更换
    labels = tf.cast(labels, tf.float32)
    error = tf.square(preds-labels)  #负样本的标签值为0，正样本的标签值为1
    mask += negative_mask
    mask = tf.cast(mask, dtype=tf.float32) 
    error *= mask  #*代表点乘 计算误差时需要将负样本的累加
#     return tf.reduce_sum(error)
    return tf.sqrt(tf.reduce_mean(error))

def euclidean_loss(preds, labels):
    euclidean_loss = tf.sqrt(tf.reduce_sum(tf.square(preds-labels),0))
    return euclidean_loss

def dot(x, y, sparse=False):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def prediction(hidden):
    num_u = h
    U=hidden[0:num_u,:]
    V=hidden[num_u:,:]
    M1 = dot(U,tf.transpose(V))
        
    U1 = tf.norm(U,axis=1,keep_dims=True, name='normal') #by row
    V1 = tf.norm(V,axis=1,keep_dims=True, name='normal')
    F = dot(U1,tf.transpose(V1))
    Score = tf.divide(M1,F)   #对应
    Score = tf.nn.sigmoid(Score)  #by long   
    Score = tf.reshape(Score,[-1,1])
    return Score

def prediction_np(hidden):
    num_u = h
    U=hidden[0:num_u,:]
    V=hidden[num_u:,:]
    M1 = np.dot(U,np.transpose(V))
        
    U1 = np.linalg.norm(U,ord=2,axis=1,keepdims=True) #by row
    V1 = np.linalg.norm(V,ord=2,axis=1,keepdims=True)
    F = np.dot(U1,np.transpose(V1))
    Score = M1/F   #对应
    #Score = tf.nn.sigmoid(Score)  #by long   
    #Score = tf.reshape(Score,[-1,1])
    return Score

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range
   
def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]  # 对向量进行缩放
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))  # 测试集预测的结果
    negative_index = np.where(predict_score_matrix < thresholds.T)  # 小于阈值的就是负类
    # positive_index = np.where(predict_score_matrix >= thresholds.T) #大于阈值的就是正类
    predict_score_matrix = np.ones((thresholds_num, h * l))
    predict_score_matrix[negative_index] = 0  # 测试集值为0的就是负类
    # predict_score_matrix[positive_index] = 1 #测试集值为1的就是正类
    TP = predict_score_matrix.dot(real_score.T)  # 真正类 正样本预测为正
    FP = predict_score_matrix.sum(axis=1) - TP  # 假正类 负样本预测为正
    FN = real_score.sum() - TP  # 假负类 正样本被预测为负
    tpr = TP / (TP + FN)  # 假负率
    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])
    return aupr


def find_optimal_cutoff(tpr,fpr,threshold):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold

def get_mcc(y_test,y_test_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    cutoff = find_optimal_cutoff(tpr, fpr, thresholds)
    y_pred = list(map(lambda x: 1 if x >= cutoff else 0, y_test_pred))
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    PRE = TP / (FP + TP)
    REC = TP / (TP+FN)
    Specificity = TN / (TN + FP)
    mcc = sklearn.metrics.matthews_corrcoef(y_test, y_pred)
    return mcc,PRE,REC,Specificity