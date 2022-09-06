from __future__ import division
from __future__ import print_function
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import average_precision_score, accuracy_score
import matplotlib.pyplot as plt
from metrics import *
import time

import numpy as np
import tensorflow as tf
from utils import *
from models import GAutoencoder
from metrics import *
h = 1663
l = 258
class Training():
  def __init__(self):
     self.model = GAutoencoder
     self.learning_rate = 0.01
     self.dropout = 0.5
     self.weight_decay = 5e-6
     self.early_stopping = 100
     self.max_degree = 3
     self.latent_factor_num = 200
     self.epochs = 100
     tf.app.flags.DEFINE_float('weight_decay',5e-6, "description3")
     tf.app.flags.DEFINE_float('learning_rate', 0.01, "description3")

  def train(self,train_arr, test_arr):
        # Settings

    # Load data
    adj, features, size_u, size_v, logits_train, logits_test, train_mask, test_mask, labels = load_data(train_arr, test_arr)
    #size_u=(1663,1663),size_v=(258,258)
    # Some preprocessing

    model_func = GAutoencoder
    # Define placeholders
    placeholders = {
        'adjacency_matrix': tf.placeholder(tf.float32, shape=adj.shape),
        'Feature_matrix': tf.placeholder(tf.float32, shape=features.shape),
        'labels': tf.placeholder(tf.float32, shape=(None, logits_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'negative_mask': tf.placeholder(tf.int32)
    }
    
    # Create model
    model = model_func(placeholders, size_u, size_v, self.latent_factor_num)
    
    # Initialize session
    sess = tf.compat.v1.Session()
    
    
    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer())
    
    # Define model evaluation function
    def evaluate(adj, features, labels, mask, negative_mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(adj, features,labels, mask, negative_mask, placeholders)
        test_predss = sess.run(model.outputs, feed_dict=feed_dict1)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)

        return accuracy_score(labels, test_predss.astype(np.int64)),mcc, PRE, REC,Specificity

    # Train model
    for epoch in range(self.epochs):
        # Construct feed dictionary
        negative_mask, label_neg = generate_mask(labels, len(train_arr))
        #negative_mask为1720*140矩阵展平列表，label_neg为五倍的训练对，12212*5
        feed_dict1 = construct_feed_dict(adj, features, logits_train, train_mask, negative_mask, placeholders)#实验验证当负样本数量为正样本数量五倍时性能结果好
        # Training step
        predss = sess.run(model.outputs, feed_dict=feed_dict1)
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict1)#计算结果

    print("Optimization Finished!")

    # Testing
    testnegative_mask, label_neg= test_negative_sample(labels, len(test_arr),negative_mask)

if __name__ == "__main__":
  # Initial model
  gcn = Training()
  # Set random seed
  seed = 101
  np.random.seed(seed)
  tf.compat.v1.set_random_seed(seed)

