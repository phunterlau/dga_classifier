import numpy as np
import pandas as pd
from sklearn.svm import SVC

raw = pd.read_csv('vectorized_feature_w_ranks_norm.txt')

X=raw.ix[:,'bi_rank':'vowel_ratio'].as_matrix()
Y=raw.ix[:,'class'].as_matrix()

domains=raw.ix[:,'ip'].as_matrix()

from sklearn import linear_model, decomposition, datasets
n_samples, n_features = X.shape
p = range(n_samples)  # Shuffle samples

import random
#random initialization
random.seed(12345)

classifier = SVC(kernel='linear', probability=True, random_state=0)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

precision_list = []
recall_list =[]
area_list = []
fpr_list = []
tpr_list = []
roc_auc_list = []
accuracy_list = []

from sklearn.metrics import roc_curve, auc
for i in range(10):#10 fold cross-validation
    print 'x-validation round %d'%i
    random.seed(i)
    random.shuffle(p)
    XX,yy = X[p],Y[p]
    cut_off = int(n_samples / 5)*4
    probas_ = classifier.fit(XX[:cut_off], yy[:cut_off]).predict(XX[cut_off:])
    precision, recall, thresholds = precision_recall_curve(yy[cut_off:], probas_)
    fpr, tpr, thresholds = roc_curve(yy[cut_off:], probas_)
    roc_auc = auc(fpr,tpr)
    area = auc(recall, precision)   
    precision_list.append(precision)
    recall_list.append(recall)
    area_list.append(area)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)
    pred = [int(i>0.5) for i in probas_]
    accuracy_list.append(accuracy_score(yy[cut_off:],pred,normalize=True))
