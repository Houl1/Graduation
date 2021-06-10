import csv
import math
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from datasketch import MinHash, MinHashLSH
from nltk import ngrams

from sklearn import metrics
import evaluate

from sklearn.datasets import load_svmlight_file

from sklearn.datasets import load_svmlight_file

def get_labels():
    with open('./lib/hierarchical3000.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        data=[]
        labels_true = []
        labels_pred  = []
        for i,row in enumerate(cs):
            if i < 0:
                continue
            # if i == 3000:
            #     break
            rows = []
            for j, s in enumerate(row):
                if j == 10:
                    labels_true.append(int(s))
                if j == 11:
                    labels_pred.append(int(s))
                if j!=10 and j!=11:
                    rows.append(float(s))
            data.append(np.array(rows))

        return labels_true, labels_pred, np.array(data)

def get():
    # with open("./lib/finally.txt") as f:
    # with open("./lib/Kmediods3000.csv") as f:
    # with open("./lib/Kmediods_vgg16.csv") as f:
    # with open("./lib/hierarchical_average_vgg16.csv") as f:
    # with open("./lib/tmp.txt") as f:
    # with open("./lib/Net_LBP.csv") as f:
    with open("./lib/Net_Vgg16.csv") as f:
        list1 = f.readlines()
        labels_pred = []
        for i in range(0, len(list1)):
            # if i==0:
            #     continue
            labels_pred.append(float(list1[i].rstrip('\n')))
        return labels_pred

if __name__  == '__main__':
    labels_true, labels_pred, data = get_labels()
    # K_labels_pred = get()
    # print(accuracy_score(labels_true, K_labels_pred))  # 0.5
    # print(accuracy_score(labels_true, K_labels_pred, normalize=False))
    #
    # print(recall_score(labels_true, K_labels_pred, average='macro'))
    # print(recall_score(labels_true, K_labels_pred, average='micro'))
    # print(recall_score(labels_true, K_labels_pred, average='weighted'))

    # data = load_svmlight_file("./lib/BIG15_vgg16.txt")
    # labels_true =   data[1]
    K_labels_pred = get()

    evaluate.rand_index(labels_true, K_labels_pred)
    evaluate.mutual_information_based_scores(labels_true, K_labels_pred)
    evaluate.fowlkes_mallows_scores(labels_true, K_labels_pred)

    evaluate.homogeneity_completeness_V(labels_true, K_labels_pred)
    evaluate.silhouette_score(data[0].A, K_labels_pred)