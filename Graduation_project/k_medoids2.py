import csv

import numpy as np
from numpy import *
import sys
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import evaluate

def fit_plot_Kmean_model(X):
    data = pd.read_csv(X, header=None, error_bad_lines=False)
    # 必须添加header=None，否则默认把第一行数据处理成列名导致缺失
    data_list = data.values.tolist()
    data_list = np.array(data_list)
    return data_list


def choice_center(data,k):
    k=int(k)
    centers = []
    for i in np.random.choice(len(data), k):
        centers.append(data[i])
    # print("随机选取的中心点(第一次):\n", centers)
    return centers


def distance(a, b):
    dis = []
    for i in range(len(a)):
        dis.append(pow(a[i] - b[i], 2))
    return sqrt(sum(dis))


def get_labels():
    with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        labels_true = []
        for i,row in enumerate(cs):
            if i < 0:
                continue
            if i == 10857:
                break

            for j, s in enumerate(row):
                if j == 10:
                    labels_true.append(int(s))
        return labels_true


def k_center(data_list,center,n_clusters,savePath):
    flag = True
    i = 0
    a=len(data_list[0])-1
    n_clusters=int(n_clusters)
    while flag:
        flag = False
        for i in range(len(data_list)):                       # 遍历所有样本，最后一列标记该样本所属簇
            min_index = -2
            min_dis = inf
            for j in range(len(center)):
                dis = distance(data_list[i][1:n_clusters],center[j][1:n_clusters])
                if dis < min_dis:
                    min_dis = dis
                    min_index = j
            if data_list[i][-1] != min_index:
                flag = True
            data_list[i][-1] = min_index
        # print("分类结果111：",data_list)
        # 重新计算簇中心
        for k in range(len(center)):                      # 遍历中心向量，取出属于当前中心向量簇的样本
            current_k = []
            for i in range(len(data_list)):
                if data_list[i][-1] == k:
                    current_k.append(data_list[i])
#            print(k, "：", current_k)
            old_dis = 0.0
            for i in range(len(current_k)):
                old_dis += distance(current_k[i][1:n_clusters], center[k][1:n_clusters])
            for m in range(len(current_k)):
                new_dis = 0.0
                for n in range(len(current_k)):
                    new_dis += distance(current_k[m][1:n_clusters], current_k[n][1:n_clusters])
                if new_dis < old_dis:
                    old_dis = new_dis
                    center[k][:] = current_k[m][:]

    # print("选中的最终中心点", center)
    for i in range(len(data_list)):  # 遍历所有样本，最后一列标记该样本所属簇
        min_index = -2
        min_dis = inf
        for j in range(len(center)):
            dis = distance(data_list[i][1:n_clusters], center[j][1:n_clusters])
            if dis < min_dis:
                min_dis = dis
                min_index = j
        data_list[i][-1] = min_index
    # print("分类结果222：", data_list)

    y_pred=data_list[:,a]
    # print("y_pred：", y_pred)
    labels_true = get_labels()
    evaluate.contingency_matrix(labels_true, y_pred)
    evaluate.rand_index(labels_true, y_pred)
    score=metrics.calinski_harabasz_score(data_list,y_pred)
    plt.figure(figsize=(12, 12))
    plt.xticks(())
    plt.yticks(())
    plt.subplot(221)
    plt.scatter(data_list[:, 0], data_list[:, 1])
    plt.title("Raw data")
    plt.subplot(222)
    plt.scatter(data_list[:, 0], data_list[:, 1], c=y_pred)
    plt.title("Kmediods:k={},score={}".format(n_clusters, int(score)))
    # plt.title("DBSCAN:k={},eps={},min_samples={}".format(n_clusters, eps, min_samples))

    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame(y_pred)

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    savePath = savePath + "\\Kmediods_result.csv"
    dataframe.to_csv(savePath, header=False, index=False, sep=',')
    plt.show()
    a = 'OK'
    return a


if __name__ == '__main__':
    # data_list=fit_plot_Kmean_model(sys.argv[2])
    # # 1为选取的中心点个数，2为文件打开路径,3为文件保存路径
    # centers = choice_center(data_list,sys.argv[1])
    # print(k_center(data_list, centers,sys.argv[1],sys.argv[3]))
    with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        d = []
        true = []
        for i, row in enumerate(cs):
            if i < 0:
                continue
            if i == 10857:
                break
            rows = []
            for j, s in enumerate(row):
                if j == 10:
                    true.append(float(s))
                    break
                rows.append(float(s))
            d.append(np.array(rows))
        data_list = np.array(d)
    centers = choice_center(data_list, 9)
    print(k_center(data_list, centers,9, '.\\lib'))

    # 1为选取的中心点个数，2为文件打开路径,3为文件保存路径
    # print(k_center(fit_plot_Kmean_model(sys.argv[2]), choice_center(fit_plot_Kmean_model(sys.argv[2]),sys.argv[1]),sys.argv[1],sys.argv[3]))


    # print(k_center(fit_plot_Kmean_model('C:\\Users\\upc\\Desktop\\testdata.csv'), choice_center(fit_plot_Kmean_model('C:\\Users\\upc\\Desktop\\testdata.csv'),4),4,'C:\\Users\\upc\\Desktop'))
