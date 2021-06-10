# -*- coding: utf-8 -*-
# @Time : 2021/1/12 22:37
# @Author : CyrusMay WJ
# @FileName: SOM.py
# @Software: PyCharm
# @Blog ：https://blog.csdn.net/Cyrus_May
import numpy as np
import random

np.random.seed(22)

class CyrusSOM(object):
    def __init__(self,net=[[1,1],[1,1]],epochs = 50,r_t = [None,None],eps=1e-6):
        """
        :param net: 竞争层的拓扑结构，支持一维及二维，1表示该输出节点存在，0表示不存在该输出节点
        :param epochs: 最大迭代次数
        :param r_t:   [C,B]    领域半径参数，r = C*e**(-B*t/eoochs),其中t表示当前迭代次数
        :param eps: learning rate的阈值
        """

        self.epochs = epochs
        self.C = r_t[0]
        self.B = r_t[1]
        self.eps = eps
        self.output_net = np.array(net)
        if len(self.output_net.shape) == 1:
            self.output_net = self.output_net.reshape([-1,1])
        self.coord = np.zeros([self.output_net.shape[0],self.output_net.shape[1],2])
        for i in range(self.output_net.shape[0]):
            for j in range(self.output_net.shape[1]):
                self.coord[i,j] = [i,j]
        print(self.coord)


    def __r_t(self,t):
        if not self.C:
            return 0.5
        else:
            return self.C*np.exp(-self.B*t/self.epochs)

    def __lr(self,t,distance):
        return (self.epochs-t)/self.epochs*np.exp(-distance)
    def standard_x(self,x):
        x = np.array(x)
        for i in range(x.shape[0]):
            x[i,:] = [value/(((x[i,:])**2).sum()**0.5) for value in x[i,:]]
        return x
    def standard_w(self,w):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                w[i,j,:] = [value/(((w[i,j,:])**2).sum()**0.5) for value in w[i,j,:]]
        return w
    def cal_similar(self,x,w):
        similar = (x*w).sum(axis=2)
        coord = np.where(similar==similar.max())
        return [coord[0][0],coord[1][0]]

    def update_w(self,center_coord,x,step):
        for i in range(self.coord.shape[0]):
            for j in range(self.coord.shape[1]):
                distance = (((center_coord-self.coord[i,j])**2).sum())**0.5
                if distance <= self.__r_t(step):
                    self.W[i,j] = self.W[i,j] + self.__lr(step,distance)*(x-self.W[i,j])

    def transform_fit(self,x):
        self.train_x = self.standard_x(x)
        self.W = np.zeros([self.output_net.shape[0],self.output_net.shape[1],self.train_x.shape[1]])
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i,j,:] = self.train_x[random.choice(range(self.train_x.shape[0])),:]
        self.W = self.standard_w(self.W)
        for step in range(int(self.epochs)):
            j = 0
            if self.__lr(step,0) <= self.eps:
                break
            for index in range(self.train_x.shape[0]):
                print("*"*8,"({},{})/{} W:\n".format(step,j,self.epochs),self.W)
                center_coord = self.cal_similar(self.train_x[index,:],self.W)
                self.update_w(center_coord,self.train_x[index,:],step)
                self.W = self.standard_w(self.W)
                j += 1
        label = []
        for index in range(self.train_x.shape[0]):
            center_coord = self.cal_similar(self.train_x[index, :], self.W)
            label.append(center_coord[1]*self.coord.shape[1] + center_coord[0])
        class_dict = {}
        for index in range(self.train_x.shape[0]):
            if label[index] in class_dict.keys():
                class_dict[label[index]].append(index)
            else:
                class_dict[label[index]] = [index]
        cluster_center = {}
        for key,value in class_dict.items():
            cluster_center[key] = np.array([x[i, :] for i in value]).mean(axis=0)
        self.cluster_center = cluster_center

        return label


    def fit(self,x):
        self.train_x = self.standard_x(x)
        self.W = np.random.rand(self.output_net.shape[0], self.output_net.shape[1], self.train_x.shape[1])
        self.W = self.standard_w(self.W)
        for step in range(int(self.epochs)):
            j = 0
            if self.__lr(step,0) <= self.eps:
                break
            for index in range(self.train_x.shape[0]):
                print("*"*8,"({},{})/{} W:\n".format(step, j, self.epochs), self.W)
                center_coord = self.cal_similar(self.train_x[index, :], self.W)
                self.update_w(center_coord, self.train_x[index, :], step)
                self.W = self.standard_w(self.W)
                j += 1
        label = []
        for index in range(self.train_x.shape[0]):
            center_coord = self.cal_similar(self.train_x[index, :], self.W)
            label.append(center_coord[1] * self.coord.shape[1] + center_coord[1])
        class_dict = {}
        for index in range(self.train_x.shape[0]):
            if label[index] in class_dict.keys():
                class_dict[label[index]].append(index)
            else:
                class_dict[label[index]] = [index]
        cluster_center = {}
        for key, value in class_dict.items():
            cluster_center[key] = np.array([x[i, :] for i in value]).mean(axis=0)
        self.cluster_center = cluster_center

    def predict(self,x):
        self.pre_x = self.standard_x(x)
        label = []
        for index in range(self.pre_x.shape[0]):
            center_coord = self.cal_similar(self.pre_x[index, :], self.W)
            label.append(center_coord[1] * self.coord.shape[1] + center_coord[1])
        return label

from sklearn.datasets import load_svmlight_file
from sklearn.datasets import load_iris,make_blobs
import matplotlib.pyplot as plt
from  sklearn.metrics import classification_report
import csv
def get_features():
    with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        features = []
        for i, row in enumerate(cs):
            if i < 0:
                continue
            if i == 3000:
                break
            rows=[]
            for j, s in enumerate(row):
                if j == 10:
                    break
                rows.append(float(s))
            features.append(rows)
        return  features


if __name__ == '__main__':
    # SOM = CyrusSOM(epochs=9)
    # data = load_svmlight_file("./lib/BIG15_vgg16.txt")
    # x = data[0].A
    # y_pre = SOM.transform_fit(x)
    # print(y_pre)
    # with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/Net_vgg16.csv"), 'w', newline='') as f2:
    #     # for i, s in enumerate(cls.labels_):
    #     for i, s in enumerate(y_pre):
    #         f2.write(str(s))
    #         f2.write('\n')

    # SOM = CyrusSOM(epochs=5)
    # data = make_blobs(n_samples=1000, n_features=2, centers=4, cluster_std=0.3)
    # x = data[0]
    # print(1111111111111)
    # print(data)
    # print(111111111111)
    # y_pre = SOM.transform_fit(x)
    # print(y_pre)

    SOM = CyrusSOM(net=[1,1,1,1,1,1,1,1,1])
    features = get_features()
    # data = load_svmlight_file("./lib/BIG15_vgg16.txt")
    #  x = data[0].A

    y_pre = SOM.transform_fit(np.array(features))
    print(y_pre)
    with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/Net_LBP.csv"), 'w', newline='') as f2:
        # for i, s in enumerate(cls.labels_):
        for i, s in enumerate(y_pre):
            f2.write(str(s))
            f2.write('\n')




