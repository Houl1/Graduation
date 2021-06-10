# -*- coding: utf-8 -*-
# @Time    : 18-12-6
# @Author  : lin
import csv
# import evaluate
from sklearn.datasets import make_blobs
from matplotlib import pyplot
import numpy as np
import random
import math
from sklearn.datasets import load_svmlight_file
from scipy.spatial import distance


class KMediod():
    """
    实现简单的k-medoid算法
    """

    def __init__(self, n_points, k_num_center):
        self.n_points = n_points
        self.k_num_center = k_num_center
        self.data = None
        self.target=None

    def get_test_data(self):
        """
        产生测试数据, n_samples表示多少个点, n_features表示几维, centers
        得到的data是n个点各自坐标
        target是每个坐标的分类比如说我规定好四个分类，target长度为n范围为0-3，主要是画图颜色区别
        :return: none
        """
        # self.data, target = make_blobs(n_samples=self.n_points, n_features=9, centers=self.n_points)
        # with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        #     cs = list(csv.reader(f))
        #     d = []
        #     true = []
        #     for i, row in enumerate(cs):
        #         if i < 0:
        #             continue
        #         if i == 1000:
        #             break
        #         rows = []
        #         for j, s in enumerate(row):
        #             if j == 10:
        #                 true.append(float(s))
        #                 break
        #             rows.append(float(s))
        #         d.append(np.array(rows))
        # self.data=np.array(d)
        # self.target =np.array(true)
        data = load_svmlight_file("./lib/BIG15_vgg16.txt")
        print(data[0].A)
        self.data = data[0].A[0:100, :]
        self.target = data[1][0:100]
        # self.data = data[0].A
        # self.target = data[1]

        # d = load_svmlight_file("./lib/BIG15_basic_lbph.txt")


        # self.data=d[0].A[0:3000,:]
        # self.target = d[1][0:3000]
        # np.put(self.data, [self.n_points, 0], 500, mode='clip')
        # np.put(self.data, [self.n_points, 1], 500, mode='clip')


        # print(self.data[0:2, 0])
        # print(self.data[0:2, 1])
        # pyplot.scatter(self.data[:,0], self.data[:,1], c=self.target)
        # # 画图
        # pyplot.show()

    def ou_distance(self,point1: np.ndarray, point2: np.ndarray):
        # 定义欧式距离的计算
        return distance.euclidean(point1,point2)

    def run_k_center(self, func_of_dis):
        """
        选定好距离公式开始进行训练
        :param func_of_dis:
        :return:
        """
        print('初始化', self.k_num_center, '个中心点')
        indexs = list(range(len(self.data)))
        random.shuffle(indexs)  # 随机选择质心
        init_centroids_index = indexs[:self.k_num_center]
        centroids = self.data[init_centroids_index, :]  # 初始中心点
        # 确定种类编号
        levels = list(range(self.k_num_center))
        print('开始迭代')
        sample_target = []
        if_stop = False

        number  =0
        while (not if_stop):
            if_stop = True
            classify_points = [[centroid] for centroid in centroids]
            sample_target = []
            # 遍历数据
            for sample in self.data:
                # 计算距离，由距离该数据最近的核心，确定该点所属类别
                distances = [func_of_dis(sample, centroid) for centroid in centroids]
                cur_level = np.argmin(distances)
                sample_target.append(cur_level)
                # 统计，方便迭代完成后重新计算中间点
                classify_points[cur_level].append(sample)
            # 重新划分质心
            for i in range(self.k_num_center):  # 几类中分别寻找一个最优点
                distances = [func_of_dis(point_1, centroids[i]) for point_1 in classify_points[i]]
                now_distances = sum(distances)  # 首先计算出现在中心点和其他所有点的距离总和
                for point in classify_points[i]:
                    distances = [func_of_dis(point_1, point) for point_1 in classify_points[i]]
                    new_distance = sum(distances)
                    # 计算出该聚簇中各个点与其他所有点的总和，若是有小于当前中心点的距离总和的，中心点去掉
                    if new_distance < now_distances:
                        now_distances = new_distance
                        centroids[i] = point  # 换成该点
                        if_stop = False
            number +=1
            print('迭代次数：{0}'.format(number))
        print('结束')
        return sample_target

    def run(self):
        """
        先获得数据，由传入参数得到杂乱的n个点，然后由这n个点，分为m个类
        :return:
        """
        self.get_test_data()
        predict = self.run_k_center(self.ou_distance)
        print(predict)
        # with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        #     cs = list(csv.reader(f))
        #     with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/2.csv"), 'a', newline='') as f2:
        #         writer = csv.writer(f2)
        #         rows = -1
        #         for row in cs:
        #             rows += 1
        #             if rows < 0:
        #                 continue
        #             if rows == 10857:
        #                 break
        #             row.append(predict[rows])
        #             writer.writerow(row)
        # with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/Kmediods_vgg16.csv"), 'w', newline='') as f2:
        with open("./lib/Kmediods_vgg16.csv", 'w', newline='') as f2:
            for i, s in enumerate(predict):
                f2.write(str(s))
                f2.write('\n')
            f2.close()
        #evaluate.contingency_matrix(self.target, predict)
        #evaluate.rand_index(self.target, predict)
        # pyplot.scatter(self.data[:, 0], self.data[:, 1], c=predict)
        # pyplot.show()


test_one = KMediod(n_points=100, k_num_center=9)
test_one.run()

def get_labels():
    with open('./lib/2.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        labels_true = []
        labels_pred  = []
        for i,row in enumerate(cs):
            if i < 0:
                continue
            if i == 10857:
                break

            for j, s in enumerate(row):
                if j == 10:
                    labels_true.append(int(s))
                if j == 11:
                    labels_pred.append(int(s))
        return labels_true, labels_pred

# labels_true, labels_pred = get_labels()
# evaluate.contingency_matrix(labels_true, labels_pred)
# evaluate.rand_index(labels_true, labels_pred)

# data,target=make_blobs(n_samples=10000,n_features=10,centers=9)
#
# print(target)
# # 在2D图中绘制样本，每个样本颜色不同
# pyplot.scatter(data[:,0],data[:,1],c=target);
# pyplot.show()