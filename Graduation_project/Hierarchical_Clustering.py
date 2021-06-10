import csv
import math
import numpy as np
from scipy.spatial import distance

from datasketch import MinHash, MinHashLSH
from nltk import ngrams

from sklearn import metrics
import evaluate

from sklearn.datasets import load_svmlight_file

#欧拉距离 point1支持多维
def euler_distance( point1: np.ndarray, point2: list):
    distance = 0.0
    for x, y in zip(point1, point2):
        distance += math.pow(x-y, 2)
    return math.sqrt(distance)

#聚类节点类
class ClusterNode(object):
    def __init__(self, root, left=None, right=None, distance=-1.0, flag=None, count=1):
        self.root = root  #聚类节点
        self.left = left  #左节点
        self.right = right  #右节点
        self.distance = distance  #左右节点距离
        self.flag = flag  #标记节点是否参与过计算(计算过时，此值为初始位置)
        self.count = count  #叶子节点个数


#层次聚类类（以分类数量作为循环跳出条件）
class Hierarchical_categories(object):
    def __init__(self, k = 1):
        assert k > 0
        self.k = k
        self.labels = None
    def fit(self, data):
        nodes = [ClusterNode(root=r, flag=i) for i, r in enumerate(data)]
        distances = {}
        point_num, future_num = np.shape(data)  # 两个维度数
        self.labels = [ -1 ] * point_num
        currentflag = -1
        while len(nodes) > self.k:
            min_dist = math.inf  # 先取最短距离为无穷大
            nodes_len = len(nodes)
            closest_part = None  # 欧拉距离最近的两个节点类的坐标元组
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    # 为了不重复计算距离，保存在字典内
                    d_key = (nodes[i].flag, nodes[j].flag)
                    if d_key not in distances:
                        # distances[d_key] = euler_distance(nodes[i].root, nodes[j].root)
                        distances[d_key] = distance.euclidean(nodes[i].root, nodes[j].root)
                    d = distances[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i, j)
            # 聚类
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_root = [ (node1.root[i] * node1.count + node2.root[i] * node2.count ) / (node1.count + node2.count)
                        for i in range(future_num)]
            new_node = ClusterNode(root=new_root,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   flag=currentflag,
                                   count=node1.count + node2.count)

            currentflag -= 1

            del nodes[part2], nodes[part1]
            nodes.append(new_node)
            print(len(nodes))
        self.nodes = nodes
        self.calc_label()

    def calc_label(self):
        for i, node in enumerate(self.nodes, 1):
            # 将节点的所有叶子节点都分类
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: ClusterNode, label):
        if node.left == None and node.right == None:
            self.labels[node.flag] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)


#层次聚类类（以阈值作为循环跳出条件）
class Hierarchical_threshold(object):
    def __init__(self, threshold = 1):
        assert threshold > 0
        self.threshold = threshold
        self.labels = None
    def fit(self, data):
        nodes = [ClusterNode(root=r, flag=i) for i, r in enumerate(data)]
        distances = {}
        point_num, future_num = np.shape(data)  # 两个维度数
        self.labels = [ -1 ] * point_num
        currentflag = -1
        while 1:
            min_dist = math.inf  # 先取最短距离为无穷大
            nodes_len = len(nodes)
            closest_part = None  # 欧拉距离最近的两个节点类的坐标元组
            for i in range(nodes_len - 1):
                for j in range(i + 1, nodes_len):
                    # 为了不重复计算距离，保存在字典内
                    d_key = (nodes[i].flag, nodes[j].flag)
                    if d_key not in distances:
                        distances[d_key] = euler_distance(nodes[i].root, nodes[j].root)
                    d = distances[d_key]
                    if d < min_dist:
                        min_dist = d
                        closest_part = (i, j)
            if  min_dist > self.threshold: break #最短距离阈值推出推出循环

            # 聚类
            part1, part2 = closest_part
            node1, node2 = nodes[part1], nodes[part2]
            new_root = [ (node1.root[i] * node1.count + node2.root[i] * node2.count ) / (node1.count + node2.count)
                        for i in range(future_num)]
            new_node = ClusterNode(root=new_root,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   flag=currentflag,
                                   count=node1.count + node2.count)

            currentflag -= 1
            del nodes[part2], nodes[part1]
            nodes.append(new_node)
            print(len(nodes))
        self.nodes = nodes
        self.calc_label()

    def calc_label(self):
        for i, node in enumerate(self.nodes,1):
            # 将节点的所有叶子节点都分类
            self.leaf_traversal(node, i)

    def leaf_traversal(self, node: ClusterNode, label):
        if node.left == None and node.right == None:
            self.labels[node.flag] = label
        if node.left:
            self.leaf_traversal(node.left, label)
        if node.right:
            self.leaf_traversal(node.right, label)



#BIG15_lbph1.csv
def main_threshold():
    with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        data = []
        row_num = -1
        for rows in cs:
            row_num += 1
            if row_num < 9000:
                continue
            if row_num == 10857:
                break
            column_num = -1
            row=[]
            for s in rows:
                column_num+=1
                if column_num == 10:
                    break
                row.append(float(s))
            data.append(row)
        # print(data)
        my = Hierarchical_threshold(0.22)
        my.fit(data)
        print(my.labels)
        with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/BIG15_lbph2.csv"), 'a', newline='') as f2:
            writer = csv.writer(f2)
            rows = -1
            for row in cs:
                rows += 1
                if rows < 9000:
                    continue
                if rows == 10857:
                    break
                row.append(my.labels[rows-9000])
                writer.writerow(row)
        # print(np.array(my.labels))


def main_K():
    data = load_svmlight_file("./lib/BIG15_vgg16.txt")
    print(data[0].A)
    # ap = data[0].A[0:1000, :]
    ap = data[0].A
    with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
    #     data = []
    #     for i, row in enumerate(cs):
    #         if i < 0:
    #             continue
    #         if i == 3000:
    #             break
    #         rows=[]
    #         for j, s in enumerate(row):
    #             if j == 10:
    #                 break
    #             rows.append(float(s))
    #         data.append(rows)
        my = Hierarchical_categories(9)
        my.fit(ap)
        # my.fit(data)
        print(my.labels)
        with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/hierarchical_vgg16.csv"), 'w', newline='') as f2:
            for i, s in enumerate(my.labels):
                f2.write(str(s))
                f2.write('\n')
        # with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/hierarchical3000_vgg16.csv"), 'a', newline='') as f2:
        #     writer = csv.writer(f2)
        #     rows = -1
        #     for row in cs:
        #         rows += 1
        #         if rows < 0:
        #             continue
        #         if rows == 3000:
        #             break
        #         row.append(my.labels[rows])
        #         writer.writerow(row)






def get_labels():
    with open('./lib/BIG15_lbph2.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        labels_true = []
        labels_pred  = []
        for i,row in enumerate(cs):
            if i < 0:
                continue
            if i == 3000:
                break

            for j, s in enumerate(row):
                if j == 10:
                    labels_true.append(int(s))
                if j == 11:
                    labels_pred.append(int(s))
        return labels_true, labels_pred



def get_features():
    with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        features = []
        for i, row in enumerate(cs):
            if i < 0:
                continue
            if i == 1000:
                break
            rows=[]
            for j, s in enumerate(row):
                if j == 10:
                    break
                rows.append(float(s))
            features.append(rows)
        return  features



if __name__ == '__main__':



    # x()
    # main_threshold()
    main_K()
    # labels_true, labels_pred = get_labels()
    # print(labels_true)
    # print(labels_pred)
    #
    # evaluate.rand_index(labels_true, labels_pred)
    # evaluate.mutual_information_based_scores(labels_true, labels_pred)
    # evaluate.homogeneity_completeness_V(labels_true, labels_pred)
    # evaluate.fowlkes_mallows_scores(labels_true, labels_pred)
    #
    #
    #
    # evaluate.contingency_matrix(labels_true, labels_pred)
    # evaluate.confusion_matrix(labels_true, labels_pred)
    # print(__name__)


    # with open("./lib/creat1.csv") as f:
    #     list1 = f.readlines()
    #     list2 = []
    #     for i in range(0, len(list1)):
    #         list2.append(int(list1[i].rstrip('\n')))
    #     print(list2)
    #     # data =   load_svmlight_file("./lib/BIG15_basic_lbph.txt")
    #     # labels_true= data[1][0:3000]
    #     with open('./lib/creatdata.csv', 'r', encoding='gbk')as f:
    #         cs = list(csv.reader(f))
    #         labels_true = []
    #         for i, row in enumerate(cs):
    #             if i < 0:
    #                 continue
    #             if i == 10857:
    #                 break
    #
    #             for j, s in enumerate(row):
    #                 if j == 10:
    #                     labels_true.append(int(s))
    #
    #     evaluate.rand_index(labels_true, list2)
    #     evaluate.mutual_infonmation_based_scores(labels_true, list2)
    #     evaluate.homogeneity_completeness_V(labels_true, list2)
    #     evaluate.fowlkes_mallows_scores(labels_true, list2)
    #     evaluate.contingency_matrix(labels_true, list2)

    # features = get_features()
    # with open("./lib/fo.txt") as f:
    #     list1 = f.readlines()
    #     labels_pred = []
    #     for i in range(0, len(list1)):
    #         if i == 0:
    #             continue
    #         labels_pred.append(float(list1[i].rstrip('\n')))
    #     print(labels_pred)
    # evaluate.silhouette_score(features, labels_pred)
    # evaluate.calinski_harabasz(features, labels_pred)
    # evaluate.davies_bouldin(features, labels_pred)


