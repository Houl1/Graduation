
import csv
import pandas as pd
import math
import numpy as np


from sklearn.datasets import load_svmlight_file

from sklearn import metrics, cluster
import evaluate

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
                        distances[d_key] = euler_distance(nodes[i].root, nodes[j].root)
                    d = distances[d_key]
                    # if d < min_dist:
                    #     min_dist = d
                    #     closest_part = (i, j)


            # 聚类
            sor = sorted(distances.items(), key=lambda asd: asd[1], reverse=False)
            flag1, flag2 = sor[0][0][0], sor[0][0][1]
            for  i,node in enumerate(nodes):
                if node.flag == flag1:
                    node1 = nodes[i]
                    part1 = i
                if node.flag == flag2:
                    node2 = nodes[i]
                    part2 = i

            new_root = [ (node1.root[i] * node1.count + node2.root[i] * node2.count ) / (node1.count + node2.count)
                        for i in range(future_num)]
            new_node = ClusterNode(root=new_root,
                                   left=node1,
                                   right=node2,
                                   distance=min_dist,
                                   flag=currentflag,
                                   count=node1.count + node2.count)


            for i in range(currentflag, point_num):
                for j in range(currentflag, point_num):
                    if i == flag1 or i == flag1 or j == flag2 or j == flag2:
                        if (i, j) in distances:
                            distances.pop((i, j))
                        if (j, i) in distances:
                            distances.pop((j, i))


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


def main_K():
    with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        data = []
        for i, row in enumerate(cs):
            if i < 0:
                continue
            if i == 3600:
                break
            rows=[]
            for j, s in enumerate(row):
                if j == 10:
                    break
                rows.append(float(s))
            data.append(rows)
        my = Hierarchical_categories(9)
        my.fit(data)
        print(my.labels)
        with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/BIG15_lbph_finally.csv"), 'a', newline='') as f2:
            writer = csv.writer(f2)
            for k,row in enumerate(cs):
                if k < 0:
                    continue
                if k == 3600:
                    break
                row.append(my.labels[k])
                writer.writerow(row)


def get_labels():
    with open('./lib/BIG15_lbph_finally.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        labels_true = []
        labels_pred  = []
        for i,row in enumerate(cs):
            if i < 0:
                continue
            if i == 3600:
                break

            for j, s in enumerate(row):
                if j == 10:
                    labels_true.append(int(s))
                if j == 11:
                    labels_pred.append(int(s))
        return labels_true, labels_pred


def contingency_matrix(labels_true, labels_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred)
    print(contingency_matrix)

if __name__ == '__main__':
    # main_K()

    # labels_true, labels_pred = get_labels()
    # contingency_matrix(labels_true, labels_pred)
    # evaluate.rand_index(labels_true, labels_pred)
    data = load_svmlight_file("./lib/BIG15_basic_lbph.txt")
    print(data[0].A)

    ap=data[0].A[0:2, :]

    print(ap)
