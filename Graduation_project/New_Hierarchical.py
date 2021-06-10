import random



import csv
import math
import numpy as np


from datasketch import MinHash, MinHashLSH
from nltk import ngrams

from sklearn import metrics
import evaluate

from sklearn.datasets import load_svmlight_file


class TableNode(object):
    def __init__(self, index):
        self.val = index
        self.buckets = {}



def genPara(n, r):
    """

    :param n: length of data vector
    :param r:
    :return: a, b
    """

    a = []
    for i in range(n):
        a.append(random.gauss(0, 1))
    b = random.uniform(0, r)

    return a, b


def gen_e2LSH_family(n, k, r):
    """

    :param n: length of data vector
    :param k:
    :param r:
    :return: a list of parameters (a, b)
    """
    result = []
    for i in range(k):
        result.append(genPara(n, r))

    return result


def gen_HashVals(e2LSH_family, v, r):
    """

    :param e2LSH_family: include k hash funcs(parameters)
    :param v: data vector
    :param r:
    :return hash values: a list
    """

    # hashVals include k values
    hashVals = []

    for hab in e2LSH_family:
        hashVal = (np.inner(hab[0], v) + hab[1]) // r
        hashVals.append(hashVal)

    return hashVals


def H2(hashVals, fpRand, k, C):
    """

    :param hashVals: k hash vals
    :param fpRand: ri', the random vals that used to generate fingerprint
    :param k, C: parameter
    :return: the fingerprint of (x1, x2, ..., xk), a int value
    """
    return int(sum([(hashVals[i] * fpRand[i]) for i in range(k)]) % C)


def e2LSH(dataSet, k, L, r, tableSize):
    """
    generate hash table

    * hash table: a list, [node1, node2, ... node_{tableSize - 1}]
    ** node: node.val = index; node.buckets = {}
    *** node.buckets: a dictionary, {fp:[v1, ..], ...}

    :param dataSet: a set of vector(list)
    :param k:
    :param L:
    :param r:
    :param tableSize:
    :return: 3 elements, hash table, hash functions, fpRand
    """

    hashTable = [TableNode(i) for i in range(tableSize)]

    n = len(dataSet[0])
    m = len(dataSet)

    C = pow(2, 32) - 5
    hashFuncs = []
    fpRand = [random.randint(0, 10) for i in range(k)]

    for times in range(L):

        e2LSH_family = gen_e2LSH_family(n, k, r)

        # hashFuncs: [[h1, ...hk], [h1, ..hk], ..., [h1, ...hk]]
        # hashFuncs include L hash functions group, and each group contain k hash functions
        hashFuncs.append(e2LSH_family)

        for dataIndex in range(m):

            # generate k hash values
            hashVals = gen_HashVals(e2LSH_family, dataSet[dataIndex], r)

            # generate fingerprint
            fp = H2(hashVals, fpRand, k, C)

            # generate index
            index = fp % tableSize

            # find the node of hash table
            node = hashTable[index]

            # node.buckets is a dictionary: {fp: vector_list}
            if fp in node.buckets:

                # bucket is vector list
                bucket = node.buckets[fp]

                # add the data index into bucket
                bucket.append(dataIndex)

            else:
                node.buckets[fp] = [dataIndex]

    return hashTable, hashFuncs, fpRand


def nn_search(temp, query, k, L, r, tableSize):
    """

    :param dataSet:
    :param query:
    :param k:
    :param L:
    :param r:
    :param tableSize:
    :return: the data index that similar with query
    """

    result = set()

    # temp = e2LSH(dataSet, k, L, r, tableSize)
    C = pow(2, 32) - 5

    hashTable = temp[0]
    hashFuncGroups = temp[1]
    fpRand = temp[2]


    for hashFuncGroup in hashFuncGroups:

        # get the fingerprint of query
        queryFp = H2(gen_HashVals(hashFuncGroup, query, r), fpRand, k, C)

        # get the index of query in hash table
        queryIndex = queryFp % tableSize

        # get the bucket in the dictionary
        if queryFp in hashTable[queryIndex].buckets:
            result.update(hashTable[queryIndex].buckets[queryFp])

    return result





def readData(fileName):
    """read csv data"""

    with open(fileName, 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        data = []
        row_num = -1
        for rows in cs:
            row_num += 1
            if row_num < 0:
                continue
            if row_num == 10857:
                break
            column_num = -1
            row = []
            for s in rows:
                column_num += 1
                if column_num == 10:
                    break
                row.append(float(s))
            data.append(row)

    return data


def euclideanDistance(v1, v2):
    """get euclidean distance of 2 vectors"""

    v1, v2 = np.array(v1), np.array(v2)
    return np.sqrt(np.sum(np.square(v1 - v2)))



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
                        distances[d_key] = euclideanDistance(nodes[i].root, nodes[j].root)
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



def main_K():
    data = load_svmlight_file("./lib/BIG15_basic_lbph.txt")
    print(data[0].A)
    ap = data[0].A[0:3000, :]
    with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        # data = []
        # for i, row in enumerate(cs):
        #     if i < 0:
        #         continue
        #     if i == 1000:
        #         break
        #     rows=[]
        #     for j, s in enumerate(row):
        #         if j == 10:
        #             break
        #         rows.append(float(s))
        #     data.append(rows)
        my = Hierarchical_categories(9)
        my.fit(ap)
        print(my.labels)
        with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/new1.csv"), 'w', newline='') as f2:
            for i, s in enumerate(my.labels):
                f2.write(str(s))
                f2.write('\n')



if __name__ == "__main__":

    C = pow(2, 32) - 5
    dataSet = readData("./lib/BIG15_lbph.csv")
    query3 = [0.5990615487098694,0.20039010047912598,0.014164621941745281,0.0216166153550148,0.010998617857694626,0.009290930815041065,0.016467303037643433,0.04268490523099899,0.07311704009771347,0.7698405981063843]

    query2 = [0.9680184125900269,0.05756811425089836,0.01726689748466015,0.012397794984281063,0.007010922767221928,0.00878072902560234,0.016891835257411003,0.05709364637732506,0.054649338126182556,0.2291649580001831]
    query1 =[0.4729744493961334,0.21972738206386566,0.07442004233598709,0.08397340774536133,0.03766612708568573,0.043039895594120026,0.057237256318330765,0.16260622441768646,0.13512371480464935,0.8149716258049011]
    temp = e2LSH(dataSet, k=20, L=5, r=1, tableSize=9)
    indexes1 = nn_search(temp, query1, k=10, L=12, r=1, tableSize=9)
    indexes2 = nn_search(temp, query2, k=10, L=12, r=1, tableSize=9)
    indexes3=   nn_search(temp, query3, k=10, L=12, r=1, tableSize=9)
    print(len(indexes1))
    print(len(indexes2))
    print(len(indexes3))
    # for index in indexes:
    #     print(euclideanDistance(dataSet[index], query3))
