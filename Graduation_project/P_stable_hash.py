import random
import numpy as np
import csv


class TableNode(object):
    def __init__(self, index):
        self.val = index
        self.buckets = {}

    def getbuckets(self):
        return self.buckets


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
    fpRand = [random.randint(-10, 10) for i in range(k)]

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


def nn_search(dataSet, query, k, L, r, tableSize):
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

    temp = e2LSH(dataSet, k, L, r, tableSize)
    C = pow(2, 32) - 5

    hashTable = temp[0]
    hashFuncGroups = temp[1]
    fpRand = temp[2]

    for i in range(len(temp[0])):
        if 1220 in  temp[0][i].getbuckets():
            print(temp[0][1].getbuckets()[1220])

    print(temp[0][0].getbuckets())
    print(len(temp[0][0].getbuckets()))
    print(len(temp[1]))
    print(temp[2])
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

    dataSet = []
    with open(fileName, "r") as csvFile:
        reader = csv.reader(csvFile)
        for line in reader:
            dataSet.append([float(item) for item in line])

    return dataSet


def euclideanDistance(v1, v2):
    """get euclidean distance of 2 vectors"""

    v1, v2 = np.array(v1), np.array(v2)
    return np.sqrt(np.sum(np.square(v1 - v2)))





if __name__ == "__main__":

    C = pow(2, 32) - 5
    dataSet = readData("./lib/test_data.csv")
    query = [-2.7769, -5.6967, 5.9179, 0.37671, 1]

    indexes = nn_search(dataSet, query, k=20, L=5, r=1, tableSize=20)
    print(len(indexes))
    print(indexes)
    for index in indexes:
        print(euclideanDistance(dataSet[index], query))
