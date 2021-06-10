import csv

import matplotlib.pyplot as plt
import numpy as np


from sklearn.manifold import TSNE
from sklearn.datasets import load_svmlight_file

def get_labels():
    # with open('./lib/hierarchical3000.csv', 'r', encoding='gbk')as f:
    with open('./lib/BIG15_lbph.csv', 'r', encoding='gbk')as f:
        cs = list(csv.reader(f))
        data=[]
        labels_true = []
        labels_pred  = []
        for i,row in enumerate(cs):
            if i < 0:
                continue
            if i == 3000:
                break
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
    # with open("./lib/tmp.txt") as f:
    # with open("./lib/hierarchical_new_vgg16.csv") as f:
    with open("./lib/GMM_vgg16.csv") as f:
    # with open("./lib/GMM_LBP.csv") as f:
    # with open("./lib/Net_LBP.csv") as f:
    # with open("./lib/Net_Vgg16.csv") as f:
        list1 = f.readlines()
        labels_pred = []
        for i in range(0, len(list1)):
            # if i==0:
            #     continue
            labels_pred.append(float(list1[i].rstrip('\n')))
        return labels_pred


labels_true,labels_pred,x=get_labels()
K_labels_pred = get()


data = load_svmlight_file("./lib/BIG15_vgg16.txt")


# 嵌入空间的维度为2，即将数据降维成2维
# ts = TSNE(n_components=2,init='pca',random_state=1000)
ts = TSNE(n_components=2,init='pca',random_state=501)
# 训练模型
ts.fit_transform(data[0].A)
# ts.fit_transform(x)
# 打印结果
print(ts.embedding_)


for i,color in enumerate(K_labels_pred):
    if color==1:
        s1= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='b', s=3,cmap='rainbow')
    if color==2:
        s2= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='c', s=3,cmap='rainbow')
    if color==3:
        s3= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='g', s=3,cmap='rainbow')
    if color==4:
        s4= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='k', s=3,cmap='rainbow')
    if color==5:
        s5= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='m', s=3,cmap='rainbow')
    if color==6:
        s6= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='r', s=3,cmap='rainbow')
    if color==7:
        s7= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='y', s=3,cmap='rainbow')
    if color == 8:
        s8 = plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='#FF00FF', s=3, cmap='rainbow')
    if color == 0:
        s9 = plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='#CD853F', s=3, cmap='rainbow')



    # if color==1072:
    #     s1= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='b', s=3,cmap='rainbow')
    # if color==1473:
    #     s2= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='c', s=3,cmap='rainbow')
    # if color==2306:
    #     s3= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='g', s=3,cmap='rainbow')
    # if color==10227:
    #     s4= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='k', s=3,cmap='rainbow')
    # if color==10244:
    #     s5= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='m', s=3,cmap='rainbow')
    # if color==2096:
    #     s6= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='r', s=3,cmap='rainbow')
    # if color==3801:
    #     s7= plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='y', s=3,cmap='rainbow')
    # if color == 3595:
    #     s8 = plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='#FF00FF', s=3, cmap='rainbow')
    # if color == 4061:
    #     s9 = plt.scatter(ts.embedding_[i][0], ts.embedding_[i][1], c='#CD853F', s=3, cmap='rainbow')




# plt.scatter(ts.embedding_[:,0],ts.embedding_[:,1],c=K_labels_pred,s=8,cmap='rainbow')
# plt.title("Real_label")
# plt.title("K-mediods")
# plt.title("Hierarchiacl")
# plt.title("CFSCDP")
# plt.title("GMM")
plt.title("VAE")
# plt.title("VAE_CFSFDP")


plt.legend((s1,s2,s3,s4,s5,s6,s7,s8,s9),('1','2','3','4','5','6','7','8','9'),loc = 'best')
# plt.legend((s1,s2,s3,),('1','2','3'),loc = 'best')


plt.savefig('./lib/1_1.png')

plt.show()
