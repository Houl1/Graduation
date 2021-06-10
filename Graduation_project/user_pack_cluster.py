import csv

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.cluster import AgglomerativeClustering
from pyclust import KMedoids
from sklearn.mixture import GaussianMixture

data = load_svmlight_file("./lib/BIG15_vgg16.txt")
x = data[0].A
# cls = AgglomerativeClustering(n_clusters=9,affinity='euclidean',memory=None,connectivity=None,compute_full_tree='auto',linkage='single').fit(x)

# k = KMedoids(n_clusters=9,distance='euclidean').fit_predict(x)



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

gmmModel = GaussianMixture(n_components=9, covariance_type='diag', random_state=0)
features = get_features()
gmmModel.fit(features)
labels = gmmModel.predict(features)
print(labels)


with open(np.os.path.join(np.os.path.dirname(__file__), "./lib/GMM_LBP.csv"), 'w', newline='') as f2:
    # for i, s in enumerate(cls.labels_):
    for i, s in enumerate(labels):
        f2.write(str(s))
        f2.write('\n')