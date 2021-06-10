
import numpy as np

from sklearn import metrics

def rand_index(labels_true,labels_pred):
    rand_score = metrics.rand_score(labels_true, labels_pred)
    adjusted_rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    print(rand_score)
    print(adjusted_rand_score)

def mutual_information_based_scores(labels_true,labels_pred):
    normalized_mutual_info_score = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    print(normalized_mutual_info_score)
    print(adjusted_mutual_info_score)

def homogeneity_completeness_V(labels_true, labels_pred):
    homogeneity_completeness_v_measure = metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
    print(homogeneity_completeness_v_measure)

def fowlkes_mallows_scores(labels_true, labels_pred):
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    print(fowlkes_mallows_score)

def silhouette_score(features, labels_pred):
    silhouette_score = metrics.silhouette_score(np.array(features), np.array(labels_pred), metric='euclidean')
    print(silhouette_score)


def calinski_harabasz(features, labels_pred):
    calinski_harabasz_score = metrics.calinski_harabasz_score(np.array(features), np.array(labels_pred))
    print(calinski_harabasz_score)

def davies_bouldin(features, labels_pred):
    davies_bouldin_score = metrics.davies_bouldin_score(np.array(features), np.array(labels_pred))
    print(davies_bouldin_score)

def contingency_matrix(labels_true, labels_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(labels_true, labels_pred)
    print(contingency_matrix)

def confusion_matrix(labels_true,labels_pred):
    confusion_matrix = metrics.confusion_matrix(labels_true, labels_pred)
    print(confusion_matrix)

