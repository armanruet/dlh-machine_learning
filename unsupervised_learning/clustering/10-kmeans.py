#!/usr/bin/env python3
"""kmeans in Dataset"""
import sklearn.cluster


def kmeans(X, k):
    """def the func"""
    model = sklearn.cluster.KMeans(n_clusters=k, random_state=0, n_init="auto")
    clss = model.fit_predict(X)
    C = model.cluster_centers_
    return C, clss
