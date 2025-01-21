# -*- coding: utf8 -*


from collections import defaultdict
from sklearn.cluster import SpectralCoclustering
import numpy as np


def get_clusters(predict_res,row_real_ids,col_real_ids,n_clusters):
    label_rowColId = defaultdict(dict)
    for i in range(n_clusters):
        row_ids = row_real_ids[np.where(predict_res.row_labels_ == i)]
        col_ids = col_real_ids[np.where(predict_res.column_labels_ == i)]
        label_rowColId[i]["row_ids"] = row_ids
        label_rowColId[i]["col_ids"] = col_ids
    return label_rowColId


def spectral_coclustering(score_matrix,row_real_ids,col_real_ids,n_clusters=2):
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
    predict_res = model.fit(score_matrix)
    label_rowColName = get_clusters(predict_res,row_real_ids,col_real_ids,n_clusters)
    return label_rowColName

