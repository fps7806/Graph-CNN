from graphcnn.helper import *
import scipy.io
import numpy as np
import datetime
from .helper import *
import graphcnn


def load_cora_dataset():
    locate_or_download_file('cora.tgz', 'http://www.cs.umd.edu/~sen/lbc-proj/data/cora.tgz')
    locate_or_extract_file('cora.tgz', 'cora')
    
    keys = []
    features = []
    labels = []
    categories = []
    with open(get_file_location('cora/cora.content'), 'r') as file:
        for line in file:
            s = line[:-1].split('\t')
            keys.append(s[0])
            features.extend([int(v) for v in s[1:-2]])
            if s[-1] not in categories:
                categories.append(s[-1])
            labels.append(categories.index(s[-1]))
        labels = np.array(labels)
        features = np.array(features).reshape((len(keys), -1))
    
    with open(get_file_location('cora/cora.cites'), 'r') as file:
        adj_mat = np.zeros((len(labels), 2, len(labels)))
        for line in file:
            s = line[:-1].split('\t')
            a = keys.index(s[0])
            b = keys.index(s[1])
            adj_mat[a, 0, b] = 1;
            adj_mat[b, 1, a] = 1;

    return graphcnn.SingleGraphInputPipeline(vertices = features, adjacency=adj_mat, labels=labels)
