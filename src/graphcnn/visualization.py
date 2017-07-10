from graphcnn.helper import *
import networkx as nx
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from math import *



def print_graph(V, A, mask=None):
    if len(V.shape) > 2:
        for i in range(V.shape[0]):
            print_ext('---------------------- Sample %d ----------------------' % (i+1))
            if mask == None:
                print_graph(V[i, ...], A[i, ...])
            else:
                print_graph(V[i, ...], A[i, ...], mask[i, ...])
    else:
        for i in range(V.shape[0]):
            if mask == None or mask[i] > 0:
                print_ext('Vertex %d: '% (i+1), V[i])
                edge_i = np.nonzero(A[i])[1]+1
                print_ext('Edges:', edge_i.astype(np.float32))
                print_ext('Value:', A[i, 0, edge_i-1])
                print_ext('')
                
def make_networkx_graph(A, V=None, weighted=True):
    G = nx.Graph()
    
    if len(A.shape) > 2:
        edges = coo_matrix(np.sum(A, axis=1))
    else:
        edges = coo_matrix(A)

    if V is not None:
        G.add_nodes_from(range(len(V)))

    if weighted:
        edges = np.stack([edges.row, edges.col, edges.data], axis=1) 
        G.add_weighted_edges_from(edges)
    else:
        edges = np.stack([edges.row, edges.col], axis=1) 
        G.add_edges_from(edges)
    return G
    
def get_spring_layout(A, V=None):
    G = make_networkx_graph(A, V=V)
    result = np.array(list(nx.spring_layout(G).values()))
    return result
def get_spectral_layout(A, V=None):
    G = make_networkx_graph(A)
    result = np.array(list(nx.spectral_layout(G).values()))
    return result
    
def get_gridded_layout(A):
    size = round(sqrt(A.shape[0]))
    
    result = np.zeros([A.shape[0], 2])
    
    for i in range(A.shape[0]):
        result[i, 0] = int(i % size)
        result[i, 1] = -int(i / size)
    return result

try:
    import pydotplus
    G = make_networkx_graph(np.zeros([20, 20]))
    pos = np.array(list(nx.drawing.nx_pydot.graphviz_layout(G).values()))
except Exception as e:
    print('=' * 30)
    print('pydotplus installation not found, could result in better looking pictures.')
    print(e)
    print('=' * 30)

def display_graph(V, A, pos, ax=None, node_size=100.):
    if len(A.shape) > 2:
        A = np.sum(A, axis=1)
    if pos is None:    
        D = np.sum(A, axis=1)
        disconnected = D == 0
        temp_A = A
        temp_A[disconnected, disconnected] = True
        try:
            # weighted graph doesn't work (No idea why)
            G = make_networkx_graph(temp_A, V=V, weighted=False)
            pos = np.array(list(nx.drawing.nx_pydot.graphviz_layout(G).values()))
        except:
            # Fall back in case pydotplus is not available, 100 is just to have similar pos as nx_pydot
            pos = np.array(list(nx.spectral_layout(G).values()))*100
    G = make_networkx_graph(A, V=V)
    
    min_l = np.min(pos, axis=0)
    max_l = np.max(pos, axis=0)
    
    # Code breaks with add_weighted_edges_from if this is skipped
    pos_dict = dict([i, pos[i, :]] for i in range(pos.shape[0]))
    nx.draw(G, pos=pos_dict, cmap = plt.get_cmap('jet'), ax=ax, node_size=node_size)
    plt.ylim(ymin=min_l[1]-node_size/10-5, ymax=max_l[1]+node_size/10+5)
    plt.xlim(xmin=min_l[0]-node_size/10-5, xmax=max_l[0]+node_size/10+5)

    return pos
    