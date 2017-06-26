from __future__ import print_function
from datetime import datetime
import os
import numpy as np

class GraphCNNKeys(object):
    TRAIN_SUMMARIES = "train_summaries"
    TEST_SUMMARIES = "test_summaries"

def print_ext(*args):
    print(str(datetime.now()), *args)
    
def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))
    
def get_node_mask(graph_size, max_size=None):
    if max_size == None:
        max_size = np.max(graph_size)
    return np.array([np.pad(np.ones([s, 1]), ((0, max_size-s), (0, 0)), 'constant', constant_values=(0)) for s in graph_size], dtype=np.float32)
    
def make_print(*args):
    import tensorflow as tf
    def _tf_print(*args):
        for i in range(len(args)):
            print(args[i].shape)
            print(args[i])
        return args
    
    result = tf.py_func(_tf_print, args, [ s.dtype for s in args])
    for i in range(len(args)):
        result[i].set_shape(args[i].get_shape())
    return result