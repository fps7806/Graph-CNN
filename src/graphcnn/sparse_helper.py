import numpy as np
import pdb

class GraphCNNSparseAdjacency(object):
    def __init__(self, indices, values, no_A):
        self.indices = indices
        self.values = values
        self.no_A = no_A

    def from_dense(A, index_type=np.int32):
        indices, values = batched_sparse_from_dense(A)

        return GraphCNNSparseAdjacency(indices, values, A.shape[2])

def batched_sparse_from_dense(mat, batch_dims=1, index_type=np.int32):
    batch_original_shape = mat.shape[:batch_dims]
    mat = np.reshape(mat, [-1] + list(mat.shape[batch_dims:]))

    batch_size = mat.shape[0]
    no_sparse_dims = len(mat.shape)-batch_dims

    batches = [mat[i, ...] for i in range(batch_size)]
    nonzero_indices = [np.stack(np.nonzero(a)).astype(index_type).T for a in batches]
    values = [a[np.nonzero(a)] for a in batches]
    max_indices = np.max([a.shape[0] for a in nonzero_indices])

    indices = np.stack([np.pad(v, ((0, max_indices-v.shape[0]), (0, 0)), mode='constant') for v in nonzero_indices])
    values = np.stack([np.pad(v, ((0, max_indices-v.shape[0])), mode='constant') for v in values])

    indices = np.reshape(indices, list(batch_original_shape) + [max_indices, -1] )
    values = np.reshape(values, list(batch_original_shape) + [max_indices] )

    return indices, values