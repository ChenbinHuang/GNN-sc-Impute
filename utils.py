import torch
import numpy as np

def gs_from_gmt_line(line):
    """
    get gene set from a single line of gmt file
    Modified from gsea_api.GeneSet 
    https://github.com/krassowski/gsea-api/
    """
    name, _, *ids = line.strip().split('\t')
    return name, ids

def from_gmt(path):
    """
    get all gene sets from gmt file
    Modified from gsea_api.GeneSets 
    https://github.com/krassowski/gsea-api/
    """
    res = dict()
    with open(path) as f:
        for line in f:
            gs = gs_from_gmt_line(line)
            res[gs[0]] = gs[1]
    return res

def sparse_mtx_2_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d