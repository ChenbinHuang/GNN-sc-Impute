import itertools
import pandas as pd
import numpy as np
import scipy.sparse as sp

def net_from_MSigDB(gsets, gene_ids, weights=True):
    """
    create network from MSigDB gensets
    @input gene sets in format of dict (output of utils.from_gmt)
    @input gene_ids in the dataset. (adata.var.index)
    should match the format of gene names /entrez/gene symbol/gene ensemble
    """
    genes = set([g for gs in gsets for g in gsets[gs] ])
    # only edges with valid symbols
    edges = []
    for gs in gsets:
        genes = gsets[gs]
        genes = gene_ids.intersection(genes)
        edges += [subset for subset in itertools.combinations(genes, 2)]

    if weights:
        edges = pd.DataFrame(edges)
        edges["support"]=1
        edges = edges.groupby([0,1])["support"].count().reset_index()
    
    edges = np.asarray(edges) 
    
    # build graph
    idx = np.array(gene_ids)
    idx_map = {j: i for i, j in enumerate(idx)}
    # the key (names) in edges_unordered --> the index (which row) in matrix
    def get_key(x):
        res = idx_map.get(x)
        return res if res else x
    edges = np.array(list(map(get_key, edges.flatten())),
                     dtype=np.int32).reshape(edges.shape) #map：map(function, element):function on element
    if weights:
        adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                        shape=(gene_ids.shape[0], gene_ids.shape[0]),
                        dtype=np.float32)
    else:
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(gene_ids.shape[0], gene_ids.shape[0]),
                        dtype=np.float32)
    del idx,idx_map,edges
    # build symmetric adjacency matrix
    # keeps the sum
    adj = adj + adj.T
    if not weights:
        adj = adj.T > 0
#    adj = adj + sp.eye(adj.shape[0])
    return adj
    

def net_from_BioGRID(bgfile, gene_ids, weights=True):
    """
    create network from BioGRID DB
    Modified from https://github.com/NabaviLab/sigGCN/
    add weights to original 
    @input gene sets in format of dict (output of utils.from_gmt)
    @input gene_ids in the dataset. (adata.var.index)
    should match the format of gene names /entrez/gene symbol/gene ensemble
    """
    edges_unordered =  pd.read_table(bgfile ,index_col=None, usecols = [7,8] )
    if weights:
        edges_unordered["support"] = 1
        edges_unordered = edges_unordered.groupby(["Official Symbol Interactor A", "Official Symbol Interactor B"])["support"].count().reset_index()
    edges_unordered = np.asarray(edges_unordered)  

    # only edges with valid symbols
    idx = []
    for i in range(len(edges_unordered)):
        if edges_unordered[i,0] in gene_ids and edges_unordered[i,1] in gene_ids:
            idx.append(i)
    edges_unordered = edges_unordered[idx]

    # build graph
    idx = np.array(gene_ids)
    idx_map = {j: i for i, j in enumerate(idx)}
    # the key (names) in edges_unordered --> the index (which row) in matrix
    def get_key(x):
        res = idx_map.get(x)
        return res if res else x
    edges = np.array(list(map(get_key, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape) #map：map(function, element):function on element

    if weights:
        adj = sp.coo_matrix((edges[:, 2], (edges[:, 0], edges[:, 1])),
                        shape=(gene_ids.shape[0], gene_ids.shape[0]),
                        dtype=np.float32)
    else:
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(gene_ids.shape[0], gene_ids.shape[0]),
                        dtype=np.float32)
    del idx,idx_map,edges_unordered
    
    # build symmetric adjacency matrix
    # keeps the bigger value
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # self loop
#    adj = adj + sp.eye(adj.shape[0])
           
    return adj

# if __name__=="__main__":
#     gsets = from_gmt("./data/C8.all.v7.5.1.symbols.gmt")
#     genes = set([g for gs in gsets for g in gsets[gs] ])