import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch_geometric.nn import GAE
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from utils import from_gmt
from GS2Graph import net_from_BioGRID, net_from_MSigDB
from GNN import GAEncoder
from train import spilt_dataset, generate_loader, train
from utils import objectview, sparse_mtx_2_sparse_tensor
# load genesets for graph constructin
gsets = from_gmt("./data/C8.all.v7.5.1.symbols.gmt")


# load original data
adata = sc.read_10x_mtx(
    'C:/Users/croco/OneDrive - Emory University/data/pbmc3k/hg19', 
    var_names='gene_symbols', #"gene_ids"
    cache=False)

gene_ids = adata.var.index


# adj = net_from_MSigDB(gsets, gene_ids, weights=True)
adj = sp.load_npz('./tmp/adj_BG_w.npz')
adj = adj>0
# adj = net_from_BioGRID("bgfile",\
#            gene_ids, weights=True)

# pre-process of sc data 
# remove double lets (bad samples) > 2500
# keep cells with too low reads < 200
# keep genes
adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc.pp.filter_cells(adata, min_genes=200) # min 200 genes detected 0.6%
adata = adata[adata.obs.n_genes_by_counts < 2500, :] # not total read;
# adata = adata[adata.obs.pct_counts_mt < 5, :]

alldata = adata.X

shuffle_index = np.random.permutation(alldata.shape[0])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {'dataset': 'cora', 'num_layers': 2, 'batch_size': 1, \
    'hidden_dim': 8, 'out_dim':1, 'dropout': 0.5, 'epochs': 200, \
    'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, \
    'lr': 1e-2}

args = objectview(args)

# adj = sparse_mtx_2_sparse_tensor(adj).cuda()
idx,w = from_scipy_sparse_matrix(adj)
# idx = idx.cuda()

train_data, val_data, test_data = spilt_dataset(alldata.todense(), shuffle_index)
train_loader, val_loader, test_loader = generate_loader(train_data, val_data, test_data, args.batch_size)

model = GAE(GAEncoder(1, args.hidden_dim, args.out_dim))
# model = model.to(device)

# optimizor = torch.optim.Adam(model.parameters(), lr=args.lr) 
# adam not support sparse
optimizor = torch.optim.SGD(model.parameters(), momentum=0.9, lr= args.lr)
model = model.to(device)
idx = idx.to(device)

res = train(model, train_loader, test_loader, optimizor, idx, args)