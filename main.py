import torch
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch_geometric.nn import GAE
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import matplotlib.pyplot as plt

from utils import from_gmt
from GS2Graph import net_from_BioGRID, net_from_MSigDB
from GNN import GAEncoder, GAEncoder_Decoder
from train import spilt_dataset, generate_loader, train
from utils import objectview, sparse_mtx_2_sparse_tensor
from torch.utils import data
# load genesets for graph constructin
# gsets = from_gmt("./data/h.all.v7.5.1.symbols.gmt")


# load original data
adata = sc.read_10x_mtx(
    'C:/Users/croco/OneDrive - Emory University/data/pbmc3k/hg19', 
    var_names='gene_symbols', #"gene_ids"
    cache=False)

gene_ids = adata.var.index


# adj = net_from_MSigDB(gsets, gene_ids, weights=True)
adj = sp.load_npz('./tmp/adj_BG_w.npz')
# adj = adj>0
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

args = {'dataset': 'cora', 'num_layers': 2, 'batch_size': alldata.shape[0], \
    'hidden_dim': 64, 'out_dim':16, 'dropout': 0.5, 'epochs': 500, \
    'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, \
    'lr': 1e-2}

args = objectview(args)

# adj = sparse_mtx_2_sparse_tensor(adj).cuda()
idx,w = from_scipy_sparse_matrix(adj)
# idx = idx.cuda()

# train_data, val_data, test_data = spilt_dataset(alldata.todense(), shuffle_index)
# train_loader, val_loader, test_loader = generate_loader(train_data, val_data, test_data, args.batch_size)
train_data = np.asarray(alldata.todense()).astype(np.float32)
train_data = torch.FloatTensor(train_data).to(device)
train_loader = data.DataLoader(data.TensorDataset(train_data), batch_size = args.batch_size, shuffle = True)

# model = GAE(GAEncoder(args.batch_size, args.hidden_dim, args.out_dim))
model = GAEncoder_Decoder(args.batch_size, args.hidden_dim, args.out_dim)
# model = model.to(device)

# optimizor = torch.optim.Adam(model.parameters(), lr=args.lr) 
# adam not support sparse
optimizor = torch.optim.SGD(model.parameters(), momentum=0.9, lr= args.lr)
model = model.to(device)
idx = idx.to(device)

res = train(model, train_loader, optimizor, idx, args)

plt.plot(res['stats']['epoch'], res['stats']['loss'])
plt.show()

est = model(train_data.view(train_data.shape[1],-1), idx)
sum(sum(est.cpu().data.numpy() < 1e-3)) / (32738*2695)
sum(sum(train_data < 1e-3)) / (32738*2695)


# scanpy analysis
# adata.raw = adata
# adata = adata.raw.to_adata()
# adata.X = est.cpu().data.numpy().T
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color='leiden')

labels = [int(x) for x in adata.obs["leiden"]]

with open('./tmp/labels.txt', 'w') as f:
    f.write("[{}]".format(",".join([str(i) for i in labels])))


# from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# Dimension reduction and clustering libraries
import umap
# import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

mtx = est.cpu().data.numpy().T
standard_embedding = umap.UMAP(random_state=42).fit_transform(mtx)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1],c = labels ,s=1, cmap='Spectral')
plt.show()
