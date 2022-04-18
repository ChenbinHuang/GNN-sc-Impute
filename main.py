import scanpy as sc
from utils import from_gmt
from GS2Graph import net_from_BioGRID, net_from_MSigDB
# load genesets for graph constructin
gsets = from_gmt("./data/C8.all.v7.5.1.symbols.gmt")


# load original data
adata = sc.read_10x_mtx(
    'C:/Users/croco/OneDrive - Emory University/data/pbmc3k/hg19', 
    var_names='gene_symbols', #"gene_ids"
    cache=False)

gene_ids = adata.var.index