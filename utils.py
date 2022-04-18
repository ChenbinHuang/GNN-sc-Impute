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


