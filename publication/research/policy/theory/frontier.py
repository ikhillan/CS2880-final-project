"""Optimize model probability retained by an exactly quota-sized graph of target matching capacity."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


def capacity(allowed):
    return int(np.count_nonzero(maximum_bipartite_matching(csr_matrix(allowed),perm_type="column")>=0))


def solve(prob, allowed, top, target):
    """Inputs in pre-shuffled coordinates. Returned witness has exactly target pairs.

    Dummy columns let n-target applicants abstain; dummy rows absorb unused listing
    seats. Disallow dummy-to-dummy pairs to impose exact cardinality.
    """
    n,m=allowed.shape
    if target==0:
        return top.copy(),0.
    row_worst=np.where(top,prob,np.inf).argmin(axis=1)
    cutoff=prob[np.arange(n),row_worst]
    cost=np.maximum(cutoff[:,None]-prob,0.)
    size=n+m-target
    square=np.zeros((size,size),dtype=float)
    square[:n,:m]=np.where(allowed,cost,np.inf)
    square[n:,m:]=np.inf
    row,col=linear_sum_assignment(square)
    keep=(row<n)&(col<m)
    idx=row[keep];j=col[keep]
    assert len(idx)==target
    graph=top.copy()
    replace=~graph[idx,j]
    i,added=idx[replace],j[replace]
    graph[i,row_worst[i]]=False
    graph[i,added]=True
    assert np.array_equal(graph.sum(1),top.sum(1)) and np.all(graph[idx,j]) and not np.any(graph&~allowed)
    loss=float(cost[idx,j].sum())
    assert np.isclose(prob[top].sum()-prob[graph].sum(),loss,atol=1.e-10)
    return graph,loss
