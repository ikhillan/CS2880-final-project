"""Construct and check an exact integer dual lower bound on a fixed-flow assignment."""
import numpy as np
from collections import deque

def certify(m,cost,selected,target):
    # Eliminate student nodes. A used edge j->i of cost -c_ij and any feasible
    # edge i->k of cost c_ik become a single residual arc j->k.
    s,t=m.m,m.m+1
    nodes=m.m+2
    partner=np.full(m.n,s,np.int32); matched_cost=np.zeros(m.n,np.int64)
    partner[m.a[selected]]=m.school[selected];matched_cost[m.a[selected]]=cost[selected]
    limit=np.iinfo(np.int64).max//4
    graph=np.full((nodes,nodes),limit,np.int64)
    np.minimum.at(graph,(partner[m.a],m.school),cost-matched_cost[m.a])
    np.minimum.at(graph,(m.school[selected],np.full(len(selected),s)),-cost[selected])
    occupancy=np.bincount(m.school[selected],minlength=m.m)
    graph[np.flatnonzero(occupancy<m.cap),t]=0
    graph[t,np.flatnonzero(occupancy)]=0
    # A virtual source of zero arcs starts every distance at zero. Build
    # label-correcting potentials; acceptance depends only on exact dual equalities.
    dist=np.zeros(nodes,np.int64);q=deque(range(nodes));inside=np.ones(nodes,bool)
    iterations=0
    while q:
        u=q.popleft();inside[u]=False
        v=np.flatnonzero(graph[u]+dist[u]<dist)
        dist[v]=graph[u,v]+dist[u]
        new=v[~inside[v]];q.extend(new);inside[new]=True
        iterations+=1
        if iterations>10_000_000:raise RuntimeError('Dual construction did not converge')
    ds,dt=int(dist[s]),int(dist[t])
    pi=dist[partner]-matched_cost
    student_used=(partner!=s)
    start_cost=ds-pi
    assert (start_cost[student_used]<=0).all() and (start_cost[~student_used]>=0).all()
    option_cost=cost+pi[m.a]-dist[m.school]
    assert (option_cost>=0).all() and (option_cost[selected]==0).all()
    exit_cost=dist[:m.m]-dt
    assert (exit_cost[occupancy>0]<=0).all() and (exit_cost[occupancy<m.cap]>=0).all()
    dual=int(target)*(dt-ds)+int(np.minimum(0,start_cost).sum())+int((np.minimum(0,exit_cost)*m.cap).sum())
    primal=int(cost[selected].sum())
    assert primal==dual
    return dict(primal=primal,dual=dual,option_arcs=len(cost),capacities_satisfied=int(len(selected)==target),
                residual_pops=iterations)
