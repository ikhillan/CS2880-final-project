"""Exact group transitions in fixed school-choice cohorts; outputs are aggregate counts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

class Market:
    def __init__(self,path):
        with path.open('rb') as f:
            def read(n):return np.fromfile(f,dtype='<i4',count=n)
            self.n,self.m,self.ne=map(int,read(3))
            self.at,self.j,self.pr,self.cap,self.off,self.ses,self.grade,self.ret,self.app,self.degree=[
                read(n) for n in [self.n+1,self.ne,self.ne,self.m,*([self.n]*6)]]
        self.i=np.repeat(np.arange(self.n),np.diff(self.at))
        self.cols=np.arange(self.ne)-self.at[self.i]
        assert np.isin(self.ses,[0,1]).all() and (np.diff(self.at)==self.degree+(self.ret>=0)).all()

    def choose(self,pool,res):
        if not len(pool):return pool
        # Returning holders have first claim. Fill up to r earmarked slots among eligible
        # remaining applications. Use ordinary priority for all residual capacity.
        def first(ordered,cap):
            dest=self.j[ordered]
            starts=np.maximum.accumulate(np.where(np.r_[True,dest[1:]!=dest[:-1]],np.arange(len(dest)),0))
            return (np.arange(len(dest))-starts)<cap[dest]
        ranked=pool[np.lexsort((self.pr[pool],self.j[pool]))]
        if res is None:return ranked[first(ranked,self.cap)]
        owner=self.i[ranked];dest=self.j[ranked];returning=(self.ret[owner]==dest)
        fixed=ranked[returning];remaining=self.cap-np.bincount(self.j[fixed],minlength=self.m)
        assert (remaining>=0).all()
        flags=np.flatnonzero((~returning)&(self.ses[owner]==1))
        protected=flags[first(ranked[flags],np.minimum(remaining,res))] if len(flags) else flags
        leftover=remaining-np.bincount(dest[protected],minlength=self.m)
        base=~returning;base[protected]=False;other=ranked[base]
        return np.r_[fixed,ranked[protected],other[first(other,leftover)] if len(other) else other]

    def solve(self,show,res=None):
        visible=np.flatnonzero(show);degree=np.bincount(self.i[visible],minlength=self.n)
        ptr=np.r_[0,np.cumsum(degree)[:-1]];end=ptr+degree
        free=np.flatnonzero(degree);held=np.empty(0,int)
        while len(free):
            new=visible[ptr[free]];ptr[free]+=1
            pool=np.r_[held,new];held=self.choose(pool,res)
            busy=np.zeros(self.n,bool);busy[self.i[held]]=True
            if len(held)!=int(busy.sum()):raise ValueError('duplicate offer')
            candidates=self.i[pool]
            freed=candidates[~busy[candidates]];free=freed[ptr[freed]<end[freed]]
        out=np.full(self.n,-1,int);out[self.i[held]]=held
        cmp=np.where(out>=0,out,self.at[1:]);prefer=show&(np.arange(self.ne)<cmp[self.i])
        extra=np.flatnonzero(prefer)
        # For a substitutable choice rule, none of the preferred alternative contracts may
        # be chosen even when all are added simultaneously; this is also verified independently.
        assert np.array_equal(np.sort(held),np.sort(self.choose(np.r_[held,extra],res)))
        assert (out[self.ret>=0]>=0).all()
        return out

def aggregate(y,case,pol,m,base,new,full,mask,baseline_mask):
    a,b=(base>=0),(new>=0)
    a_rank=np.where(a,base,m.at[1:]);b_rank=np.where(b,new,m.at[1:])
    dst=np.where(b,m.j[np.maximum(new,0)],-1)
    shown=np.bincount(m.i,weights=(m.j==m.off[m.i])&mask,minlength=m.n).astype(bool)
    changed=np.zeros(m.n,bool);changed[m.i[mask!=baseline_mask]]=True
    data=dict(n=np.ones(m.n,int), placed_base=a,placed=b,gain=~a&b,loss=a&~b,
              better_held=a&b&(b_rank<a_rank),worse_held=a&b&(b_rank>a_rank),same=a&b&(base==new),
              official=(m.off>=0),covered=shown,exact_official=(dst==m.off),
              exact_placed_official=(dst==m.off)&(m.off>=0),full_recipient=full>=0,
              kept_full=(full>=0)&b,new_nonrecipient=(full<0)&b,changed=changed)
    # Each combination is an exhaustive disjoint cell. Downstream aggregates are exact sums.
    frame=pd.DataFrame(data)
    frame['priority']=m.ses;frame['grade']=m.grade;frame['entitled']=(m.ret>=0).astype(int)
    frame['length']=np.select([m.degree<=1,m.degree<=5,m.degree<=10],[0,1,2],default=3)
    out=frame.groupby(['priority','grade','entitled','length'],as_index=False).sum()
    assert out.n.sum()==m.n and (out.placed_base+out.gain-out.loss==out.placed).all()
    assert (out.kept_full+out.new_nonrecipient==out.placed).all()
    out.insert(0,'policy',pol);out.insert(0,'mechanism',case);out.insert(0,'year',y)
    return out

def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    allcells=[]
    for year in args.years:
        f=args.old if year<=2021 else args.new
        m=Market(f/f'{year}.bin')
        masks={k:np.fromfile(f/f'{year}-{k}.mask',dtype=np.uint8).astype(bool) for k in ('blind','rank','full')}
        reserves=np.fromfile(args.new/f'{year}-reserves.bin',dtype='<i4')
        assert len(reserves)==m.m and (reserves>=0).all()
        flag=m.ses[m.i]==1
        masks['only_designated']=np.where(flag,masks['rank'],masks['blind'])
        masks['only_other']=np.where(flag,masks['blind'],masks['rank'])
        summaries=[]
        for case in args.mechanisms:
            res=None if case=='pooled' else reserves
            sols={}
            for name in ('full','blind','rank','only_designated','only_other'):
                sols[name]=m.solve(masks[name],res)
                if case=='reserved' and args.serial_checks:
                    ref=np.fromfile(args.serial_checks/f'{year}-{name}-reserve.edges',dtype='<i4')
                    np.testing.assert_array_equal(ref,sols[name])
                if case=='pooled' and name in ('full','blind','rank'):
                    chosen=sols[name];ref=np.fromfile(f/f'{year}-{name}.partners',dtype='<i4')
                    assert np.array_equal(ref,np.where(chosen>=0,m.j[np.maximum(chosen,0)],-1))
            for name in sols:
                block=aggregate(year,case,name,m,sols['blind'],sols[name],sols['full'],masks[name],masks['blind'])
                summaries.append(block)
            print(year,case,[(s,int((v>=0).sum())) for s,v in sols.items()],flush=True)
        df=pd.concat(summaries,ignore_index=True)
        df.to_csv(args.out/f'{year}-cells.csv.gz',index=False)
        allcells.append(df)
    print('completed',len(allcells),'years',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['old','new','out']:p.add_argument(k,type=Path)
    p.add_argument('--serial-checks',type=Path)
    p.add_argument('--years',type=int,nargs='+',default=[2020,2021,2022,2023])
    p.add_argument('--mechanisms',choices=['pooled','reserved'],nargs='+',default=['pooled','reserved'])
    main(p.parse_args())
