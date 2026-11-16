"""Check the retained-partner lower bound for all full 3x3 profiles and every menu subgraph."""
from itertools import permutations,product
import json
from pathlib import Path
import numpy as np
from evaluate import da


def main():
    orders=np.array(list(permutations(range(3))))
    sides=np.array(list(product(range(len(orders)),repeat=3)))
    ranks=np.argsort(orders[sides],axis=2)
    menus=np.array(list(product([False,True],repeat=9))).reshape(-1,3,3)
    tested=0;different=0;minimum=3;maximum=0
    for sr in ranks:
        pref=ranks; sellers=np.broadcast_to(sr.T,pref.shape);E=np.ones_like(pref,dtype=bool)
        y=da(pref,sellers,E); assert (y<3).all()
        for h in np.array_split(menus,8):
            batched=np.repeat(pref,len(h),axis=0);other=np.repeat(sellers,len(h),axis=0);graph=np.tile(h,(len(pref),1,1));target=np.repeat(y,len(h),axis=0)
            m=da(batched,other,graph); actual=(m<3).sum(1); retained=np.take_along_axis(graph,target[:,:,None],axis=2).sum((1,2));slack=actual-retained
            assert np.all(slack>=0);tested+=len(slack);different+=int((slack>0).sum());minimum=min(minimum,int(slack.min()));maximum=max(maximum,int(slack.max()))
    result=dict(n=3,complete_profiles=len(ranks)**2,menu_graphs=len(menus),profile_by_menu_pairs=tested,strict_inequalities=different,min_slack=minimum,max_slack=maximum)
    print(json.dumps(result));Path(__file__).with_name('certificate-check.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
