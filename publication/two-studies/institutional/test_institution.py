"""Compare the institution.hpp kernel to a small expanded-contract synchronous DA."""
import argparse
from itertools import chain
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory
import numpy as np

def reference(n, programs, owner, reports, shown, priority_flags, current, grades, links, chairs, order, disadvantaged, honors, pie, exam, transition):
    held_by={}
    assignment=[None]*n
    lists={}
    priorities={}
    for grade in sorted(set(grades),reverse=True):
        applicants=[i for i,g in enumerate(grades) if g==grade]
        for i in applicants:
            linked=[(assignment[older],sibling,block) for older,younger,sibling,block in links if younger==i and assignment[older] is not None]
            sibling_schools={p[0]%3 for p,s,b in linked if s}
            block_schools={p[0]%3 for p,s,b in linked if b}
            edges=[e for e in reports[i] if shown[e]]
            edges.sort(key=lambda e:programs[e]%3 not in block_schools)
            rows=[]
            for e in edges:
                j=programs[e]; flag=priority_flags[e]
                sibling0=bool(flag&2)
                sibling1=j%3 in sibling_schools
                incumbent=j==current[i]
                if incumbent:
                    cats=[1,0,3] if disadvantaged[i] else [0,1,3]
                    cats=[2]+cats if flag&16 else cats+[2]
                else:
                    cats=[2,3,0,1] if (sibling0 or sibling1) and not disadvantaged[i] else [2,3,1,0]
                for cat in cats:
                    spec=0
                    if incumbent: tier=0
                    elif cat==2 and pie[e]>=0: tier,spec=1,pie[e]
                    elif cat==3 and (exam[e]>=0 if transition[j] else honors[i]):tier,spec=1,exam[e] if transition[j] else 0
                    elif sibling0:tier=2
                    elif sibling1:tier=3
                    elif cat==1 and disadvantaged[i]:tier=4
                    elif flag&4:tier=5
                    elif flag&8:tier=6
                    else:tier=7
                    slot=(j,cat)
                    rows.append(slot);priorities[i,slot]=(tier,spec,order[e],i)
                if incumbent:
                    rows.append((j,4+i))
            lists[i]=rows
        positions={i:0 for i in applicants}
        while True:
            offers={}
            for i in applicants:
                if assignment[i] is None and positions[i]<len(lists[i]):
                    slot=lists[i][positions[i]];positions[i]+=1
                    offers.setdefault(slot,[]).append(i)
            if not offers:break
            for (j,cat),new in offers.items():
                slot=(j,cat)
                pool=held_by.get(slot,[])+new
                if cat<4:
                    pool.sort(key=lambda i:priorities[i,slot])
                    cap=chairs[j][cat]
                else:cap=1
                held_by[slot]=pool[:cap]
                for i in pool:assignment[i]=slot if i in held_by[slot] else None
    return assignment

def fixture(rng,prefix):
    n,m=9,6
    grades=[1]*4+[0]*5
    prog=[];flags=[];order=[];pie=[];exam=[];shown=[];offset=[0];ret=[];reports=[];owner=[]
    ses=[rng.randrange(2) for _ in grades];honors=[rng.randrange(2) for _ in grades]
    transition=[rng.randrange(2) for _ in range(m)]
    chairs=[[rng.randrange(3) for _ in range(4)] for j in range(m)]
    if rng.random()<.2:chairs[rng.randrange(m)]=[0,0,0,0]
    for i,g in enumerate(grades):
        row=[]
        pool=rng.sample(range(3*g,3*g+3),rng.randint(1,3))
        current=rng.choice(pool) if rng.random()<.35 else -1;ret.append(current)
        for j in pool:
            row.append(len(prog));prog.append(j);owner.append(i)
            flags.append((j==current)+2*(rng.random()<.25)+4*(rng.random()<.25)+8*(rng.random()<.25)+16*(rng.random()<.25))
            order.append(rng.randrange(3));pie.append(rng.choice([-1,-1,1,2]));exam.append(rng.choice([-1,-1,1,2]))
            shown.append(j==current or rng.random()<.7)
        reports.append(row);offset.append(len(prog))
    links=[(hi,lo,rng.randrange(2),rng.randrange(2)) for hi in range(4) for lo in range(4,9) if rng.random()<.2]
    expected=reference(n,prog,owner,reports,shown,flags,ret,grades,links,chairs,order,ses,honors,pie,exam,transition)
    with prefix.with_suffix('.bin').open('wb') as f:
        for arr in ([n,m,len(prog)],offset,prog,order,[sum(r) for r in chairs],[-1]*n,ses,grades,ret,[-1]*n,[len(r) for r in reports]):
            np.asarray(arr,'<i4').tofile(f)
        for _ in range(2):np.zeros(len(prog),'<f8').tofile(f)
    with prefix.with_suffix('.side').open('wb') as f:
        data=[n,m,len(prog),len(links),*([0,1,2]*2),0,0,0,1,1,1]
        np.asarray(data,'<i4').tofile(f)
        caps=[]
        for j,(reg,dis,p,a) in enumerate(chairs):caps.extend([reg,dis,p,a if transition[j] else 0,0 if transition[j] else a])
        np.asarray(caps,'<i4').tofile(f);np.asarray(honors,'u1').tofile(f);np.asarray(flags,'u1').tofile(f)
        for a in (order,pie,exam,links):np.asarray(a,'<i4').tofile(f)
    np.asarray(shown,'u1').tofile(prefix.with_suffix('.mask'))
    expected=[(-1,-1) if slot is None else (slot[0],min(4,slot[1])) for slot in expected]
    return list(chain.from_iterable(expected))

def main(args):
    here=Path(__file__).resolve().parent
    with TemporaryDirectory() as tmp:
        temp=Path(tmp);src=temp/'driver.cpp';exe=temp/'driver'
        src.write_text('''#define main ignore_main
#include "replay.cpp"
#undef main
#include "institution.hpp"
int main(int argc,char**argv){for(int k=1;k<argc;k++){string path=argv[k]; Market m(path+".bin");
    ifstream in(path+".mask",ios::binary);auto mask=readv<char>(in,m.ne);
    Institutional engine(m,path+".side");auto o=engine.clear(mask);
    for(int i=0;i<m.n;++i){int e=o.partner[i];cout<<(e<0?-1:m.prog[e])<<" "<<o.category[i]<<" ";
      for(int f=m.at[i];f<m.at[i+1];++f)if(mask[f]&&engine.blocks(i,f,o))return 6;}
    cout<<"\\n";}}
''')
        subprocess.run(["c++","-std=c++17","-O2",f"-I{args.school}",f"-I{here}",src,"-o",exe],check=True)
        rng=random.Random(89211);paths=[];expect=[]
        for k in range(args.markets):
            p=temp/str(k);paths.append(p);expect.append(fixture(rng,p))
        lines=subprocess.check_output([exe,*paths],text=True).splitlines()
        assert [[int(v) for v in line.split()] for line in lines]==expect
    print(f"{args.markets} independent contract-by-contract DA comparisons passed")
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--school",required=True,type=Path)
    p.add_argument("--markets",type=int,default=240)
    main(p.parse_args())
