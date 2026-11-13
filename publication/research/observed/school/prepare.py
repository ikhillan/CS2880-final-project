"""Prepare observed grade-markets and fit two prior-cohort offer models. Local private scratch only."""
import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

FILES = {2019: 'SAE_2019.zip', 2020:'SAE-2020.zip', 2021:'SAE_2021.rar'}
URLS={2019:'https://datosabiertos.mineduc.cl/wp-content/uploads/2021/12/SAE_2019.zip',
      2020:'https://datosabiertos.mineduc.cl/wp-content/uploads/2021/12/SAE-2020.zip',
      2021:'https://datosabiertos.mineduc.cl/wp-content/uploads/2022/02/SAE_2021.rar'}
PRIO=['prioridad_matriculado','prioridad_hermano','prioridad_hijo_funcionario','prioridad_exalumno']
AKEY=['rbd','cod_curso']
def read(p, year, tab, cols):
    path=p/FILES[year]
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:
            f=next(x for x in z.namelist() if Path(x).name.startswith(tab+'_') and x.endswith('.csv'))
            with z.open(f) as h:return pd.read_csv(h,usecols=cols,sep=';',encoding='utf-8-sig',low_memory=False)
    items=subprocess.check_output(['bsdtar','-tf',str(path)]).decode().splitlines()
    f=next(x for x in items if Path(x).name.startswith(tab+'_') and x.endswith('.csv'))
    data=subprocess.check_output(['bsdtar','-xOf',str(path),f]);return pd.read_csv(io.BytesIO(data),usecols=cols,sep=';',encoding='utf-8-sig',low_memory=False)

def prepare(p,year):
    a=read(p,year,'A1',AKEY+['cod_nivel','vacantes','vacantes_regular','vacantes_prioritarios','vacantes_pie','vacantes_alta_exigencia_t','vacantes_alta_exigencia_r'])
    b=read(p,year,'B1',['mrun','cod_nivel','prioritario']).sort_values('mrun').reset_index(drop=True)
    c=read(p,year,'C1',['mrun','cod_nivel','preferencia_postulante','agregada_por_continuidad','loteria_original']+AKEY+PRIO)
    d=read(p,year,'D1',['mrun','cod_nivel','rbd_admitido','cod_curso_admitido'])
    assert (a.vacantes>=0).all() and not a.duplicated(AKEY).any() and not b.mrun.duplicated().any()
    assert not c.duplicated(['mrun','preferencia_postulante']).any() and not c.duplicated(['mrun']+AKEY).any()
    assert not d.mrun.duplicated().any() and set(b.mrun)==set(d.mrun)==set(c.mrun)
    c=c.merge(a[AKEY+['cod_nivel']].assign(program=np.arange(len(a))),on=AKEY+['cod_nivel'],how='left',validate='many_to_one')
    c=c.merge(b[['mrun','cod_nivel']].assign(person=np.arange(len(b))),on=['mrun','cod_nivel'],how='left',validate='many_to_one').sort_values(['person','preferencia_postulante']).reset_index(drop=True)
    d=d.sort_values('mrun');assert np.array_equal(d.mrun,b.mrun)
    lookup=pd.MultiIndex.from_frame(a[AKEY]); keys=pd.DataFrame({k:pd.to_numeric(d[x],errors='coerce').fillna(-1).astype('int64') for k,x in zip(AKEY,['rbd_admitido','cod_curso_admitido'])})
    official=lookup.get_indexer(pd.MultiIndex.from_frame(keys));assert np.array_equal(official<0,keys.rbd.to_numpy()<0)
    secured=c[c.prioridad_matriculado==1];assert not secured.person.duplicated().any() and (c.agregada_por_continuidad<=c.prioridad_matriculado).all()
    end=np.full(len(b),10**7,int);end[secured.person]=secured.preferencia_postulante
    n_raw=len(c);c=c[c.preferencia_postulante.to_numpy()<=end[c.person.to_numpy()]].reset_index(drop=True)
    i=c.person.to_numpy(dtype='int32');j=c.program.to_numpy(dtype='int32');matched=(official[i]==j).astype(float)
    tier=np.select([c[k].to_numpy()==1 for k in PRIO],np.arange(4),default=4)
    ties=c[['program','loteria_original']].duplicated().sum()
    priord=np.lexsort((i,c.loteria_original.to_numpy(),tier,j));prior=np.empty(len(j),'int32');prior[priord]=np.arange(len(j))
    assert np.all(pd.notna(i)) and np.all(pd.notna(j))
    # All continuing seats we protect must fit in each program. Newcomer capacity is
    # pooled public vacancies, with returning applicants first at their current program.
    ret_count=np.bincount(j[tier==0],minlength=len(a))
    caps=a.vacantes.to_numpy(dtype='int32').copy()
    short=np.maximum(ret_count-caps,0)
    caps=np.maximum(caps,ret_count).astype('int32')
    fallback=np.full(len(b),-1,'int32');fallback[i[tier==0]]=j[tier==0]
    appended=np.full(len(b),-1,'int32');m=c.agregada_por_continuidad.to_numpy()==1;appended[i[m]]=j[m]
    degree=np.bincount(i[tier!=0],minlength=len(b))
    nload=np.bincount(j,minlength=len(a))
    pressure=np.log((1+nload[j])/(1+caps[j]))
    x=pd.DataFrame(dict(log_capacity=np.log1p(caps[j]),log_load=np.log1p(nload[j]),pressure_sq=pressure**2,
                        sibling=(tier==1)*1.,staff=(tier==2)*1.,alumnus=(tier==3)*1.,
                        log_list_length=np.log1p(degree[i]),grade_scaled=c.cod_nivel.to_numpy()/12))
    info=dict(year=year,applicants=len(b),programs=len(a),raw_edges=n_raw,excluded_below_entitlement=n_raw-len(c),included_edges=len(c),
              official_assigned=int((official>=0).sum()),official_outside_retained_list=int((official>=0).sum()-matched.sum()),
              protected_entitlements=int((tier==0).sum()),administrative_appended=int(m.sum()),raised_to_protect_incumbents=int(short.sum()),
              programs_with_cap_raised=int((short>0).sum()),lottery_ties_in_program=int(ties),source_sha256=hashlib.sha256((p/FILES[year]).read_bytes()).hexdigest(),
              source=URLS[year],priority_flags=int(b.prioritario.sum()))
    data=dict(i=i,j=j,n=len(b),m=len(a),tier=tier,priority=prior,cap=caps,off=official,sen=b.prioritario.to_numpy(),g=b.cod_nivel.to_numpy(),
              ret=fallback,appended=appended,degree=degree)
    return a[AKEY],c.preferencia_postulante.to_numpy(),matched,x,data,info

def fit(X,y):
    mean=X.mean(axis=0);sd=X.std(axis=0);sd=np.maximum(sd,1e-6)
    Z=np.ascontiguousarray((X-mean)/sd);n=len(y);penalty=.0001
    def likelihood(beta):
        v=Z@beta[1:]+beta[0];err=expit(v)-y
        loss=np.mean(np.logaddexp(0,v)-y*v)+penalty*.5*(beta[1:]@beta[1:])
        return loss,np.r_[err.mean(), (err@Z)/n+penalty*beta[1:]]
    res=minimize(likelihood,np.zeros(Z.shape[1]+1),method='L-BFGS-B',jac=True,options={'ftol':1e-12,'gtol':1e-7,'maxiter':250})
    assert res.success,str(res)
    return dict(mean=mean.tolist(),sd=sd.tolist(),coef=res.x.tolist(),gradient=float(np.max(np.abs(res.jac))))
def predict(X,fit):
    # Scores are logits. Ranking does not require the sigmoid.
    return ((X-np.array(fit['mean']))/np.array(fit['sd']))@np.array(fit['coef'][1:])+fit['coef'][0]

def main(args):
    p=args.data;out=args.out;scratch=args.scratch;out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    reg={};cohorts=[];hist=None
    for year in FILES:
        a,position,y,x,d,info=prepare(p,year);cohorts.append(info)
        edit=d['tier']!=0;j=d['j'];count=np.bincount(j[edit],minlength=len(a));win=np.bincount(j[edit],weights=y[edit],minlength=len(a))
        if year==2019:
            globalrate=y[edit].mean()
            hist=a.copy();hist['win']=win;hist['n']=count
            ph=(win[j]-y*edit+20*globalrate)/(count[j]-edit.astype(int)+20)
        else:
            h=a.merge(hist,how='left',on=AKEY,validate='one_to_one').fillna(0)
            ph=((h.win+20*globalrate)/(h.n+20)).to_numpy()[j]
        x['log_odds_prior']=np.log(ph/(1-ph))
        x['prior_sq']=x.log_odds_prior.to_numpy()**2
        for name in ['blind','rank']:
            if name=='rank':
                x['log_rank']=np.log(position);x['rank1']=(position==1)*1.;x['rank_pressure']=x.log_rank*np.log((1+np.bincount(j,minlength=len(a))[j])/(1+d['cap'][j]))
            if year==2019:
                reg[name]=fit(x.to_numpy()[edit],y[edit]);reg[name]['features']=x.columns.tolist();continue
            assert x.columns.tolist()==reg[name]['features']
            d[name]=predict(x.to_numpy(),reg[name])
        if year!=2019:
            n=d['n'];m=d['m'];N=len(j);offset=np.r_[0,np.cumsum(np.bincount(d['i'],minlength=n))].astype('int32')
            assert offset[-1]==N and np.all((d['g']>=-1)&(d['g']<=12))
            with (scratch/f'{year}.bin').open('wb') as f:
                for arr in [np.array([n,m,N]),offset,j,d['priority'],d['cap'],d['off'],d['sen'],d['g'],d['ret'],d['appended'],d['degree']]:
                    np.asarray(arr,dtype='<i4').tofile(f)
                for name in ['blind','rank']:np.asarray(d[name],dtype='<f8').tofile(f)
        print(year,info,flush=True)
    (out/'fit-and-inputs.json').write_text(json.dumps(dict(train_year=2019,models=reg,cohorts=cohorts),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--scratch',required=True,type=Path);main(p.parse_args())
