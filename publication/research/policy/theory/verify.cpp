// Independent finite enumeration, using bit masks only for 3 people and at most 3 programs.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>
using V=std::vector<int>;
using VV=std::vector<V>;
struct Counts {
  long long opt=0,enumerated_displays=0,profile=0,worst_graphs=0,omission=0,threat=0,rankfloor=0;
};
int size(const V& a){return std::count_if(a.begin(),a.end(),[](int j){return j>=0;});}
int power(int n,int k){int r=1;while(k-->0)r*=n;return r;}
V digits(int code,int b,int n){
  V out(n);for(int& x:out){x=code%b;code/=b;}return out;
}
bool shown(int display,int i,int j,int m){return (display>>(i*m+j))&1;}
int degree(int display,int i,int m){
  int d=0;for(int j=0;j<m;j++)d+=shown(display,i,j,m);return d;
}
VV assignments(int n,int m,const V& cap){
  VV out;
  for(int a=0;a<power(m+1,n);a++){
    auto row=digits(a,m+1,n); V load(m,0);
    for(auto& j:row) {--j;if(j>=0)++load[j];}
    bool ok=true;for(int j=0;j<m;j++)ok&=load[j]<=cap[j];
    if(ok)out.push_back(row);
  }return out;
}
bool visible(const V& a,int D,int m){
  for(int i=0;i<(int)a.size();i++)if(a[i]>=0&&!shown(D,i,a[i],m))return false;
  return true;
}
V da(int n,int m,int D,const V& cap,const VV& ranks,const VV& priorities){
  V result(n,-1),next(n,0),free(n);std::iota(free.begin(),free.end(),0);
  VV held(m);
  while(!free.empty()){
    int i=free.back();free.pop_back();
    while(next[i]<m){
      int j=ranks[i][next[i]++];if(!shown(D,i,j,m))continue;
      auto& h=held[j];
      if((int)h.size()<cap[j]){h.push_back(i);result[i]=j;break;}
      auto worst=std::max_element(h.begin(),h.end(),[&](int a,int b){return priorities[j][a]<priorities[j][b];});
      if(priorities[j][*worst]>priorities[j][i]){free.push_back(*worst);result[*worst]=-1;*worst=i;result[i]=j;break;}
    }
  }return result;
}
VV perms(int n){
  V a(n);std::iota(a.begin(),a.end(),0);VV ps;
  do{ps.push_back(a);}while(std::next_permutation(a.begin(),a.end()));return ps;
}
void frontier(int n,int m,Counts& cnt){
  const int lim=1<<(m*n);
  for(int v=0;v<(1<<m);v++){
    V cap(m);for(int j=0;j<m;j++)cap[j]=1+((v>>j)&1);
    auto alloc=assignments(n,m,cap);V nu(lim,0);
    for(int E=0;E<lim;E++)for(const V& A:alloc)if(visible(A,E,m))nu[E]=std::max(nu[E],size(A));
    for(int E=0;E<lim;E++)for(int score=0;score<lim;score++){
      for(int q=1;q<=m;q++){
        V count(n),tau(n),Q(nu[E]+1,std::numeric_limits<int>::min());
        for(int i=0;i<n;i++){
          count[i]=std::min(degree(E,i,m),q);
          tau[i]=(count[i]>0&&degree(score&E,i,m)>=count[i]);
        }
        for(int D=E;;D=(D-1)&E){
          bool proper=true;for(int i=0;i<n;i++)proper&=degree(D,i,m)==count[i];
          if(proper){
            ++cnt.enumerated_displays;
            for(int r=0;r<=nu[D];r++)Q[r]=std::max(Q[r],__builtin_popcount((unsigned)(score&D)));
          }
          if(!D)break;
        }
        V loss(nu[E]+1,10000);
        for(const V& A:alloc)if(visible(A,E,m)){
          int c=0;for(int i=0;i<n;i++)if(A[i]>=0)c+=std::max(0,tau[i]-(int)shown(score,i,A[i],m));
          loss[size(A)]=std::min(loss[size(A)],c);
        }
        for(int r=0;r<=nu[E];r++){
          assert(Q[r]==Q[0]-loss[r]);
          if(r>0)assert(loss[r]>=loss[r-1]);
          if(r>1)assert(loss[r]-loss[r-1]>=loss[r-1]-loss[r-2]);
          ++cnt.opt;
        }
      }
    }
  }
}
void worst_omission(int n,int m,Counts& cnt){
  int lim=1<<(m*n);VV rows=perms(m),cols=perms(n);
  for(int cc=0;cc<(1<<m);cc++){
    V cap(m);for(int j=0;j<m;j++)cap[j]=1+((cc>>j)&1);
    auto alloc=assignments(n,m,cap);V lower(lim,n),upper(lim,0),achieved_min(lim,n),achieved_max(lim,0);
    for(int E=0;E<lim;E++)for(auto& a:alloc)if(visible(a,E,m)){
      V occ(m);for(int j:a)if(j>=0)occ[j]++;
      bool maximal=true;
      for(int i=0;i<n;i++)if(a[i]<0)for(int j=0;j<m;j++)if(shown(E,i,j,m)&&occ[j]<cap[j])maximal=false;
      if(maximal)lower[E]=std::min(lower[E],size(a));
      upper[E]=std::max(upper[E],size(a));
    }
    for(int R=0;R<power(rows.size(),n);R++)for(int P=0;P<power(cols.size(),m);P++){
      VV r,p;for(auto a:digits(R,rows.size(),n))r.push_back(rows[a]);
      for(auto b:digits(P,cols.size(),m))p.push_back(cols[b]);
      VV all(lim);
      for(int D=0;D<lim;D++){
        all[D]=da(n,m,D,cap,r,p);int M=size(all[D]);
        achieved_min[D]=std::min(achieved_min[D],M);achieved_max[D]=std::max(achieved_max[D],M);
        assert(M*2>=upper[D]);++cnt.profile;
      }
      // With 3 people and 2 capacitated programs, independently vary every
      // outsider display, hold the incumbents' rows fixed and compare the two
      // DA allocations against the exact threat matching and its zero-rank floor.
      if(n==3&&m==2) for(int I=1;I+1<(1<<n);I++){
        int fixed=0; for(int i=0;i<n;i++)if((I>>i)&1)fixed|=((1<<m)-1)<<(m*i);
        int outside=(lim-1)^fixed;
        for(int E=0;E<lim;E++)for(int hmask=outside;;hmask=(hmask-1)&outside){
          int D=(E&fixed)|hmask;
          int lost=0,worse=0;VV adj(n);
          for(int i=0;i<n;i++)if((I>>i)&1){
            int j=all[E][i],now=all[D][i]; if(j<0)continue;
            if(now<0)lost++;
            if(now<0||std::find(r[i].begin(),r[i].end(),j)<std::find(r[i].begin(),r[i].end(),now))worse++;
            for(int h=0;h<n;h++)if(!((I>>h)&1)&&shown(D,h,j,m)&&p[j][h]<p[j][i])adj[h].push_back(i);
          }
          // With three applicants, one of the two sides of this certificate has size one.
          int capthreat=0;for(auto& row:adj)if(!row.empty())capthreat=1;
          assert(lost<=capthreat);++cnt.threat;
          if(!capthreat){assert(!worse);++cnt.rankfloor;}
          if(!hmask)break;
        }
      }
      for(int E=0;E<lim;E++)for(int D=E;;D=(D-1)&E){
        V old(m),now(m);int omitted=0,lost=0,gained=0;
        for(int i=0;i<n;i++){
          int a=all[E][i],b=all[D][i];
          if(a>=0&&!shown(D,i,a,m)){++omitted;++old[a];if(b>=0)++now[b];}
          if(a>=0&&b<0)++lost;
          if(a<0&&b>=0)++gained;
        }
        for(int j=0;j<m;j++)omitted-=std::min(old[j],now[j]);
        assert(std::max(lost,gained)<=omitted);++cnt.omission;
        if(!D)break;
      }
    }
    for(int D=0;D<lim;D++){assert(achieved_min[D]==lower[D]&&achieved_max[D]==upper[D]);++cnt.worst_graphs;}
  }
}
int main(){
  Counts c;
  frontier(3,3,c);
  worst_omission(3,2,c);worst_omission(2,3,c);
  std::cout<<"{\n  \"independent_optimal_values\": "<<c.opt<<",\n  \"enumerated_quota_displays\": "<<c.enumerated_displays
    <<",\n  \"preference_priority_graph_combinations\": "<<c.profile<<",\n  \"graph_capacity_extrema\": "<<c.worst_graphs
    <<",\n  \"responsive_omission_tests\": "<<c.omission<<",\n  \"responsive_threat_tests\": "<<c.threat<<",\n  \"zero_threat_rank_tests\": "<<c.rankfloor<<"\n}\n";
}
