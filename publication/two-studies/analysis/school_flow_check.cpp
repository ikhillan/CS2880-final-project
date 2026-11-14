// Independent full-cohort Dinic implementation. No grade decomposition, scipy, or archived flow code.
#define main archived_replay_main
#include "replay.cpp"
#undef main
struct Arc { int to, next, residual; };
struct Flow {
    vector<int> head, layer, cursor; vector<Arc> arcs; int s,t;
    Flow(int n,int m):head(n+m+2,-1),s(n+m),t(s+1){}
    int add(int u,int v,int k) {
        int i=arcs.size(); arcs.push_back({v,head[u],k});head[u]=i;
        arcs.push_back({u,head[v],0});head[v]=i+1;return i;
    }
    bool layers() {
        layer.assign(head.size(),-1);layer[s]=0;queue<int> frontier;frontier.push(s);
        while(!frontier.empty()) {int u=frontier.front();frontier.pop();
            for(int i=head[u];i>=0;i=arcs[i].next) {
                Arc a=arcs[i];if(a.residual<=0 || layer[a.to]>=0)continue;
                layer[a.to]=layer[u]+1;frontier.push(a.to);
            }
        }
        return layer[t]>=0;
    }
    int send(int u,int v) {
        if(u==t)return v;
        for(int &i=cursor[u];i>=0;i=arcs[i].next) {auto &a=arcs[i];
            if(a.residual<=0 || layer[a.to]!=layer[u]+1)continue;
            int sent=send(a.to,min(v,a.residual));if(sent){a.residual-=sent;arcs[i^1].residual+=sent;return sent;}
        }
        return 0;
    }
    int solve() {
        int value=0;
        while(layers()) {cursor=head;while(int k=send(s,1000000000))value+=k;}
        return value;
    }
};
struct Solution { vector<int> edges; int eligible=0;long long reachable=0; };
Solution independent(const Market& m,const vector<char>& visible,const vector<int>& cap) {
    Flow f(m.n,m.m);vector<pair<int,int>> mapping;vector<int> left(m.n),right(m.m);
    for(int j=0;j<m.m;j++){assert(cap[j]>=0);if(cap[j])f.add(m.n+j,f.t,cap[j]);}
    for(int i=0;i<m.n;i++) {
       for(int e=m.at[i];e<m.at[i+1];e++) {int j=m.prog[e];if(!visible[e])continue;
          if(cap[j]>0){left[i]=right[j]=1;mapping.push_back({f.add(i,m.n+j,1),e});}
       }
       if(left[i])f.add(f.s,i,1);
    }
    int value=f.solve(); Solution result;
    for(auto [index,edge]:mapping)if(f.arcs[index].residual==0)result.edges.push_back(edge);
    assert(value==(int)result.edges.size());result.eligible=accumulate(left.begin(),left.end(),0);
    for(int j=0;j<m.m;j++)if(right[j])result.reachable+=cap[j];
    return result;
}
int main(int argc,char**argv){
    assert(argc==4);int year=stoi(argv[2]);Market m(argv[1]);string dir=argv[3];
    cout<<"year,policy,scope,maximum,source_renters_or_pupils,reachable_capacity,baseline,after,worse,named\n";
    for(string policy:{"blind","rank"}){
       ifstream fi(dir+"/"+to_string(year)+"-"+policy+".mask",ios::binary);
       vector<char> shown=readv<char>(fi,m.ne);
       auto before=m.da(shown);m.verify(before,shown);
       int placed=count_if(before.begin(),before.end(),[](int x){return x>=0;});
       auto full=independent(m,shown,m.cap);
       // A capacity-zero school cannot house an applicant; many unused schools may have ample capacity.
       assert((int)full.edges.size()>=placed);
       cout<<year<<","<<policy<<",display,"<<full.edges.size()<<","<<full.eligible<<","<<full.reachable<<","<<placed<<",,,\n";
       vector<int> free=m.cap,owner(m.ne);
       for(int i=0;i<m.n;i++){for(int e=m.at[i];e<m.at[i+1];e++)owner[e]=i;
          if(before[i]>=0)assert(--free[m.prog[before[i]]]>=0);
       }
       for(string scope:{"top5","any"}){
          vector<char> residual(m.ne);
          for(int i=0;i<m.n;i++)if(before[i]<0)
            for(int e=m.at[i];e<m.at[i+1];e++)if((scope=="any"||e-m.at[i]<5)&&free[m.prog[e]]>0){
                assert(!shown[e]);residual[e]=true;
            }
          auto pick=independent(m,residual,free);
          vector<char> revised=shown;
          for(int new_edge:pick.edges){int i=owner[new_edge],old=m.at[i+1]-1;
              while(old>=m.at[i]&&(!revised[old]||m.prog[old]==m.ret[i]))old--;
              assert(old>=m.at[i]);revised[old]=0;revised[new_edge]=1;
          }
          auto after=m.da(revised);m.verify(after,revised);int losses=0,named=0,final=0;
          for(int i=0;i<m.n;i++){final+=after[i]>=0;losses+=before[i]>=0&&(after[i]<0||after[i]>before[i]);}
          for(int e:pick.edges)named+=after[owner[e]]==e;
          assert(!losses&&named==(int)pick.edges.size()&&final==placed+named);
          cout<<year<<","<<policy<<","<<scope<<","<<pick.edges.size()<<","<<pick.eligible<<","<<pick.reachable<<","<<placed<<","<<final<<","<<losses<<","<<named<<"\n";
       }
    }
}
