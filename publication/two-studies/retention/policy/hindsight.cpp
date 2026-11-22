// Attained hindsight lower bound within the prospective rule's per-row certificate.
#define RETENTION_LIBRARY
#include "retention.cpp"

int main(int argc,char** argv){
    if(argc!=7)throw runtime_error("usage: hindsight PREFIX SIDE REGISTER FLAGS YEAR CSV");
    Inputs x(argv[1],argv[2]);auto april=read_april(x,argv[3]);
    auto no_change=Inputs::read_mask(argv[4],x.m.ne);
    map<int,array<int,2>> gained;
    for(int i:x.switched){
        if(april[i]<0)continue;
        bool covered=false,removable=false,reachable=false;
        for(int e=x.m.at[i];e<x.m.at[i+1];++e){
            bool target=april[i]==x.inst.school[x.m.prog[e]];
            covered|=x.a[e]&&target;
            removable|=x.a[e]&&!x.b[e]&&!no_change[e];
            reachable|=!x.a[e]&&x.b[e]&&target&&!no_change[e];
        }
        if(covered)continue;
        gained[x.m.grade[i]][0]+=removable&&reachable;
        gained[x.m.grade[i]][1]++;
    }
    ofstream f(argv[6]);f<<"year,grade,additional_reachable,linked_switcher_misses\n";
    for(auto row:gained)f<<argv[5]<<','<<row.first<<','<<row.second[0]<<','<<row.second[1]<<'\n';
    if(!f)throw runtime_error("write results");
}
