// Independently implemented sequential DA, with -1 for an unmatched renter.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
using Matrix = std::vector<std::vector<int>>;
using std::vector;
struct Totals { long long profiles=0, losses=0, gains=0, strict=0, tight=0; };
vector<int> match(const Matrix& order,const Matrix& priority,const Matrix& graph) {
    const int n=order.size(), m=priority.size();
    vector<int> held(m,-1), pos(n,0), out(n,-1), waiting(n);
    std::iota(waiting.begin(),waiting.end(),0);
    while (!waiting.empty()) {
        int i=waiting.back();waiting.pop_back();
        while(pos[i]<m) {
            int j=order[i][pos[i]++];
            if(!graph[i][j]) continue;
            int old=held[j];
            if(old>=0 && priority[j][old]<priority[j][i]) continue;
            if(old>=0) {out[old]=-1;waiting.push_back(old);}
            held[j]=i;out[i]=j;break;
        }
    }
    return out;
}
void verify(const Matrix& order,const Matrix& priority,const Matrix& full,const Matrix& display,Totals& s) {
    auto x=match(order,priority,full), y=match(order,priority,display);
    vector<bool> k(priority.size()), t(priority.size());
    int r=0,lost=0,gain=0;
    for(size_t i=0;i<x.size();i++) {
        if(x[i]>=0 && !display[i][x[i]]) {
            ++r;k[x[i]]=true;
            if(y[i]>=0)t[y[i]]=true;
        }
        lost+=x[i]>=0 && y[i]<0;
        gain+=x[i]<0 && y[i]>=0;
    }
    int overlap=0;
    for(size_t j=0;j<k.size();j++) overlap+=k[j] && t[j];
    assert(lost<=r-overlap && gain<=r-overlap);
    ++s.profiles;s.losses+=lost>0;s.gains+=gain>0;s.strict+=overlap>0;s.tight+=lost>0 && lost==r-overlap;
}
vector<vector<int>> perms(int n) {
    vector<int> a(n);std::iota(a.begin(),a.end(),0);
    vector<vector<int>> out;
    do {out.push_back(a);} while(std::next_permutation(a.begin(),a.end()));
    return out;
}
void sweep(int n,int m, Totals& s){
    auto rows=perms(m), cols=perms(n);
    int profiles=1,graph_pairs=1;
    for(int i=0;i<n;i++)profiles*=rows.size();
    for(int j=0;j<m;j++)profiles*=cols.size();
    for(int e=0;e<n*m;e++)graph_pairs*=3;
    for(int p=0;p<profiles;p++) {
        Matrix order(n),priority(m,vector<int>(n));int code=p;
        for(auto& row:order){row=rows[code%rows.size()]; code/=rows.size();}
        for(auto& col:priority){
            auto rank=cols[code%cols.size()]; code/=cols.size();
            for(int i=0;i<n;i++)col[rank[i]]=i;
        }
        Matrix full(n,vector<int>(m)), display=full;
        for(int g=0;g<graph_pairs;g++){
            int v=g;
            for(int i=0;i<n;i++)for(int j=0;j<m;j++){
                int digit=v%3;v/=3;full[i][j]=digit>0;display[i][j]=digit==2;
            }
            verify(order,priority,full,display,s);
        }
    }
}
int main() {
    Totals e,r;
    for(auto pair:std::array<std::array<int,2>,3>{{{2,2},{2,3},{3,2}}}) sweep(pair[0],pair[1],e);
    std::mt19937 gen(28801314);
    for(int b=0;b<100000;b++) {
        int n=2+gen()%7,m=2+gen()%7;
        Matrix a(n,vector<int>(m)),p(m,vector<int>(n)),f(n,vector<int>(m)),d=f;
        for(auto& row:a){std::iota(row.begin(),row.end(),0);std::shuffle(row.begin(),row.end(),gen);}
        for(auto& col:p){std::iota(col.begin(),col.end(),0);std::shuffle(col.begin(),col.end(),gen);}
        for(int i=0;i<n;i++)for(int j=0;j<m;j++){f[i][j]=(gen()%4)!=0;d[i][j]=f[i][j] && (gen()%2)==0;}
        verify(a,p,f,d,r);
    }
    std::cout<<"{\"exhaustive\":"<<e.profiles<<",\"random\":"<<r.profiles
      <<",\"loss_observed\":"<<e.losses+r.losses<<",\"gain_observed\":"<<e.gains+r.gains
      <<",\"strict_overlap_bound\":"<<e.strict+r.strict<<",\"positive_loss_bound_attained\":"<<e.tight+r.tight<<"}\n";
}
