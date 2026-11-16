// Capacity-constrained applicant-proposing DA on unchanged reported preferences and realized priorities.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;
template<class T> vector<T> readv(ifstream& f, size_t n){
    vector<T> v(n);
    f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));
    if(!f)throw runtime_error("short input");
    return v;
}
struct Market {
    int n,m,ne;
    vector<int> at,prog,priority,cap,official,ses,grade,ret,appended,degree;
    vector<double> blind,rank;
    bool enforce_entitlements=true;
    Market(const string& path){
        ifstream f(path,ios::binary);
        auto h=readv<int>(f,3);n=h[0];m=h[1];ne=h[2];
        at=readv<int>(f,n+1);prog=readv<int>(f,ne);priority=readv<int>(f,ne);
        cap=readv<int>(f,m);official=readv<int>(f,n);ses=readv<int>(f,n);grade=readv<int>(f,n);
        ret=readv<int>(f,n);appended=readv<int>(f,n);degree=readv<int>(f,n);
        blind=readv<double>(f,ne);rank=readv<double>(f,ne);
        if(f.peek()!=EOF)throw runtime_error("extra bytes");
    }
    vector<int> da(const vector<char>& shown) const {
        vector<int> next=at,chosen(n,-1);
        vector<priority_queue<pair<int,int>>> held(m);
        for(int entrant=0;entrant<n;entrant++){
            int i=entrant;
            while(i>=0 && next[i]<at[i+1]){
                int e=next[i]++;if(!shown[e])continue;
                int j=prog[e];if(cap[j]==0)continue;
                auto& pool=held[j];
                if(int(pool.size())<cap[j]){
                    pool.emplace(priority[e],i);chosen[i]=e;break;
                }
                if(priority[e]<pool.top().first){
                    int displaced=pool.top().second;pool.pop();
                    pool.emplace(priority[e],i);chosen[i]=e;chosen[displaced]=-1;i=displaced;
                }
            }
        }
        return chosen;
    }
    void verify(const vector<int>& match,const vector<char>& shown) const {
        vector<int> count(m),worst(m,-1);
        for(int i=0;i<n;i++){
            int e=match[i];if(e<0){assert(!enforce_entitlements || ret[i]<0);continue;}
            assert(at[i]<=e&&e<at[i+1]&&shown[e]);int j=prog[e];
            count[j]++;worst[j]=max(worst[j],priority[e]);
        }
        for(int j=0;j<m;j++)assert(count[j]<=cap[j]);
        for(int i=0;i<n;i++){
            int end=match[i]<0?at[i+1]:match[i];
            for(int e=at[i];e<end;e++){
                if(!shown[e])continue;
                int j=prog[e];
                assert(count[j]==cap[j]&&priority[e]>=worst[j]);
            }
        }
    }
    vector<char> menu(const vector<double>& score,const vector<uint64_t>& ties,int k,const string& name) const {
        vector<char> show(ne,0);
        vector<int> list;
        for(int i=0;i<n;i++){
            list.clear();
            for(int e=at[i];e<at[i+1];e++){
                if(prog[e]==ret[i]||degree[i]<=k)show[e]=1;else list.push_back(e);
            }
            if(list.empty())continue;
            auto ordering=[&](int a,int b){
                if(name=="first")return a<b;
                if(name!="uniform"&&score[a]!=score[b])return score[a]>score[b];
                return ties[a]==ties[b]?a<b:ties[a]>ties[b];
            };
            nth_element(list.begin(),list.begin()+k,list.end(),ordering);
            for(int l=0;l<k;l++)show[list[l]]=1;
        }
        return show;
    }
};
struct Counts{long long n=0,placed=0,reported=0,full=0,hit=0,off=0,offhit=0,agreement=0;};
void emit(ofstream& out,const Market& a,const vector<int>& oracle,const vector<char>& menu,const vector<int>& result,
          int year,int k,const string& pol,int draw){
    vector<Counts> totals(20);
    for(int i=0;i<a.n;i++){
        int y=oracle[i],z=result[i],off=a.official[i];
        bool has_off=false;for(int e=a.at[i];e<a.at[i+1];e++) if(menu[e]&&a.prog[e]==off)has_off=true;
        for(int g:{0,a.grade[i]+3,16+(a.ses[i]!=0),a.degree[i]>k?1:18,a.degree[i]<=k && a.ret[i]<0?19:-1}){
            if(g<0)continue;
            Counts& v=totals[g];
            v.n++;v.placed+=z>=0;v.reported+=(z>=0 && a.prog[z]!=a.appended[i]);v.full+=y>=0;
            v.hit+=y>=0&&menu[y];v.off+=off>=0;v.offhit+=has_off;
            v.agreement+=(z>=0?a.prog[z]:-1)==off;
        }
    }
    for(int j=0;j<20;j++){
        // Group 1 is affected; grades -1 through 12 use indices 2 through 15.
        auto& v=totals[j];
        string group=j==0?"all":j==1?"affected":j==18?"unedited":j==19?"unedited_no_entitlement":j>=16?(j==16?"nonpriority":"priority"):"grade"+to_string(j-3);
        out<<year<<','<<k<<','<<pol<<','<<draw<<','<<group<<','<<v.n<<','<<v.placed<<','<<v.reported<<','<<v.full<<','<<v.hit<<','<<v.off<<','<<v.offhit<<','<<v.agreement<<'\n';
    }
}
int main(int argc,char** argv){
    if(argc!=4 && argc!=6)return 2;
    Market a(argv[1]);int year=stoi(argv[2]);ofstream out(argv[3]);
    if (argc==6) {
      ifstream capfile(argv[4],ios::binary);a.cap=readv<int>(capfile,a.m);assert(capfile.peek()==EOF);
      a.enforce_entitlements=string(argv[5])=="protected";
    }
    out<<"year,k,policy,draw,group,n,placed,on_reported,full_assigned,full_covered,official_assigned,official_covered,identical_official\n";
    vector<char> full(a.ne,1);auto oracle=a.da(full);a.verify(oracle,full);emit(out,a,oracle,full,oracle,year,0,"full",0);
    vector<uint64_t> keys(a.ne);vector<double> dummy;
    for(int rep=0;rep<49;rep++){
        mt19937_64 rng(28311+year*1000+rep);
        generate(keys.begin(),keys.end(),[&]{return rng();});
        vector<string> policies=rep==0?vector<string>{"first","blind","rank"}:vector<string>{"uniform"};
        for(const string& name:policies)for(int k:{1,2,3,5}){
            if(argc==6 && k!=5)continue;
            const vector<double>& s=name=="rank"?a.rank:name=="blind"?a.blind:dummy;
            const auto show=a.menu(s,keys,k,name);auto res=a.da(show);a.verify(res,show);
            emit(out,a,oracle,show,res,year,k,name,rep);
        }
    }
    return 0;
}
