// Independent one-by-one proposer replay for the priority-seat sensitivity.
// Holds all available returning entitlements, then fills the program's reserve, then open seats.
#define main regular_runner_main
#include "replay.cpp"
#undef main

vector<int> choose_serial(const Market& m,const vector<int>& reserve,int j,const vector<int>& offered,
                          const vector<int>& owner){
    vector<int> selected,ordinary,remaining,flagged;
    for(int e:offered){
        if(m.ret[owner[e]]==j)selected.push_back(e);
        else if(m.ses[owner[e]])flagged.push_back(e);
        else ordinary.push_back(e);
    }
    auto better=[&](int e,int f){return m.priority[e]<m.priority[f];};
    sort(flagged.begin(),flagged.end(),better);
    assert(int(selected.size())<=m.cap[j]);
    int r=min({int(flagged.size()),reserve[j],m.cap[j]-int(selected.size())});
    selected.insert(selected.end(),flagged.begin(),flagged.begin()+r);
    ordinary.insert(ordinary.end(),flagged.begin()+r,flagged.end());
    sort(ordinary.begin(),ordinary.end(),better);
    int open=min(int(ordinary.size()),m.cap[j]-int(selected.size()));
    selected.insert(selected.end(),ordinary.begin(),ordinary.begin()+open);
    return selected;
}
int main(int argc,char**argv){
    if(argc!=5)return 2;
    Market m(argv[1]);int year=stoi(argv[2]);string path=argv[4],prefix=path+"/"+to_string(year)+"-";
    ifstream reserve_stream(argv[3],ios::binary);auto reserve=readv<int>(reserve_stream,m.m);
    vector<int> owner(m.ne);for(int i=0;i<m.n;i++)fill(owner.begin()+m.at[i],owner.begin()+m.at[i+1],i);
    mt19937_64 rng(28311+year*1000);vector<uint64_t> keys(m.ne);generate(keys.begin(),keys.end(),[&]{return rng();});
    auto base=m.menu(m.blind,keys,5,"blind"),rank=m.menu(m.rank,keys,5,"rank");
    for(string name:{"full","blind","rank","only_designated","only_other"}){
        vector<char> show(m.ne);for(int e=0;e<m.ne;e++){
            bool local=name=="rank"||(name=="only_designated"&&m.ses[owner[e]])||(name=="only_other"&&!m.ses[owner[e]]);
            show[e]=name=="full"?1:(local?rank[e]:base[e]);
        }
        vector<int> next=m.at,match(m.n,-1);vector<vector<int>> held(m.m);
        deque<int> free;for(int i=m.n-1;i>=0;i--)free.push_back(i);
        while(!free.empty()){
            int i=free.front();free.pop_front();
            while(next[i]<m.at[i+1]){
                int e=next[i]++;if(!show[e])continue;
                int j=m.prog[e];auto& current=held[j];
                if(int(current.size())<m.cap[j]){current.push_back(e);match[i]=e;break;}
                auto offers=current;offers.push_back(e);auto newhold=choose_serial(m,reserve,j,offers,owner);
                if(find(newhold.begin(),newhold.end(),e)==newhold.end())continue;
                assert(newhold.size()==current.size());
                auto evicted=find_if(current.begin(),current.end(),[&](int v){return find(newhold.begin(),newhold.end(),v)==newhold.end();});
                assert(evicted!=current.end());match[owner[*evicted]]=-1;free.push_back(owner[*evicted]);
                current=std::move(newhold);match[i]=e;break;
            }
        }
        int checks=0;
        for(int i=0;i<m.n;i++){
            assert(match[i]>=0||m.ret[i]<0);
            int bound=match[i]<0?m.at[i+1]:match[i];
            for(int e=m.at[i];e<bound;e++){
                if(!show[e])continue;int j=m.prog[e];auto offers=held[j];offers.push_back(e);
                auto ch=choose_serial(m,reserve,j,offers,owner);
                assert(find(ch.begin(),ch.end(),e)==ch.end());checks++;
            }
        }
        ofstream out(prefix+name+"-reserve.edges",ios::binary);
        out.write(reinterpret_cast<char*>(match.data()),match.size()*sizeof(int));
        cout<<year<<" "<<name<<" "<<checks<<" preferred contracts rejected\n";
    }
}
