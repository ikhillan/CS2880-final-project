// Hold the frozen shortlist rules fixed and rerandomize priorities within each published tier.
#define main archived_replay
#include "replay.cpp"
#undef main
#include <array>

struct Result {
    int placed=0, visible=0, named=0;
};

Result count(const Market& m, const vector<char>& shown, const vector<int>& assigned,
             const vector<int>& full) {
    Result result;
    for(int i=0;i<m.n;++i) {
        result.placed+=assigned[i]>=0;
        result.visible+=full[i]>=0 && shown[full[i]];
        result.named+=assigned[i]>=0 && assigned[i]==full[i];
    }
    return result;
}

int main(int argc, char** argv) {
    if(argc!=5) {cerr<<"Usage: school_lotteries NUMERIC YEAR DRAWS OUT.csv < checked.tiers\n";return 2;}
    const string prefix=string(argv[1])+"/"+argv[2];
    int year=stoi(argv[2]), draws=stoi(argv[3]);
    Market m(prefix+".bin");
    vector<unsigned char> tier(m.ne);
    cin.read(reinterpret_cast<char*>(tier.data()),tier.size());
    if(!cin || cin.peek()!=EOF) throw runtime_error("tier stream must align exactly with prepared edges");
    ifstream af(prefix+"-blind.mask",ios::binary),bf(prefix+"-rank.mask",ios::binary);
    const auto a=readv<char>(af,m.ne), b=readv<char>(bf,m.ne);
    const vector<char> all(m.ne,1);
    vector<int> edge_owner(m.ne);
    vector<vector<int>> by_program(m.m);
    for(int i=0;i<m.n;++i) {
        for(int e=m.at[i];e<m.at[i+1];++e) {
            edge_owner[e]=i;
            by_program[m.prog[e]].push_back(e);
        }
    }
    // Stable sorting by original priority preserves all original coarse comparisons in the observed arm.
    for(auto& edges:by_program) {
        sort(edges.begin(),edges.end(),[&](int l,int r){return m.priority[l]<m.priority[r];});
        for(size_t k=1;k<edges.size();++k)
            if(tier[edges[k]]<tier[edges[k-1]]) throw runtime_error("input priority tiers out of order");
    }
    ofstream out(argv[4]);
    if(!out) throw runtime_error("output unavailable");
    out<<"year,lottery,draw,n,blind_placed,rank_placed,blind_full_covered,rank_full_covered,blind_named,rank_named\n";
    for(int mode=0; mode<3; ++mode) {
        const string label=mode==0?"observed":mode==1?"independent_program":"common_applicant";
        for(int r=0;r<(mode?draws:1);++r) {
            if(mode) {
                seed_seq seq{147358u,static_cast<unsigned>(year),static_cast<unsigned>(mode),static_cast<unsigned>(r)};
                mt19937_64 rng(seq);
                vector<uint64_t> values(mode==1?m.ne:m.n);
                generate(values.begin(),values.end(),[&]{return rng();});
                for(auto& edges:by_program) {
                    sort(edges.begin(),edges.end(),[&](int l,int h) {
                        if(tier[l]!=tier[h]) return tier[l]<tier[h];
                        auto x=values[mode==1?l:edge_owner[l]], y=values[mode==1?h:edge_owner[h]];
                        return x==y?l<h:x<y;
                    });
                    for(size_t k=0;k<edges.size();++k) m.priority[edges[k]]=static_cast<int>(k);
                }
            }
            auto full=m.da(all), da_a=m.da(a), da_b=m.da(b);
            m.verify(full,all);m.verify(da_a,a);m.verify(da_b,b);
            auto ca=count(m,a,da_a,full),cb=count(m,b,da_b,full);
            out<<year<<','<<label<<','<<r<<','<<m.n<<','<<ca.placed<<','<<cb.placed<<','
               <<ca.visible<<','<<cb.visible<<','<<ca.named<<','<<cb.named<<'\n';
        }
    }
}
