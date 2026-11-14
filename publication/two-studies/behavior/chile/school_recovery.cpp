// Paired school assignment with a response to the offered shortlist.
#define main original_school_main
#include "replay.cpp"
#undef main

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

vector<char> draw_quarters(int year, int rep, int n) {
    // The same high two bits are used for both policies and both response scenarios.
    const uint64_t key = mix64(202611140314ULL ^ (uint64_t(year) << 32) ^ uint64_t(rep));
    vector<char> result(n);
    for (int i = 0; i < n; ++i) result[i] = char(mix64(key + uint64_t(i)) >> 62);
    return result;
}

template<class T>
vector<T> load_vector(const string& path, size_t n) {
    ifstream f(path, ios::binary);
    auto v = readv<T>(f, n);
    if (f.peek() != EOF) throw runtime_error("unexpected bytes in " + path);
    return v;
}

struct RecoveryCounts {
    int shown_official = 0, shown_full = 0, placed = 0, first = 0, exact_full = 0, n_edges = 0;
};

void write_counts(ostream& out, const RecoveryCounts& c) {
    out << ',' << c.shown_official << ',' << c.shown_full << ',' << c.placed << ',' << c.first
        << ',' << c.exact_full << ',' << c.n_edges;
}

struct Experiment {
    Market m;
    vector<char> a, b, full;
    vector<int> full_partner, historical_edge, hidden_a, hidden_b;

    Experiment(const string& prefix): m(prefix+".bin"), a(load_vector<char>(prefix+"-blind.mask",m.ne)),
        b(load_vector<char>(prefix+"-rank.mask",m.ne)), full(m.ne,1), full_partner(m.da(full)),
        historical_edge(m.n,-1), hidden_a(m.n,-1), hidden_b(m.n,-1) {
        m.verify(full_partner, full);
        for (int i=0; i<m.n; ++i) {
            int count_a=0, count_b=0;
            for (int e=m.at[i]; e<m.at[i+1]; ++e) {
                assert((a[e]==0 || a[e]==1) && (b[e]==0 || b[e]==1));
                count_a += a[e]; count_b += b[e];
                if (m.prog[e] == m.ret[i]) assert(a[e] && b[e]);
                if (m.prog[e] == m.official[i]) {
                    assert(historical_edge[i] < 0);
                    historical_edge[i]=e;
                }
                if (!a[e] && hidden_a[i]<0) hidden_a[i]=e;
                if (!b[e] && hidden_b[i]<0) hidden_b[i]=e;
            }
            assert(count_a==min(5,m.degree[i])+int(m.ret[i]>=0) && count_a==count_b);
            assert((hidden_a[i]<0)==(hidden_b[i]<0));
        }
        for (const string policy:{"full","blind","rank"}) {
            const auto z=policy=="full"?full_partner:m.da(policy=="blind"?a:b);
            const auto archive=load_vector<int>(prefix+"-"+policy+".partners", m.n);
            for (int i=0;i<m.n;++i) assert(archive[i]==(z[i]<0?-1:m.prog[z[i]]));
        }
    }

    RecoveryCounts measure(const vector<char>& menu, const vector<int>& z) const {
        RecoveryCounts c;
        for (int i=0;i<m.n;++i) {
            c.placed += z[i]>=0;
            c.shown_official += historical_edge[i]>=0 && menu[historical_edge[i]];
            c.shown_full += full_partner[i]>=0 && menu[full_partner[i]];
            c.exact_full += z[i]>=0 && z[i]==full_partner[i];
            c.first += z[i]==m.at[i] && z[i]>=0 && m.prog[z[i]]!=m.appended[i];
        }
        c.n_edges=accumulate(menu.begin(),menu.end(),0);
        return c;
    }

    uint64_t digest(const vector<int>& z) const {
        uint64_t v=0;
        for (int i=0;i<m.n;++i) {
            const auto partner = uint64_t(z[i]<0?0:1+m.prog[z[i]]);
            v ^= mix64((uint64_t(i)<<32) | partner);
        }
        return v;
    }

    void replay(int year,const string& rule,int step,int rep,int half,const vector<char>& q,
                ostream& out,ostream& check) const {
        vector<char> shown_a=a,shown_b=b;
        int allowed=0, retrieved=0;
        for (int i=0;i<m.n;++i) {
            if (step==0 || (step<4 && (half ? 3-q[i] : q[i])>=step)) continue;
            ++allowed;
            if (hidden_a[i]<0) continue;
            ++retrieved;
            if (rule=="full") {
                fill(shown_a.begin()+m.at[i],shown_a.begin()+m.at[i+1],1);
                fill(shown_b.begin()+m.at[i],shown_b.begin()+m.at[i+1],1);
            } else {
                shown_a[hidden_a[i]]=1; shown_b[hidden_b[i]]=1;
            }
        }
        const auto za=m.da(shown_a),zb=m.da(shown_b);
        m.verify(za,shown_a); m.verify(zb,shown_b);
        const auto c_a=measure(shown_a,za),c_b=measure(shown_b,zb);
        int gain=0,loss=0,same=0,same_gain=0,same_loss=0;
        for (int i=0; i<m.n; ++i) {
            const bool unchanged=equal(shown_a.begin()+m.at[i],shown_a.begin()+m.at[i+1],shown_b.begin()+m.at[i]);
            bool g=za[i]<0 && zb[i]>=0, l=za[i]>=0 && zb[i]<0;
            gain+=g;loss+=l;same+=unchanged;same_gain+=unchanged&&g;same_loss+=unchanged&&l;
        }
        assert(gain-loss==c_b.placed-c_a.placed && c_a.n_edges==c_b.n_edges);
        if (step==4 && rule=="full") assert(same==m.n && za==full_partner && zb==full_partner);
        out<<year<<','<<rule<<','<<step*0.25<<','<<rep<<','<<half<<','<<m.n<<','<<allowed<<','<<retrieved<<','<<same;
        write_counts(out,c_a);write_counts(out,c_b);
        out<<','<<gain<<','<<loss<<','<<same_gain<<','<<same_loss<<'\n';
        if (rep==0 && half==0 && step==2)
            check<<year<<','<<rule<<','<<digest(za)<<','<<digest(zb)<<'\n';
    }
};

#ifndef ADVICE_RECOVERY_LIBRARY
int main(int argc,char** argv) {
    if (argc!=5) { cerr<<"Usage: school_recovery NUMERIC YEAR OUT PAIRS\n"; return 2; }
    const int year=stoi(argv[2]),pairs=stoi(argv[4]);
    if(year<2022 || year>2025 || pairs<2) throw runtime_error("unsupported year or pair count");
    const string prefix=string(argv[1])+"/"+argv[2], output=string(argv[3])+"/school-recovery-"+argv[2];
    Experiment e(prefix);
    ofstream out(output+".csv"),check(output+"-checks.csv");
    if(!out||!check) throw runtime_error("could not open output");
    out<<"year,response,recovery,pair,half,n,drawn_for_recovery,rows_extended,identical_effective_menus";
    for (const string& p:{"blind","rank"}) for(const string& v:{"history","full","placed","first","named","edges"}) out<<','<<p<<'_'<<v;
    out<<",gains,losses,identical_gains,identical_losses\n";
    check<<"year,response,blind_hash,rank_hash\n";
    for(const string rule:{"full","one"}) for (int step:{0,4}) e.replay(year,rule,step,-1,0,{},out,check);
    for(int rep=0;rep<pairs;++rep){
        const auto quarters=draw_quarters(year,rep,e.m.n);
        for(const string rule:{"full","one"}) for(int step:{1,2,3}) for(int half:{0,1})
            e.replay(year,rule,step,rep,half,quarters,out,check);
    }
}
#endif
