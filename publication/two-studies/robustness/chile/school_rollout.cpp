// Whole-cohort applicant-proposing DA under paired fractional adoption of frozen menus.
#define main original_school_replay
#include "replay.cpp"
#undef main
#include <array>
#include <cmath>

struct Score {
    int placed = 0, covered = 0, covered_recorded = 0, gains = 0, losses = 0;
    int fixed_gains = 0, fixed_losses = 0, named = 0;
};

struct Cohort {
    Market m;
    vector<int> full, official_edge;
    vector<uint64_t> tie;
    explicit Cohort(const string& path, int year)
        : m(path), official_edge(m.n,-1), tie(m.ne) {
        vector<char> all(m.ne,1);
        full = m.da(all);
        m.verify(full,all);
        for (int i=0; i<m.n; ++i)
            for (int e=m.at[i]; e<m.at[i+1]; ++e)
                if (m.prog[e] == m.official[i]) {
                    if (official_edge[i]!=-1) throw runtime_error("duplicate recorded partner");
                    official_edge[i]=e;
                }
        mt19937_64 random(28311+1000*year);
        generate(tie.begin(),tie.end(),[&]{return random();});
    }
};

struct Paired {
    const Cohort& c;
    vector<char> a,b,same;
    vector<int> initial,final;
    int n_same=0;
    Paired(const Cohort& c, int k):c(c), a(c.m.menu(c.m.blind,c.tie,k,"blind")),
        b(c.m.menu(c.m.rank,c.tie,k,"rank")), same(c.m.n) {
        initial=c.m.da(a);final=c.m.da(b);
        c.m.verify(initial,a);c.m.verify(final,b);
        for(int i=0;i<c.m.n;++i) {
            const int start=c.m.at[i], end=c.m.at[i+1];
            same[i]=equal(a.begin()+start,a.begin()+end,b.begin()+start);
            n_same+=same[i];
        }
    }
    Score score(const vector<char>& shown,const vector<int>& assignment) const {
        Score out;
        for(int i=0;i<c.m.n;++i) {
            const bool placed=assignment[i]>=0, was=initial[i]>=0;
            out.placed+=placed;
            out.covered+=c.full[i]>=0 && shown[c.full[i]];
            out.named+=placed && assignment[i]==c.full[i];
            out.covered_recorded+=c.official_edge[i]>=0 && shown[c.official_edge[i]];
            out.gains+=placed && !was;out.losses+=!placed && was;
            out.fixed_gains+=same[i] && placed && !was;out.fixed_losses+=same[i] && !placed && was;
        }
        return out;
    }
};

void check_archive(const Paired& pair, const string& prefix) {
    for (string pol:{"blind","rank","full"}) {
        const auto& result = pol=="blind"?pair.initial:pol=="rank"?pair.final:pair.c.full;
        ifstream p(prefix+"-"+pol+".partners",ios::binary);
        auto partners=readv<int>(p,pair.c.m.n);
        if(p.peek()!=EOF) throw runtime_error("unexpected partner data");
        for(int i=0;i<pair.c.m.n;++i)
            if(partners[i]!=(result[i]<0?-1:pair.c.m.prog[result[i]])) throw runtime_error("baseline partner mismatch");
        if(pol=="full") continue;
        ifstream f(prefix+"-"+pol+".mask",ios::binary);
        auto mask=readv<char>(f,pair.c.m.ne);
        if(f.peek()!=EOF || mask!=(pol=="blind"?pair.a:pair.b)) throw runtime_error("baseline menu mismatch");
    }
}

void rollout(const Paired& pair,int year,int pairs,ostream& out,ostream& own) {
    const Cohort& c=pair.c;
    const Score start=pair.score(pair.a,pair.initial),finish=pair.score(pair.b,pair.final);
    vector<char> quarter(c.m.n);
    out<<"year,pair,half,uptake,n,n_same,placed,full_covered,recorded_covered,named,gains,losses,fixed_gains,fixed_losses\n";
    own<<"year,pair,n,n_changed,direct_delta_placed\n";
    for(int rep=0;rep<pairs;++rep) {
        seed_seq seed{937557u,static_cast<unsigned>(year),static_cast<unsigned>(rep)};
        mt19937_64 random(seed);
        // The top two random bits encode the four equal-probability intervals of U.
        // 3-q selects the interval of 1-U. Complementary halves share every row draw.
        for(auto& x:quarter) x=static_cast<char>(random() >> 62);
        for(int step=0;step<=4;++step) {
            array<int,2> hits{},recorded{};
            vector<int> midpoint;
            for(int half=0;half<2;++half) {
                Score s;
                if(step==0) s=start;
                else if(step==4) s=finish;
                else {
                    vector<char> shown=pair.a;
                    for(int i=0;i<c.m.n;++i) {
                        const int q=half?3-quarter[i]:quarter[i];
                        if(!pair.same[i] && q<step)
                            copy(pair.b.begin()+c.m.at[i],pair.b.begin()+c.m.at[i+1],shown.begin()+c.m.at[i]);
                    }
                    auto assignment=c.m.da(shown);
                    c.m.verify(assignment,shown);
                    s=pair.score(shown,assignment);
                    if(step==2 && half==0) midpoint=assignment;
                    if(step==2 && half==1) {
                        int direct=0;
                        for(int i=0;i<c.m.n;++i) if(!pair.same[i]) {
                            int diff=(midpoint[i]>=0)-(assignment[i]>=0);
                            direct+=quarter[i]<2?diff:-diff;
                        }
                        own<<year<<','<<rep<<','<<c.m.n<<','<<c.m.n-pair.n_same<<','<<direct<<'\n';
                    }
                }
                assert(s.placed-start.placed==s.gains-s.losses);
                hits[half]=s.covered;recorded[half]=s.covered_recorded;
                out<<year<<','<<rep<<','<<half<<','<<step*0.25<<','<<c.m.n<<','<<pair.n_same<<','
                   <<s.placed<<','<<s.covered<<','<<s.covered_recorded<<','<<s.named<<','<<s.gains<<','
                   <<s.losses<<','<<s.fixed_gains<<','<<s.fixed_losses<<'\n';
            }
            if(step==2) {
                assert(hits[0]+hits[1]==start.covered+finish.covered);
                assert(recorded[0]+recorded[1]==start.covered_recorded+finish.covered_recorded);
            }
        }
    }
}

int main(int argc,char** argv) {
    if(argc!=5) {cerr<<"Usage: school_rollout NUMERIC YEAR OUTPUT_DIR PAIRS\n";return 2;}
    const string numeric=argv[1],year=argv[2],outdir=argv[3];
    const int y=stoi(year), pairs=stoi(argv[4]);
    if((y!=2022 && y!=2023)||pairs<2) throw runtime_error("unexpected year or replication count");
    const string src=numeric+"/"+year;
    Cohort c(src+".bin",y);
    const string prefix=outdir+"/school-"+year;
    ofstream quota(prefix+"-quota.csv"),mix(prefix+"-rollout.csv"),own(prefix+"-own.csv");
    if(!quota||!mix||!own) throw runtime_error("failed to open output files");
    quota<<"year,k,n,n_same,blind_placed,rank_placed,blind_full_covered,rank_full_covered,blind_recorded_covered,rank_recorded_covered,blind_named,rank_named,gains,losses,fixed_gains,fixed_losses\n";
    // 1,3 and 5 were used before; 8/10 check wider menus. The full graph is the common limiting display.
    for(int k:{1,3,5,8,10}) {
        Paired pair(c,k);
        Score a=pair.score(pair.a,pair.initial),b=pair.score(pair.b,pair.final);
        assert(b.placed-a.placed==b.gains-b.losses);
        quota<<year<<','<<k<<','<<c.m.n<<','<<pair.n_same<<','<<a.placed<<','<<b.placed<<','<<a.covered<<','<<b.covered<<','
             <<a.covered_recorded<<','<<b.covered_recorded<<','<<a.named<<','<<b.named<<','<<b.gains<<','<<b.losses<<','
             <<b.fixed_gains<<','<<b.fixed_losses<<'\n';
        if(k==5) {
            check_archive(pair,src);
            rollout(pair,y,pairs,mix,own);
        }
    }
}
