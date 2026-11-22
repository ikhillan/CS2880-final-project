// Verify cross-tier coupling against the exact archived within-tier orders.
#define POOLED_ONLY
#define CEILING_LIBRARY
#include "institution_ceiling.cpp"

int main(int argc,char**argv) {
    if(argc!=4){cerr<<"usage: check MARKET_BIN YEAR TIERS\n";return 2;}
    Market m(argv[1]);int year=stoi(argv[2]);
    ifstream tif(argv[3],ios::binary);
    auto tier=readv<unsigned char>(tif,m.ne);
    if(tif.peek()!=EOF)throw runtime_error("tier length mismatch");
    SimulatedPriority sim(m,tier,year,513);
    Interleave interleave(m,tier,year,513);
    vector<vector<int>> edges(m.m);
    for(int e=0;e<m.ne;++e)edges[m.prog[e]].push_back(e);
    for(int d:{1,16,64,256,1024}) {
        sim.draw(m.priority,d);
        auto archived=m.priority;
        interleave.apply(m,d);
        for(int j=0;j<m.m;++j) {
            auto group=edges[j];
            sort(group.begin(),group.end(),[&](int x,int y){return m.priority[x]<m.priority[y];});
            array<int,5> old;old.fill(-1);
            for(size_t pos=0;pos<group.size();++pos) {
                int e=group[pos],t=tier[e];
                if(m.priority[e]!=int(pos)||archived[e]<=old[t])throw runtime_error("incorrect cross-tier interleaving");
                old[t]=archived[e];
            }
        }
    }
    cout<<year<<": all "<<m.ne<<" applicant-course edges passed the within-category order check at five checkpoints.\n";
}
