// The allocator receives no initial-offer labels in the unabridged market.
#define main school_replay_entry
#include "replay.cpp"
#undef main
#include "institution.hpp"
int main(int argc,char** argv) {
    if(argc!=4){cerr<<"usage: export_programs MARKET SIDECAR PRIVATE_PROGRAM_VECTOR\n";return 2;}
    Market m(argv[1]);
    Institutional institution(m,argv[2]);
    vector<char> all(m.ne,1);
    auto result=institution.clear(all);
    ofstream f(argv[3],ios::binary);
    if(!f)throw runtime_error("private output unavailable");
    for(int i=0;i<m.n;++i) {
        for(int e=m.at[i];e<m.at[i+1];++e)
            if(institution.blocks(i,e,result)) throw runtime_error("reported contract blocks final typed allocation");
        int j=result.partner[i]<0 ? -1 : m.prog[result.partner[i]];
        f.write(reinterpret_cast<const char*>(&j),sizeof(int));
    }
    if(!f)throw runtime_error("failed writing program indices");
}
