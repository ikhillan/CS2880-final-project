// Freeze both menus, run Chile's grade sequence on the full market, then measure entry grades.
#define main original_replay_main
#include "replay.cpp"
#undef main
#include "institution.hpp"
#include <map>

struct CohortCounts {
    int n=0, linked=0, official_placed=0, correct=0;
    int blind_all=0, rank_all=0, blind_linked=0, rank_linked=0;
    int blind_hits=0, rank_hits=0, gains=0, losses=0;
};
int main(int argc, char** argv) {
    if (argc != 6) {cerr << "usage: entry_grades PREFIX SIDE REGISTER RAW_PREDICTIONS CSV\n";return 2;}
    const string prefix=argv[1];
    Market m(prefix+".bin");
    Institutional engine(m,argv[2]);
    array<vector<char>,2> masks;
    array<Institutional::Outcome,2> outcomes;
    for(int k=0;k<2;++k){
        ifstream in(prefix+(k?"-rank.mask":"-blind.mask"),ios::binary);
        masks[k]=readv<char>(in,m.ne);
        outcomes[k]=engine.clear(masks[k]);
    }
    ifstream register_file(argv[3],ios::binary);
    auto school=readv<int>(register_file,m.m);
    auto enrolled=readv<int>(register_file,m.n);
    if(school!=engine.school || register_file.peek()!=EOF)throw runtime_error("Register alignment");
    ifstream predictions(argv[4],ios::binary);
    auto historical=readv<int>(predictions,m.n);
    if(predictions.peek()!=EOF)throw runtime_error("Prediction alignment");
    map<string,CohortCounts> strata;
    for(int i=0;i<m.n;++i){
        const bool linked=enrolled[i]>=0;
        bool hit[2]={false,false}, placed[2];
        for(int k=0;k<2;++k){
            placed[k]=outcomes[k].partner[i]>=0;
            for(int e=m.at[i];e<m.at[i+1];++e) {
                if(masks[k][e] && school[m.prog[e]]==enrolled[i])hit[k]=true;
                if(masks[k][e] && engine.blocks(i,e,outcomes[k]))throw runtime_error("Blocked displayed course");
            }
        }
        vector<string> groups={"all"};
        if(m.grade[i]>=-1 && m.grade[i]<=1){
            groups.push_back("entry"); groups.push_back("grade_"+to_string(m.grade[i]));
        }else groups.push_back("other");
        for(const auto& group:groups){
            auto& c=strata[group]; ++c.n; c.linked+=linked;
            c.official_placed+=m.official[i]>=0; c.correct+=historical[i]==m.official[i];
            c.blind_all+=placed[0];c.rank_all+=placed[1];
            c.blind_linked+=linked && placed[0]; c.rank_linked+=linked && placed[1];
            c.blind_hits+=hit[0];c.rank_hits+=hit[1];
            c.gains+=linked && !placed[0] && placed[1]; c.losses+=linked && placed[0] && !placed[1];
        }
    }
    ofstream out(argv[5]);
    out<<"group,applicants,linked,official_placed,exact_official,blind_placed,rank_placed,blind_linked_placed,rank_linked_placed,blind_april_hits,rank_april_hits,linked_gains,linked_losses\n";
    for(const auto& [label,c]:strata)out<<label<<','<<c.n<<','<<c.linked<<','<<c.official_placed<<','<<c.correct<<','<<c.blind_all<<','<<c.rank_all<<','<<c.blind_linked<<','<<c.rank_linked<<','<<c.blind_hits<<','<<c.rank_hits<<','<<c.gains<<','<<c.losses<<'\n';
    if(!out)throw runtime_error("Aggregate write failed");
}
