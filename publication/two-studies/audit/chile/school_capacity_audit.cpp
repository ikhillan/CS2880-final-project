// Replay a specified capacity vector under archived and recalculated list scores.
// Compile beside the archived solver with -I publication/research/observed/school.
#define main ordinary_replay_main
#include "replay.cpp"
#undef main
#include <array>
#include <cmath>

int main(int argc, char** argv) {
    if (argc != 6) return 2;
    const string prefix=argv[1], capfile=argv[2], scorefile=argv[3], outpath=argv[5];
    const int year=stoi(argv[4]);
    Market m(prefix+".bin");
    ifstream newcap(capfile,ios::binary), scores(scorefile,ios::binary);
    auto capacity=readv<int>(newcap,m.m);
    auto newblind=readv<double>(scores,m.ne);
    auto newrank=readv<double>(scores,m.ne);
    assert(newcap.peek()==EOF && scores.peek()==EOF);
    const int extra=accumulate(capacity.begin(),capacity.end(),0)-accumulate(m.cap.begin(),m.cap.end(),0);
    mt19937_64 rng(28311+1000*year);
    vector<uint64_t> tie(m.ne);
    generate(tie.begin(),tie.end(),[&]{return rng();});
    vector<char> full(m.ne,1), blind=m.menu(m.blind,tie,5,"blind"), rank=m.menu(m.rank,tie,5,"rank");
    ofstream out(outpath);
    out << "year,scenario,n,total_seats,added_seats,full_matched,full_exact,full_exact_given_offer,officially_offered,edited_vs_each_other,blind_rows_vs_base,rank_rows_vs_base,hist_delta,full_shown_delta,placed_blind,placed_rank,placed_delta,first_delta,gains,losses\n";
    for(string scenario:{"baseline","sum_frozen_menus","sum_rescored_menus"}) {
        if(scenario!="baseline") m.cap=capacity;
        auto left=scenario=="sum_rescored_menus"?m.menu(newblind,tie,5,"blind"):blind;
        auto right=scenario=="sum_rescored_menus"?m.menu(newrank,tie,5,"rank"):rank;
        auto f=m.da(full), a=m.da(left), b=m.da(right);
        m.verify(f,full);m.verify(a,left);m.verify(b,right);
        if(scenario=="baseline"){
            for(string name:{"full","blind","rank"}) {
                ifstream file(prefix+"-"+name+".partners",ios::binary);
                auto recorded=readv<int>(file,m.n);
                const auto& solution=name=="full"?f:name=="blind"?a:b;
                for(int i=0;i<m.n;++i) assert(recorded[i]==(solution[i]>=0?m.prog[solution[i]]:-1));
                ifstream masks(prefix+"-"+name+".mask",ios::binary);
                assert(readv<char>(masks,m.ne)==(name=="full"?full:name=="blind"?left:right));
            }
        }
        int n_full=0, samefull=0, given=0, exactgiven=0, difrows=0, arows=0,brows=0;
        int hist=0, covered=0, na=0, nb=0, first=0,gain=0,loss=0;
        for(int i=0;i<m.n;++i) {
            const auto low=m.at[i],high=m.at[i+1];
            difrows+=!equal(left.begin()+low,left.begin()+high,right.begin()+low);
            arows+=!equal(left.begin()+low,left.begin()+high,blind.begin()+low);
            brows+=!equal(right.begin()+low,right.begin()+high,rank.begin()+low);
            n_full+=(f[i]>=0);
            const bool exact=(f[i]>=0?m.prog[f[i]]:-1)==m.official[i];
            samefull+=exact;given+=m.official[i]>=0;exactgiven+=exact&&m.official[i]>=0;
            for(int e=low;e<high;++e)
                if(m.prog[e]==m.official[i]) hist+=right[e]-left[e];
            if(f[i]>=0) covered+=right[f[i]]-left[f[i]];
            na+=a[i]>=0;nb+=b[i]>=0;gain+=(a[i]<0 && b[i]>=0);loss+=(a[i]>=0 && b[i]<0);
            if(low<high && m.prog[low]!=m.appended[i]) first+=int(b[i]==low)-int(a[i]==low);
        }
        assert(nb-na==gain-loss);
        out<<year<<','<<scenario<<','<<m.n<<','<<accumulate(m.cap.begin(),m.cap.end(),0)<<','
           <<(scenario=="baseline"?0:extra)<<','<<n_full<<','<<samefull<<','<<exactgiven<<','<<given<<','
           <<difrows<<','<<arows<<','<<brows<<','<<hist<<','<<covered<<','<<na<<','<<nb<<','<<nb-na<<','
           <<first<<','<<gain<<','<<loss<<'\n';
    }
}
