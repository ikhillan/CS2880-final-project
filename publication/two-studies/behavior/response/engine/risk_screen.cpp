// An ex ante warning screen, independent of D1 and April labels.
#define main original_market_main
#include "replay.cpp"
#undef main
#include "institution.hpp"
#include "family_lottery.hpp"

int main(int argc,char** argv) {
    if(argc!=7) {
        cerr<<"usage: risk_screen PREFIX SIDE COURSE_FAMILY_GROUPS YEAR PRIVATE_ELIGIBLE AGGREGATE_CSV\n";
        return 2;
    }
    string prefix=argv[1];int year=stoi(argv[4]);
    if(year<2022||year>2024)throw runtime_error("Only cohorts with a random historical tie procedure");
    Market market(prefix+".bin");
    Institutional rules(market,argv[2]);
    FamilySchoolLottery lottery(market,argv[3],year,719);
    ifstream left(prefix+"-blind.mask",ios::binary), right(prefix+"-rank.mask",ios::binary);
    auto a=readv<char>(left,market.ne),b=readv<char>(right,market.ne);
    vector<int> count_a(market.n),count_b(market.n);
    rules.set_simulated();
    for(int draw=1;draw<=64;++draw) {
        lottery.draw(market.priority,draw);
        auto x=rules.clear(a).partner, y=rules.clear(b).partner;
        for(int i=0;i<market.n;++i){count_a[i]+=x[i]<0;count_b[i]+=y[i]<0;}
    }
    vector<unsigned char> flag(market.n);
    int high_a=0,high_b=0,both=0,near=0;
    for(int i=0;i<market.n;++i) {
        // .30 threshold on a grid of 64 draws. Record also both .20-.40 to expose resolution.
        high_a+=count_a[i]>=20;high_b+=count_b[i]>=20;
        flag[i]=count_a[i]>=20 && count_b[i]>=20;both+=flag[i];
        bool cut=13<=count_a[i] && count_a[i]<=25 && 13<=count_b[i] && count_b[i]<=25;
        near+=cut;
        if(cut)flag[i]|=2;
    }
    ofstream out(argv[5],ios::binary);
    out.write(reinterpret_cast<const char*>(flag.data()),flag.size());
    ofstream agg(argv[6]);
    agg<<"year,n,draws,batch,blind_at_least_30,rank_at_least_30,both_at_least_30,both_between_20_and_40\n";
    agg<<year<<','<<market.n<<",64,719,"<<high_a<<','<<high_b<<','<<both<<','<<near<<'\n';
    return out&&agg?0:1;
}
