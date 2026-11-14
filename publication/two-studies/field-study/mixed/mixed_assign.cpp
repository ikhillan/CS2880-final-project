// Estimate direct own-menu effects and assignment headcount at varied adoption levels.
#define main original_replay_main
#include "replay.cpp"
#undef main
#include "institution.hpp"
#include <random>

struct MixedCounts {
    int n=0, changed=0, a=0, b=0, a_same=0, b_same=0;
    int next=0, anti=0, next_same=0, anti_same=0;
    int next_changed_treated=0, anti_changed_treated=0;
    long long centered_next_sum=0, centered_anti_sum=0;
};

int main(int argc, char** argv) {
    if(argc!=9){cerr<<"usage: mixed_assign PREFIX SIDE REGISTER START_REP LAST_REP ENDPOINTS RAW AUDIT_SEED\n";return 2;}
    string prefix=argv[1];
    Market market(prefix+".bin");
    Institutional rules(market,argv[2]);
    vector<char> a_mask,b_mask;
    {
        ifstream la(prefix+"-blind.mask",ios::binary),lb(prefix+"-rank.mask",ios::binary);
        a_mask=readv<char>(la,market.ne); b_mask=readv<char>(lb,market.ne);
        if(la.peek()!=EOF||lb.peek()!=EOF) throw runtime_error("mask length");
    }
    ifstream reg(argv[3],ios::binary);auto school=readv<int>(reg,market.m),april=readv<int>(reg,market.n);
    if(reg.peek()!=EOF || school!=rules.school)throw runtime_error("register alignment");
    const int seed=stoi(argv[8]);
    const auto out_a=rules.clear(a_mask).partner,out_b=rules.clear(b_mask).partner;
    vector<string> strata={"all","all_linked","entry","entry_linked"};
    vector<vector<int>> groups(market.n);
    vector<MixedCounts> base(4);
    vector<int> switchers;
    vector<unsigned char> is_switch(market.n,0);
    for(int i=0;i<market.n;++i) {
        const auto lo=market.at[i],hi=market.at[i+1];
        is_switch[i]=!equal(a_mask.begin()+lo,a_mask.begin()+hi,b_mask.begin()+lo);
        if(is_switch[i]) switchers.push_back(i);
        groups[i].push_back(0);
        if(april[i]>=0) groups[i].push_back(1);
        if(market.grade[i]>=-1 && market.grade[i]<=1) {
            groups[i].push_back(2);
            if(april[i]>=0)groups[i].push_back(3);
        }
        for(int g:groups[i]){
            MixedCounts& s=base[g];++s.n;s.changed+=is_switch[i];s.a+=(out_a[i]>=0);s.b+=(out_b[i]>=0);
            if(!is_switch[i]){s.a_same+=(out_a[i]>=0);s.b_same+=(out_b[i]>=0);}
        }
    }
    ofstream endpoint(argv[6]);
    endpoint<<"group,n,changed,blind_placed,rank_placed,blind_unchanged_placed,rank_unchanged_placed\n";
    for(size_t g=0;g<strata.size();++g){const auto& s=base[g];endpoint<<strata[g]<<','<<s.n<<','<<s.changed<<','<<s.a<<','<<s.b<<','<<s.a_same<<','<<s.b_same<<'\n';}
    if(!endpoint)throw runtime_error("cannot write endpoints");
    ofstream result(argv[7]);
    result<<"pair,permille,group,population,changed,base_placed,rollout_placed,next_placed,anti_placed,base_same_placed,next_same_placed,anti_same_placed,next_treated,anti_treated,ht_next_numerator,ht_anti_numerator\n";
    for(int rep=stoi(argv[4]);rep<=stoi(argv[5]);++rep){
        seed_seq sequence{seed,2021,rep};
        mt19937_64 generator(sequence);
        vector<unsigned char> labels(market.n);
        vector<unsigned> uniforms;
        const uint64_t reject=(uint64_t(0)-uint64_t(1000))%1000;
        for(size_t j=0;j<switchers.size();++j) {
            uint64_t draw;
            do { draw=generator(); } while(draw<reject);
            uniforms.push_back(draw%1000);
        }
        for(int k:{250,450,500,550,750}){
            vector<char> x=a_mask,y=a_mask;
            for(size_t j=0;j<switchers.size();++j) {
                int i=switchers[j],lo=market.at[i],hi=market.at[i+1];
                bool z=uniforms[j]<unsigned(k),w=(999-uniforms[j])<unsigned(k);
                labels[i]=z+2*w;
                if(k==500 && z==w)throw runtime_error("non-complementary pair");
                if(z)copy(b_mask.begin()+lo,b_mask.begin()+hi,x.begin()+lo);
                if(w)copy(b_mask.begin()+lo,b_mask.begin()+hi,y.begin()+lo);
            }
            auto first=rules.clear(x).partner, second=rules.clear(y).partner;
            vector<MixedCounts> counts=base;
            for(int i=0;i<market.n;++i){
                int f=(first[i]>=0),t=(second[i]>=0),a=(out_a[i]>=0);
                int z=labels[i]&1,w=labels[i]>>1;
                for(int g:groups[i]){
                    auto& s=counts[g];s.next+=f;s.anti+=t;
                    if(!is_switch[i]){s.next_same+=f;s.anti_same+=t;continue;}
                    s.next_changed_treated+=z;s.anti_changed_treated+=w;
                    // k/1000 is the assignment probability. Multiply by
                    // 1000/(k*(1000-k)) in postprocessing to obtain the HT sum.
                    s.centered_next_sum+=(1000*z-k)*(f-a);
                    s.centered_anti_sum+=(1000*w-k)*(t-a);
                }
            }
            for(size_t g=0;g<strata.size();++g){auto& s=counts[g];
                result<<rep<<','<<k<<','<<strata[g]<<','<<s.n<<','<<s.changed<<','<<s.a<<','<<s.b<<','
                    <<s.next<<','<<s.anti<<','<<s.a_same<<','<<s.next_same<<','<<s.anti_same<<','
                    <<s.next_changed_treated<<','<<s.anti_changed_treated<<','<<s.centered_next_sum<<','<<s.centered_anti_sum<<'\n';}
        }
        result.flush();
    }
    if(!result)throw runtime_error("cannot write cohort aggregates");
}
