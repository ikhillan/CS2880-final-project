// Keep the overlap objective and holdout fixed while increasing design draws.
#define LOTTERY_DESIGN_LIBRARY
#include "lottery_design.cpp"
#include <map>
template<typename T>
void permute(vector<T>& values, mt19937_64& rng) {
    for(size_t n=values.size();n>1;--n) {
        uint64_t excess=(uint64_t(0)-uint64_t(n))%uint64_t(n),draw;
        do draw=rng();while(draw<excess);
        swap(values[n-1],values[draw%n]);
    }
}
vector<char> control(const Design& design,const Policy& certified,int year,int seed) {
    seed_seq init{82178u,unsigned(year),unsigned(seed)};
    mt19937_64 rng(init);
    map<size_t,vector<int>> allocations;
    for(const auto& row:design.rows) {
        int edits=0;
        for(int e:row.new_only) edits+=certified.menu[e];
        allocations[row.new_only.size()].push_back(edits);
    }
    for(auto& group:allocations) permute(group.second,rng);
    auto out=design.a;
    int total=0;
    for(const auto& row:design.rows) {
        auto& counts=allocations.at(row.new_only.size());
        int edits=counts.back();counts.pop_back();total+=edits;
        if(!edits) continue;
        auto old=row.old_only,added=row.new_only;
        permute(old,rng);permute(added,rng);
        for(int k=0;k<edits;++k) {out[old[k]]=0;out[added[k]]=1;}
    }
    assert(total==certified.adopted);
    return out;
}
#ifndef DRAW_SENSITIVITY_LIBRARY
int main(int argc,char**argv) {
    if(argc!=5){cerr<<"usage: sensitivity NUMERIC YEAR TIER_FILE OUTPUT\n";return 2;}
    int year=stoi(argv[2]); string prefix=string(argv[1])+"/"+argv[2];
    Market m(prefix+".bin");auto recorded=m.priority;
    ifstream input(argv[3],ios::binary);
    auto tier=readv<unsigned char>(input,m.ne);
    if(input.peek()!=EOF)throw runtime_error("unexpected tier length");
    Design design(m,prefix);
    SimulatedPriority train(m,tier,year,513),test(m,tier,year,514);
    vector<Policy> heldout,record_only;
    for(int d=1;d<=1024;++d) {
        train.draw(m.priority,d);
        design.learn(Baseline(m,design.a),1);
        if(d!=16 && d!=64 && d!=256 && d!=1024)continue;
        Policy policy{"cert"+to_string(d),design.menu(1),0,design.adopted(1)};
        heldout.push_back(policy);
        for(int seed=0;seed<8;++seed) {
            Policy random{"control"+to_string(d)+"_"+to_string(seed),
                          control(design,policy,year,seed),0,policy.adopted};
            (seed?record_only:heldout).push_back(random);
        }
    }
    ofstream out(argv[4]);if(!out)throw runtime_error("output failure");
    out<<"year,split,draw,policy,excluded_mask,adopted_slots,available_slots,placed,exits,entrants,worse,missing_old_pairs,rows_with_blockers\n";
    for(int d=1;d<=testing;++d) {
        test.draw(m.priority,d);
        Baseline base(m,design.a);
        for(const auto& p:heldout)evaluate(out,m,base,p,year,"holdout",d,design.available);
    }
    m.priority=recorded;
    Baseline base(m,design.a);
    for(const auto* group:{&heldout,&record_only})for(const auto& p:*group)
        evaluate(out,m,base,p,year,"recorded",1,design.available);
}
#endif
