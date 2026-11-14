// Check observed labels only after constructing the menus; use fresh lottery seeds.
#define DRAW_SENSITIVITY_LIBRARY
#include "draw_sensitivity.cpp"

int main(int argc,char**argv) {
    if(argc!=7){cerr<<"usage: validation NUMERIC YEAR TIERS REGISTER LABEL_CSV TEST_CSV\n";return 2;}
    int year=stoi(argv[2]); string prefix=string(argv[1])+"/"+argv[2];
    Market m(prefix+".bin");auto recorded=m.priority;
    ifstream input(argv[3],ios::binary);
    auto tier=readv<unsigned char>(input,m.ne);
    Design design(m,prefix);SimulatedPriority train(m,tier,year,513),test(m,tier,year,515);
    vector<Policy> policies={{"blind",design.a,0,0},{"rank",design.b,0,design.available}};
    for(int d=1;d<=1024;++d) {
        train.draw(m.priority,d);design.learn(Baseline(m,design.a),1);
        if(d==16 || d==64 || d==256 || d==1024)
            policies.push_back({"cert"+to_string(d),design.menu(1),0,design.adopted(1)});
    }
    const auto final=policies.back();
    Policy random={"control1024_0",control(design,final,year,0),0,final.adopted};
    policies.push_back(random);
    ifstream reg(argv[4],ios::binary);
    auto schools=readv<int>(reg,m.m),april=readv<int>(reg,m.n);
    if(reg.peek()!=EOF || input.peek()!=EOF)throw runtime_error("input length mismatch");
    m.priority=recorded;
    auto old=Baseline(m,design.a);
    ofstream labels(argv[5]);if(!labels)throw runtime_error("output failure");
    labels<<"year,policy,stratum,applicants,placed,exits,entrants,first_choices,initial_offers_shown,linked,april_schools_shown,linked_placed\n";
    for(const auto& p:policies){
        auto after=m.da(p.menu);m.verify(after,p.menu);
        array<array<int,9>,2> result{};
        for(int i=0;i<m.n;++i){
            auto& row=result[m.official[i]>=0];bool hit=false,shown=false;
            for(int e=m.at[i];e<m.at[i+1];++e)if(p.menu[e]){
                hit|=m.prog[e]==m.official[i];shown|=april[i]>=0 && schools[m.prog[e]]==april[i];
            }
            ++row[0];row[1]+=after[i]>=0;
            row[2]+=old.partner[i]>=0 && after[i]<0; row[3]+=old.partner[i]<0 && after[i]>=0;
            row[4]+=after[i]>=0 && after[i]==m.at[i] && m.prog[after[i]]!=m.appended[i];
            row[5]+=hit;row[6]+=april[i]>=0;row[7]+=shown;row[8]+=april[i]>=0 && after[i]>=0;
        }
        for(int g=0;g<2;++g){labels<<year<<','<<p.name<<','<<(g?"offered":"no_offer");
            for(int x:result[g])labels<<','<<x; labels<<'\n';}
    }
    ofstream out(argv[6]);if(!out)throw runtime_error("output failure");
    out<<"year,split,draw,policy,excluded_mask,adopted_slots,available_slots,placed,exits,entrants,worse,missing_old_pairs,rows_with_blockers\n";
    for(int d=1;d<=256;++d){
        test.draw(m.priority,d);Baseline base(m,design.a);
        for(const auto* p:{&final,static_cast<const Policy*>(&random)})evaluate(out,m,base,*p,year,"validation",d,design.available);
    }
    return 0;
}
