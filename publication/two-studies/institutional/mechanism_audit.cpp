// Recorded-order ablation ladder. Every configuration has the same frozen choices.
#define main original_replay
#include "replay.cpp"
#undef main
#include "institution.hpp"
#include <map>
struct Fit {
    int n=0,exact=0,school=0,status=0,assigned=0,stable=0,repaired=0,regressed=0;
    void add(int p,int q,int old,const Institutional& engine){
        ++n; exact+=p==q; assigned+=p>=0; status+=(p<0)==(q<0);
        school+=(p<0&&q<0)||(p>=0&&q>=0&&engine.school[p]==engine.school[q]);
        stable+=p==old;repaired+=old!=q&&p==q;regressed+=old==q&&p!=q;
    }
};
void emit_fit(ostream& out,int year,const string& design,const Market& m,const Institutional& institution,
              const vector<int>& assignment,const vector<int>& pooled){
    Fit fit;
    for(int i=0;i<m.n;i++){
        auto program=[&](int e){return e<0?-1:m.prog[e];};
        fit.add(program(assignment[i]),m.official[i],program(pooled[i]),institution);
    }
    out<<year<<','<<design<<','<<fit.n<<','<<fit.assigned<<','<<fit.exact<<','<<fit.school<<','<<fit.status
       <<','<<fit.stable<<','<<fit.repaired<<','<<fit.regressed<<'\n';
}
int main(int argc,char**argv){
    if(argc!=7){cerr<<"usage: mechanism_audit PREFIX SIDE YEAR AGREEMENT_CSV TRANSITION_CSV REGISTER\n";return 2;}
    string prefix=argv[1];int year=stoi(argv[3]); Market m(prefix+".bin");
    Institutional engine(m,argv[2]); vector<char> all(m.ne,1);
    const auto pooled=m.da(all);
    ofstream out(argv[4]);out<<"year,design,applicants,simulated_placed,correct_program_or_outside,correct_school_or_outside,correct_status,unchanged_program_vs_pooled,corrected_vs_pooled,new_errors_vs_pooled\n";
    emit_fit(out,year,"pooled",m,engine,pooled,pooled);
    auto capacities=engine.capacity;
    auto links=engine.relatives;
    engine.relatives.assign(m.n,{});
    for(int j=0;j<m.m;++j){
        int v=accumulate(capacities.begin()+4*j,capacities.begin()+4*j+4,0);
        for(int c=0;c<4;++c)engine.capacity[4*j+c]=c==0?v:0;
    }
    emit_fit(out,year,"published_total_private_fallback",m,engine,engine.clear(all).partner,pooled);
    engine.capacity=capacities;
    emit_fit(out,year,"plus_typed_seats",m,engine,engine.clear(all).partner,pooled);
    engine.relatives=links;
    for(auto& row:engine.relatives)for(auto& link:row)link.block=false;
    emit_fit(out,year,"plus_sibling_priority",m,engine,engine.clear(all).partner,pooled);
    engine.relatives=std::move(links);
    emit_fit(out,year,"plus_family_promotions",m,engine,engine.clear(all).partner,pooled);
    vector<vector<char>> menus;
    vector<Institutional::Outcome> results;
    for(string name:{"blind","rank"}){
        ifstream f(prefix+"-"+name+".mask",ios::binary);menus.push_back(readv<char>(f,m.ne));
        results.push_back(engine.clear(menus.back()));
    }
    ifstream reg(argv[6],ios::binary);
    auto school_ids=readv<int>(reg,m.m),april=readv<int>(reg,m.n);
    if(school_ids!=engine.school || reg.peek()!=EOF)throw runtime_error("register alignment mismatch");
    ofstream trans(argv[5]);trans<<"year,stratum,n,blind_placed,rank_placed,gross_exits,gross_entries,blind_coverage,rank_coverage,submitted_rank_improved,submitted_rank_declined,blind_april_shown,rank_april_shown\n";
    map<string,array<int,11>> groups;
    for(int i=0;i<m.n;++i){
        int old=results[0].partner[i],next=results[1].partner[i];
        array<int,2> cover={},april_shown={};bool changed=false;
        for(int e=m.at[i];e<m.at[i+1];e++){
            changed|=menus[0][e]!=menus[1][e];
            for(int k=0;k<2;++k){
                cover[k]|=menus[k][e]&&m.prog[e]==m.official[i];
                april_shown[k]|=menus[k][e]&&school_ids[m.prog[e]]==april[i];
            }
        }
        array<int,11> row={1,old>=0,next>=0,old>=0&&next<0,old<0&&next>=0,cover[0],cover[1],next>=0&&(old<0||next<old),old>=0&&(next<0||old<next),april_shown[0],april_shown[1]};
        vector<string> strata={"all",m.official[i]>=0?"initial_offer":"no_initial_offer",changed?"changed_menu":"identical_menu",engine.relatives[i].empty()?"no_higher_link":"higher_link"};
        if(april[i]>=0){
            strata.push_back("all_linked");
            strata.push_back(m.official[i]<0?"linked_no_offer":school_ids[m.official[i]]==april[i]?"enrolled_at_initial_offer":"enrolled_elsewhere");
        }
        for(const auto& s:strata)
            for(int k=0;k<11;++k)groups[s][k]+=row[k];
    }
    for(const auto& pair:groups){trans<<year<<','<<pair.first;for(int v:pair.second)trans<<','<<v;trans<<'\n';}
}
