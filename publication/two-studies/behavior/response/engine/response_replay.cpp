// Replay reported-course recovery under the ministry rule reconstruction.
#define main original_market_main
#include "replay.cpp"
#undef main
#include "institution.hpp"
#include <cmath>
#include <memory>

template<class T> vector<T> read_stream(const string& path, size_t count) {
    ifstream f(path,ios::binary);
    auto v=readv<T>(f,count);
    if(f.peek()!=EOF) throw runtime_error("Unexpected trailing bytes in "+path);
    return v;
}

uint64_t response_key(uint64_t x) {
    // Counter-based SplitMix64 finalizer. Cohort, repetition and roster index have separate bit fields.
    x+=0x9e3779b97f4a7c15ULL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
bool responds(int year,int rep,int i,int per_mille) {
    if(per_mille==1000) return true;
    if(per_mille==0) return false;
    const uint64_t key=(uint64_t(year)<<48) | (uint64_t(rep)<<32) | uint32_t(i);
    return (response_key(key)>>11)*0x1.0p-53 < per_mille/1000.;
}

enum class Added { None, Course, NewSchool, LowDemand, MultiSchool, All, Relisted, RelistedNoOffer };
enum class Position { Original, Tail };
struct Scenario { string name; Added added; Position order; vector<int> rates; int repeats; int screen=0; };

struct Observations {
    vector<int> school, april, relist;
    vector<unsigned char> nooffer;
    Observations(const Market& m,const Institutional& rule,const string& reg,
                 const string& reapp,const string& no){
        ifstream f(reg,ios::binary);
        school=readv<int>(f,m.m);april=readv<int>(f,m.n);
        if(f.peek()!=EOF||school!=rule.school)throw runtime_error("Register school index differs");
        if(reapp!="-")relist=read_stream<int>(reapp,m.ne);
        if(no!="-")nooffer=read_stream<unsigned char>(no,m.n);
    }
};

struct Levels {
    int n=0,placed=0,linked=0,linked_placed=0,nooffer_linked_placed=0,first=0;
    int historical=0,april=0,linked_nooffer_hit=0,added=0,adding=0;
};
struct Arm {vector<int> partner,category;Levels totals;};

template<class T> void reorder(vector<T>& values,const vector<int>& original_edge) {
    const vector<T> before=values;
    for(size_t e=0;e<values.size();++e)values[e]=before[original_edge[e]];
}

// The allocator always scans edges in increasing index within each applicant. Reindex all
// edge-level attributes together so the published clear() still performs family promotion,
// category-specific proposals, and secured continuity without behavioral modifications.
struct Pursuit {
    const Market& original;
    const Observations& observed;
    Market m;
    Institutional engine;
    const vector<char>& display;
    vector<int> original_edge;
    vector<char> eligible_add;
    int potential=0;
    Pursuit(const Market& base,const Observations& obs,const vector<char>& shown,
            const string& side,const Scenario& scenario)
        :original(base),observed(obs),m(base),engine(m,side),display(shown),
         original_edge(m.ne),eligible_add(m.ne,0) {
        iota(original_edge.begin(),original_edge.end(),0);
        vector<int> demand(m.m),vacancies(m.m);
        for(int j:m.prog)++demand[j];
        for(int j=0;j<m.m;++j)vacancies[j]=accumulate(engine.capacity.begin()+4*j,engine.capacity.begin()+4*j+4,0);
        for(int i=0;i<m.n;++i){
            int lo=m.at[i],hi=m.at[i+1],best=-1;
            vector<int> schools;
            for(int e=lo;e<hi;++e)if(shown[e])schools.push_back(obs.school[m.prog[e]]);
            for(int e=lo;e<hi;++e) {
                if(shown[e] || scenario.added==Added::None)continue;
                switch(scenario.added){
                  case Added::All:eligible_add[e]=true;break;
                  case Added::Relisted:case Added::RelistedNoOffer:
                    eligible_add[e]=obs.relist.at(e)>0 && (scenario.added==Added::Relisted || obs.nooffer.at(i));
                    break;
                  case Added::Course:if(best<0)best=e;break;
                  case Added::MultiSchool:case Added::NewSchool:case Added::LowDemand: {
                    int j=m.prog[e];
                    if(find(schools.begin(),schools.end(),obs.school[j])!=schools.end())continue;
                    if(best<0)best=e;
                    else if(scenario.added==Added::LowDemand) {
                        int k=m.prog[best];
                        if(int64_t(demand[j]+1)*(vacancies[k]+1) < int64_t(demand[k]+1)*(vacancies[j]+1))best=e;
                    }
                    break;
                  }
                  case Added::None:break;
                }
            }
            if(best>=0){
                eligible_add[best]=1;
                if(scenario.added==Added::MultiSchool){
                    for(int e=best+1;e<hi;++e){
                        int school=obs.school[m.prog[e]];
                        if(!shown[e] && school!=obs.school[m.prog[best]] &&
                           find(schools.begin(),schools.end(),school)==schools.end()){
                            eligible_add[e]=2;
                            break;
                        }
                    }
                }
            }
            if(scenario.screen && !(scenario.screen & obs.nooffer.at(i)))
                fill(eligible_add.begin()+lo,eligible_add.begin()+hi,0);
            potential+=find(eligible_add.begin()+lo,eligible_add.begin()+hi,1)!=eligible_add.begin()+hi;
            if(scenario.order==Position::Tail){
                // Preserve the order within the display. At its end use the family's register
                // reapplication order, when applicable, and otherwise their original C1 rank.
                auto after=stable_partition(original_edge.begin()+lo,original_edge.begin()+hi,
                                            [&](int e){return bool(shown[e]);});
                if(scenario.added==Added::Relisted || scenario.added==Added::RelistedNoOffer)
                    stable_sort(after,original_edge.begin()+hi,[&](int e,int f){
                        return obs.relist[e]<obs.relist[f];
                    });
            }
        }
        if(scenario.order==Position::Tail){
            reorder(m.prog,original_edge);reorder(m.priority,original_edge);
            reorder(m.blind,original_edge);reorder(m.rank,original_edge);
            reorder(engine.flags,original_edge);reorder(engine.lottery,original_edge);
            reorder(engine.pie_order,original_edge);reorder(engine.academic_order,original_edge);
        }
    }
    Arm run(int year,int rep,int per_mille){
        vector<char> active(m.ne);
        Levels l;l.n=m.n;
        for(int i=0;i<m.n;++i){
            const bool act=responds(year,rep,i,per_mille);
            bool added=false,hit=false,history=false;
            for(int e=m.at[i];e<m.at[i+1];++e){
                const int orig=original_edge[e];
                bool add=act && (eligible_add[orig]==1 || (eligible_add[orig]==2 && responds(year,rep+10000,i,573)));added|=add;l.added+=add;
                active[e]=display[orig]||add;
                if(active[e]){
                    hit|=observed.school[m.prog[e]]==observed.april[i];
                    history|=m.prog[e]==m.official[i];
                }
            }
            l.adding+=added;l.april+=hit;l.historical+=history;
            l.linked_nooffer_hit+=hit && m.official[i]<0;
        }
        auto outcome=engine.clear(active);
        if(rep==0){ // Every deterministic arm is stable conditional on its own higher-grade state.
            for(int i=0;i<m.n;++i)for(int e=m.at[i];e<m.at[i+1];++e)
                if(active[e]&&engine.blocks(i,e,outcome))throw runtime_error("Blocking displayed course");
        }
        vector<int> partner(m.n,-1);
        for(int i=0;i<m.n;++i){
            int e=outcome.partner[i];bool placed=e>=0,linked=observed.april[i]>=0;
            if(placed) {
                const int orig=original_edge[e];
                if(!active[e]||original.prog[orig]!=m.prog[e])throw runtime_error("Reindexed course changed identity");
                partner[i]=orig;
                l.first+=orig==m.at[i] && m.prog[e]!=m.appended[i];
            } else if(m.ret[i]>=0)throw runtime_error("Lost continuity");
            l.placed+=placed;l.linked+=linked;l.linked_placed+=linked && placed;
            l.nooffer_linked_placed+=linked && placed && m.official[i]<0;
        }
        return {std::move(partner),std::move(outcome.category),l};
    }
};

#ifndef RESPONSE_REPLAY_LIBRARY
int main(int argc,char** v){
    if(argc!=10){cerr<<"usage: response_replay PREFIX SIDE REGISTER YEAR SCENARIOS OUTPUT LOST_OR_DASH C2_RANK_OR_DASH NOFFER_OR_DASH\n";
        return 2;}
    string prefix=v[1],side=v[2],kind=v[5];int year=stoi(v[4]);
    Market m(prefix+".bin"); Institutional institution(m,side);
    Observations obs(m,institution,v[3],v[8],v[9]);
    auto a=read_stream<char>(prefix+"-blind.mask",m.ne),b=read_stream<char>(prefix+"-rank.mask",m.ne);
    vector<Scenario> scenarios;
    if(kind=="endpoints") {
        scenarios={{"binding",Added::None,Position::Original,{0},1},{"one_course_original",Added::Course,Position::Original,{1000},1},
                   {"one_course_tail",Added::Course,Position::Tail,{1000},1},{"all_original",Added::All,Position::Original,{1000},1},
                   {"all_tail",Added::All,Position::Tail,{1000},1}};
    } else if(kind=="field"){
        for(auto choice:{Added::NewSchool,Added::LowDemand})for(auto order:{Position::Original,Position::Tail}){
            string name=(choice==Added::NewSchool?"one_school":"least_congested_school")+string(order==Position::Original?"_original":"_tail");
            vector<int> rates=choice==Added::NewSchool?vector<int>{100,199,216,415,500,1000}:vector<int>{199,415,1000};
            scenarios.push_back({name,choice,order,rates,8});
        }
    } else if(kind=="reapplication"){
        for(auto choice:{Added::Relisted,Added::RelistedNoOffer})for(auto order:{Position::Original,Position::Tail}){
            string name=(choice==Added::Relisted?"relisted":"relisted_nooffer")+string(order==Position::Original?"_original":"_tail");
            scenarios.push_back({name,choice,order,{1000},1});
        }
    }else if(kind=="multiple"){
        for(auto order:{Position::Original,Position::Tail})
            scenarios.push_back({order==Position::Original?"one_or_two_schools_original":"one_or_two_schools_tail",
                                 Added::MultiSchool,order,{415},8});
    }else if(kind=="risk"){
        for(int screen:{1,2})for(auto order:{Position::Original,Position::Tail}){
            string name=(screen==1?"joint_risk_school":"near_cutoff_school")+string(order==Position::Original?"_original":"_tail");
            scenarios.push_back({name,Added::NewSchool,order,{199,415},8,screen});
        }
    }else throw runtime_error("Unknown set of scenarios");
    ofstream out(v[6]);
    out<<"year,scenario,permille,rep,n,linked,adding_blind,adding_rank,edges_added_blind,edges_added_rank,blind_placed,rank_placed,blind_linked_placed,rank_linked_placed,blind_april,rank_april,blind_first,rank_first,blind_offers,rank_offers,blind_nooffer_linked_placed,rank_nooffer_linked_placed,blind_nooffer_april,rank_nooffer_april,exits,entries,potential_blind,potential_rank\n";
    for(const auto& s:scenarios) {
        Pursuit blind(m,obs,a,side,s),rank(m,obs,b,side,s);
        for(int freq:s.rates){
            int repeats=(freq==0||freq==1000)?1:s.repeats;
            for(int rep=0;rep<repeats;++rep){
                Arm old=blind.run(year,rep,freq),next=rank.run(year,rep,freq);
                const auto& x=old.totals;const auto& y=next.totals;
                if(s.name=="all_original"&&(old.partner!=next.partner||old.category!=next.category))throw runtime_error("Full common reports differ");
                int lost=0,gained=0;
                vector<unsigned char> departures(m.n);
                for(int i=0;i<m.n;++i){
                    departures[i]=old.partner[i]>=0 && next.partner[i]<0;
                    lost+=departures[i];gained+=old.partner[i]<0 && next.partner[i]>=0;
                }
                if(y.placed-x.placed!=gained-lost)throw runtime_error("Accounting error");
                if(s.name=="binding" && string(v[7])!="-"){
                    ofstream bin(v[7],ios::binary);bin.write(reinterpret_cast<const char*>(departures.data()),m.n);
                    if(!bin)throw runtime_error("Failed to save optional private diagnostic");
                }
                out<<year<<','<<s.name<<','<<freq<<','<<rep;
                for(int count:{x.n,x.linked,x.adding,y.adding,x.added,y.added,x.placed,y.placed,x.linked_placed,y.linked_placed,
                               x.april,y.april,x.first,y.first,x.historical,y.historical,x.nooffer_linked_placed,y.nooffer_linked_placed,
                               x.linked_nooffer_hit,y.linked_nooffer_hit,lost,gained,blind.potential,rank.potential})out<<','<<count;
                out<<'\n';
            }
            out.flush();
        }
    }
    return out?0:1;
}
#endif
