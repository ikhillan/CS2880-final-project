// Separately order original edge identifiers and advance synchronous proposal waves.
// This never physically reindexes a market or calls the serial allocator for the comparison result.
#define RESPONSE_REPLAY_LIBRARY
#include "response_replay.cpp"

struct ApplicantContracts {
    vector<pair<int,int>> order;
    size_t cursor=0;
};
struct Holder {
    Institutional::Score score;
    int applicant,edge,category;
};

pair<vector<int>,vector<int>> synchronous(const Market& m,const Institutional& rules,const Observations& obs,
                                          const vector<char>& base,const Scenario& scenario,
                                          int year,int seed,int rate) {
    vector<int> partner(m.n,-1),cat(m.n,-1),demand(m.m);
    for(int e:m.prog)++demand[e];
    Institutional::Outcome family;
    family.recent_sibling.assign(m.ne,0);
    vector<vector<Holder>> pools(4*m.m);
    vector<ApplicantContracts> applicants(m.n);
    for(auto it=rules.grades.rbegin();it!=rules.grades.rend();++it){
        for(int i:*it){
            vector<int> recent,promote,school_set,listed,recovered;
            for(const auto& rel:rules.relatives[i])if(partner[rel.older]>=0){
                int sch=obs.school[m.prog[partner[rel.older]]];
                if(rel.sibling)recent.push_back(sch);
                if(rel.block)promote.push_back(sch);
            }
            for(int e=m.at[i];e<m.at[i+1];++e)if(base[e]) {
                listed.push_back(e);school_set.push_back(obs.school[m.prog[e]]);
            }
            auto contains=[](const vector<int>& values,int value){return find(values.begin(),values.end(),value)!=values.end();};
            const bool act=responds(year,seed,i,rate)&&(!scenario.screen || (scenario.screen&obs.nooffer.at(i)));
            if(act && scenario.added!=Added::None){
                for(int e=m.at[i];e<m.at[i+1];++e)if(!base[e]){
                    bool include=true;
                    if(scenario.added==Added::NewSchool||scenario.added==Added::MultiSchool||scenario.added==Added::LowDemand)include=!contains(school_set,obs.school[m.prog[e]]);
                    if(scenario.added==Added::Relisted||scenario.added==Added::RelistedNoOffer)
                        include=obs.relist[e]>0 && (scenario.added!=Added::RelistedNoOffer || obs.nooffer.at(i));
                    if(include)recovered.push_back(e);
                }
                if(scenario.added==Added::LowDemand){
                    auto ratio=[&](int e){
                        int j=m.prog[e],vac=accumulate(rules.capacity.begin()+4*j,rules.capacity.begin()+4*j+4,0);
                        return double(1+demand[j])/(1+vac);
                    };
                    stable_sort(recovered.begin(),recovered.end(),[&](int a,int b){return ratio(a)<ratio(b);});
                }
                if(scenario.added==Added::MultiSchool&&!recovered.empty()){
                    vector<int> distinct{recovered.front()};
                    if(responds(year,seed+10000,i,573)){
                        auto second=find_if(recovered.begin(),recovered.end(),[&](int e){
                            return obs.school[m.prog[e]]!=obs.school[m.prog[distinct.front()]];
                        });
                        if(second!=recovered.end())distinct.push_back(*second);
                    }
                    recovered=std::move(distinct);
                }
                if(!recovered.empty()&&(scenario.added==Added::Course||scenario.added==Added::NewSchool||scenario.added==Added::LowDemand))
                    recovered.resize(1);
            }
            if(scenario.order==Position::Tail&&(scenario.added==Added::Relisted||scenario.added==Added::RelistedNoOffer))
                stable_sort(recovered.begin(),recovered.end(),[&](int a,int b){return obs.relist[a]<obs.relist[b];});
            listed.insert(listed.end(),recovered.begin(),recovered.end());
            if(scenario.order==Position::Original)sort(listed.begin(),listed.end());
            stable_partition(listed.begin(),listed.end(),[&](int e){return contains(promote,obs.school[m.prog[e]]);});
            auto& contracts=applicants[i].order;
            for(int e:listed){
                int j=m.prog[e];
                family.recent_sibling[e]=contains(recent,obs.school[j]);
                for(int kind:rules.route(i,e,family))if(rules.capacity[4*j+kind]>0)contracts.emplace_back(e,kind);
                if(rules.flags[e]&Institutional::Current){contracts.emplace_back(e,Institutional::Fallback);break;}
            }
        }
        vector<int> free=*it;
        while(!free.empty()){
            vector<int> active,next;
            vector<char> touched(pools.size());
            for(int i:free){
                auto& a=applicants[i];
                if(a.cursor==a.order.size())continue;
                auto [edge,kind]=a.order[a.cursor++];
                if(kind==Institutional::Fallback){partner[i]=edge;cat[i]=kind;continue;}
                int id=4*m.prog[edge]+kind;
                pools[id].push_back({rules.score(i,edge,kind,family),i,edge,kind});
                if(!touched[id]){active.push_back(id);touched[id]=1;}
            }
            for(int id:active){
                auto& proposals=pools[id];
                sort(proposals.begin(),proposals.end(),[](const Holder& a,const Holder& b){return a.score<b.score;});
                int keep=min(int(proposals.size()),rules.capacity[id]);
                for(int j=0;j<int(proposals.size());++j){
                    const auto& h=proposals[j];
                    if(j<keep){partner[h.applicant]=h.edge;cat[h.applicant]=h.category;}
                    else {partner[h.applicant]=-1;cat[h.applicant]=-1;next.push_back(h.applicant);}
                }
                proposals.resize(keep);
            }
            free=std::move(next);
        }
    }
    return {partner,cat};
}
int main(int argc,char** v){
    if(argc!=8){cerr<<"usage: check_response_order PREFIX SIDE REGISTER YEAR RELISTED NOOFFER OUTPUT\n";return 2;}
    string prefix=v[1],side=v[2];int year=stoi(v[4]);
    Market market(prefix+".bin");Institutional rules(market,side);
    Observations obs(market,rules,v[3],v[5],v[6]);
    ofstream out(v[7]);out<<"year,scenario,policy,permille,seed,applicants,changed_original_course,changed_contract_category\n";
    for(string policy:{"blind","rank"}){
        auto shown=read_stream<char>(prefix+"-"+policy+".mask",market.ne);
        for(auto sample:vector<pair<Scenario,int>>{
            {{"binding",Added::None,Position::Original,{},1},0},
            {{"all_tail",Added::All,Position::Tail,{},1},1000},
            {{"one_school_original",Added::NewSchool,Position::Original,{},1},415},
            {{"one_school_tail",Added::NewSchool,Position::Tail,{},1},415},
            {{"least_congested_school_tail",Added::LowDemand,Position::Tail,{},1},415},
            {{"one_or_two_schools_tail",Added::MultiSchool,Position::Tail,{},1},415},
            {{"relisted_tail",Added::Relisted,Position::Tail,{},1},1000},
            {{"relisted_nooffer_original",Added::RelistedNoOffer,Position::Original,{},1},1000}}){
            auto [scenario,rate]=sample;
            auto explicit_result=synchronous(market,rules,obs,shown,scenario,year,2,rate);
            Pursuit transformed(market,obs,shown,side,scenario);auto result=transformed.run(year,2,rate);
            auto mismatches=[](const vector<int>& a,const vector<int>& b){return inner_product(a.begin(),a.end(),b.begin(),0,plus<int>(),not_equal_to<int>());};
            int partner_diff=mismatches(result.partner,explicit_result.first),
                category_diff=mismatches(result.category,explicit_result.second);
            out<<year<<','<<scenario.name<<','<<policy<<','<<rate<<",2,"<<market.n<<','<<partner_diff<<','<<category_diff<<'\n';
            out.flush();
            if(partner_diff||category_diff)throw runtime_error("Independent proposal waves disagree");
        }
    }
    return out?0:1;
}
