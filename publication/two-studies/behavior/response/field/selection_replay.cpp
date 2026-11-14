// Re-clear a fixed, retrospective set of common responders attaining each hit bound.
#define RESPONSE_REPLAY_LIBRARY
#include "response_replay.cpp"

vector<char> could_gain_hit(const Pursuit& pursuit) {
    const auto& market=pursuit.original;
    vector<char> gain(market.n);
    for(int i=0;i<market.n;++i){
        int enrolled=pursuit.observed.april[i];
        if(enrolled<0) continue;
        bool initially_shown=false, could_show=false;
        for(int e=market.at[i];e<market.at[i+1];++e){
            if(pursuit.observed.school[market.prog[e]]!=enrolled) continue;
            initially_shown |= pursuit.display[e];
            could_show |= pursuit.eligible_add[e];
        }
        gain[i]=!initially_shown && could_show;
    }
    return gain;
}

int main(int argc, char** argv) {
    if(argc!=6){
        cerr << "usage: selection_replay PREFIX SIDECAR REGISTER YEAR OUTPUT_CSV\n";
        return 2;
    }
    const string prefix=argv[1],side=argv[2];
    const int year=stoi(argv[4]);
    Market market(prefix+".bin");
    Institutional engine(market,side);
    Observations observed(market,engine,argv[3],"-","-");
    auto blind=read_stream<char>(prefix+"-blind.mask",market.ne);
    auto rank=read_stream<char>(prefix+"-rank.mask",market.ne);
    ofstream out(argv[5]);
    out << "year,scenario,position,target_bound,responder_set,responders,eligible_union,neutral_padding,"
        << "adding_blind,adding_rank,blind_april,rank_april,blind_linked_placed,rank_linked_placed,"
        << "blind_placed,rank_placed,exits,entries\n";
    for(auto choice:{Added::NewSchool,Added::LowDemand}){
        for(auto position:{Position::Original,Position::Tail}){
            Scenario scenario{choice==Added::NewSchool?"one_school":"least_congested_school",choice,position,{1000},1};
            Pursuit a(market,observed,blind,side,scenario),b(market,observed,rank,side,scenario);
            const auto blind_add=a.eligible_add,rank_add=b.eligible_add;
            const auto blind_gain=could_gain_hit(a),rank_gain=could_gain_hit(b);
            vector<int> union_eligible,neutral;
            for(int i=0;i<market.n;++i){
                auto first=market.at[i], end=market.at[i+1];
                if(find(blind_add.begin()+first,blind_add.begin()+end,1)==blind_add.begin()+end &&
                   find(rank_add.begin()+first,rank_add.begin()+end,1)==rank_add.begin()+end)continue;
                union_eligible.push_back(i);
                if(blind_gain[i]==rank_gain[i]) neutral.push_back(i);
            }
            // Neutral padding fixes the same budget in both extremes. Order is keyed only by cohort and roster index.
            sort(neutral.begin(),neutral.end(),[&](int i,int j){
                const uint64_t base=uint64_t(year)<<48;
                return response_key(base|uint32_t(i))<response_key(base|uint32_t(j));
            });
            for(bool lower:{true,false}){
                for(int per_mille:{0,199}){
                    vector<char> respond(market.n);
                    int exclusives=0;
                    for(int i:union_eligible){
                        respond[i]=lower?(blind_gain[i]&&!rank_gain[i]):(rank_gain[i]&&!blind_gain[i]);
                        exclusives+=respond[i];
                    }
                    int padding=0;
                    if(per_mille){
                        int budget=static_cast<int>((union_eligible.size()*per_mille)/1000);
                        padding=budget-exclusives;
                        if(padding<0||size_t(padding)>neutral.size())throw runtime_error("cannot attain exact target with neutral padding");
                        for(int j=0;j<padding;++j)respond[neutral[j]]=1;
                    }
                    a.eligible_add=blind_add;b.eligible_add=rank_add;
                    for(int i=0;i<market.n;++i){
                        if(respond[i])continue;
                        fill(a.eligible_add.begin()+market.at[i],a.eligible_add.begin()+market.at[i+1],0);
                        fill(b.eligible_add.begin()+market.at[i],b.eligible_add.begin()+market.at[i+1],0);
                    }
                    const auto left=a.run(year,0,1000),right=b.run(year,0,1000);
                    int exits=0,entries=0;
                    for(int i=0;i<market.n;++i){
                        exits+=(left.partner[i]>=0&&right.partner[i]<0);
                        entries+=(left.partner[i]<0&&right.partner[i]>=0);
                    }
                    auto x=left.totals,y=right.totals;
                    if(entries-exits!=y.placed-x.placed)throw runtime_error("placement accounting");
                    out<<year<<','<<scenario.name<<','<<(position==Position::Tail?"tail":"original")<<','
                       <<(lower?"lower":"upper")<<','<<(per_mille?"padded_199":"only_exclusive")<<','
                       <<exclusives+padding<<','<<union_eligible.size()<<','<<padding;
                    for(int total:{x.adding,y.adding,x.april,y.april,x.linked_placed,y.linked_placed,x.placed,y.placed,exits,entries})out<<','<<total;
                    out<<'\n';
                }
            }
        }
    }
    if(!out)throw runtime_error("cannot write aggregate output");
}
