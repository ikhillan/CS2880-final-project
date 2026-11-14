// Hold the observed same-grade reapplicant roster fixed while varying the added school.
#define RESPONSE_REPLAY_LIBRARY
#include "response_replay.cpp"

int main(int argc,char** argv){
    if(argc!=8){
        cerr<<"usage: participation_replay NUMERIC_PREFIX SIDECAR REGISTER YEAR RETURN_MASK UNPLACED_MASK OUTPUT\n";
        return 2;
    }
    const string prefix=argv[1];
    Market market(prefix+".bin");
    Institutional engine(market,argv[2]);
    Observations obs(market,engine,argv[3],"-","-");
    auto back=read_stream<unsigned char>(argv[5],market.n);
    auto never=read_stream<unsigned char>(argv[6],market.n);
    auto a=read_stream<char>(prefix+"-blind.mask",market.ne),b=read_stream<char>(prefix+"-rank.mask",market.ne);
    const int year=stoi(argv[4]);
    ofstream out(argv[7]);
    out<<"year,scenario,position,responders,reapplicants,can_add_either,adding_blind,adding_rank,"
       <<"blind_april,rank_april,blind_linked_placed,rank_linked_placed,blind_placed,rank_placed,exits,entries\n";
    for(auto candidate:{Added::NewSchool,Added::LowDemand}){
        for(auto order:{Position::Tail,Position::Original}){
            Scenario s{candidate==Added::NewSchool?"one_school":"least_congested_school",candidate,order,{1000},1};
            Pursuit blind(market,obs,a,argv[2],s),rank(market,obs,b,argv[2],s);
            const auto potential_a=blind.eligible_add,potential_b=rank.eligible_add;
            for(bool never_only:{false,true}){
                int total=0,eligible=0;
                blind.eligible_add=potential_a;rank.eligible_add=potential_b;
                for(int i=0;i<market.n;++i){
                    int lo=market.at[i],hi=market.at[i+1];
                    if(back[i]>1||never[i]>1)throw runtime_error("invalid returner input");
                    if(back[i] && (!never_only||never[i])){
                        ++total;
                        eligible+=find(potential_a.begin()+lo,potential_a.begin()+hi,1)!=potential_a.begin()+hi||
                                  find(potential_b.begin()+lo,potential_b.begin()+hi,1)!=potential_b.begin()+hi;
                    }else{
                        fill(blind.eligible_add.begin()+lo,blind.eligible_add.begin()+hi,0);
                        fill(rank.eligible_add.begin()+lo,rank.eligible_add.begin()+hi,0);
                    }
                }
                auto old=blind.run(year,0,1000),next=rank.run(year,0,1000);
                int losses=0,gains=0;
                for(int i=0;i<market.n;++i){
                    losses+=old.partner[i]>=0 && next.partner[i]<0;
                    gains+=old.partner[i]<0 && next.partner[i]>=0;
                }
                auto x=old.totals,y=next.totals;
                if(y.placed-x.placed!=gains-losses)throw runtime_error("placement totals");
                out<<year<<','<<s.name<<','<<(order==Position::Tail?"tail":"original")<<','
                   <<(never_only?"same_grade_never_offered":"same_grade");
                for(int value:{total,eligible,x.adding,y.adding,x.april,y.april,x.linked_placed,y.linked_placed,x.placed,y.placed,losses,gains})
                    out<<','<<value;
                out<<'\n';
            }
        }
    }
    return out?0:1;
}
