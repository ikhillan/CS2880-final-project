// Exact label ceilings conditional on fixed draw-specific baseline contracts.
// Compile with -DPOOLED_ONLY to verify the prior model without the new engine.
#define main original_replay_entry
#include "replay.cpp"
#undef main
#include "simulated_priority.hpp"
#include <type_traits>
#include <memory>
#ifndef POOLED_ONLY
#include "institution.hpp"
#include "family_lottery.hpp"
#endif

struct PooledOutcome {
    vector<int> partner, occupancy, worst;
};

struct PooledRule {
    const Market& market;
    PooledOutcome clear(const vector<char>& menus) const {
        auto match=market.da(menus);
        vector<int> occupancy(market.m),worst(market.m,-1);
        for(int e:match) if(e>=0) {
            int j=market.prog[e];
            ++occupancy[j];worst[j]=max(worst[j],market.priority[e]);
        }
        return {std::move(match),std::move(occupancy),std::move(worst)};
    }
    bool blocks(int i,int e,const PooledOutcome& o) const {
        int j=market.prog[e];
        return market.cap[j]>0 && (o.partner[i]<0 || e<o.partner[i]) &&
               (o.occupancy[j]<market.cap[j] || market.priority[e]<o.worst[j]);
    }
};

struct ChangedRow {
    int applicant;
    vector<int> old, proposed;
    bool baseline_hit[2]={false,false};
};

struct Masks {
    const Market& m;
    vector<char> a,b;
    vector<int> school,april;
    vector<ChangedRow> rows;
    int shown[2]={0,0};
    Masks(const Market& market,const string& prefix,const string& regpath):m(market) {
        for(string name:{"blind","rank"}) {
            ifstream input(prefix+"-"+name+".mask",ios::binary);
            (name=="blind"?a:b)=readv<char>(input,m.ne);
            if(input.peek()!=EOF) throw runtime_error("mask length mismatch");
        }
        ifstream reg(regpath,ios::binary);
        school=readv<int>(reg,m.m);
        april=readv<int>(reg,m.n);
        if(reg.peek()!=EOF) throw runtime_error("register length mismatch");
        for(int i=0;i<m.n;++i) {
            ChangedRow row;
            row.applicant=i;
            for(int e=m.at[i];e<m.at[i+1];++e) {
                if(a[e]) for(int target=0;target<2;++target)
                    row.baseline_hit[target]|=hit(target,i,e);
                if(a[e]&&!b[e]) row.old.push_back(e);
                if(!a[e]&&b[e]) row.proposed.push_back(e);
                if(m.prog[e]==m.ret[i] && !(a[e]&&b[e]))
                    throw runtime_error("missing entitlement");
            }
            if(row.old.size()!=row.proposed.size())throw runtime_error("different quotas");
            for(int target=0;target<2;++target) shown[target]+=row.baseline_hit[target];
            if(!row.old.empty()) rows.push_back(std::move(row));
        }
    }
    bool hit(int target,int i,int e) const {
        return target==0 ? m.official[i]>=0 && m.official[i]==m.prog[e]
                         : april[i]>=0 && april[i]==school[m.prog[e]];
    }
};

// Recover a uniform raw lottery ordering over each course while preserving
// every prior within-tier pair comparison. Conditional on the five orders,
// every sequence of tier labels with the given multiplicities is equiprobable.
class Interleave {
    vector<vector<unsigned char>> labels;
    vector<array<int,5>> offsets;
    vector<vector<int>> lookup;
    int year;
    unsigned batch;
public:
    Interleave(const Market& m,const vector<unsigned char>& tier,int yr,unsigned batch_index)
        :labels(m.m),offsets(m.m),lookup(m.m),year(yr),batch(batch_index) {
        for(int e=0;e<m.ne;++e)labels[m.prog[e]].push_back(tier[e]);
        for(int j=0;j<m.m;++j) {
            auto& row=labels[j];sort(row.begin(),row.end());
            for(int t=0;t<5;++t)
                offsets[j][t]=int(lower_bound(row.begin(),row.end(),t)-row.begin());
            lookup[j].resize(row.size());
        }
    }
    void apply(Market& m,int draw) {
        seed_seq seed{357199303u,unsigned(year),unsigned(draw),batch};
        mt19937_64 rng(seed);
        for(int j=0;j<m.m;++j) {
            const auto& labels_j=labels[j];
            if(labels_j.empty())continue;
            if(labels_j.front()==labels_j.back()) {
                iota(lookup[j].begin(),lookup[j].end(),0);
                continue;
            }
            auto mixed=labels_j;
            for(size_t n=mixed.size();n>1;--n) {
                const uint64_t threshold=(uint64_t(0)-n)%n;
                uint64_t a;
                do a=rng();while(a<threshold);
                swap(mixed[n-1],mixed[a%n]);
            }
            auto next=offsets[j];
            for(int pos=0;pos<int(mixed.size());++pos)
                lookup[j][next[mixed[pos]]++]=pos;
        }
        for(int e=0;e<m.ne;++e)m.priority[e]=lookup[m.prog[e]][m.priority[e]];
    }
};

struct Certificate {
    const Masks& menus;
    vector<char> required,forbidden;
    Certificate(const Masks& data):menus(data),required(data.m.ne),forbidden(data.m.ne){}
    template<class Rule, class Outcome>
    void observe(const Rule& rule,const Outcome& outcome) {
        for(const auto& row:menus.rows) {
            int i=row.applicant;
            int partner=outcome.partner[i];
            if(partner>=0) {
                if(!menus.a[partner])throw runtime_error("DA returned hidden baseline program");
                required[partner]=true;
            }
            for(int e:row.proposed)
                if(!forbidden[e])forbidden[e]=rule.blocks(i,e,outcome);
        }
    }
#ifndef POOLED_ONLY
    // Guarantee that every applicant visits the same sequence of courses
    // through their last proposal, including everyone who exhausts their list.
    void observe_prefix(const Institutional::Outcome& outcome) {
        for(const auto& row:menus.rows) {
            int p=outcome.partner[row.applicant];
            auto reached=[&](int e) {
                if(p<0)return true;
                if(outcome.promoted[e]!=outcome.promoted[p])return bool(outcome.promoted[e]);
                return e<=p;
            };
            for(int e:row.old)required[e]|=reached(e);
            for(int e:row.proposed)forbidden[e]|=reached(e);
        }
    }
#endif
    array<vector<unsigned char>,2> summarize(ostream& csv,int year,const string& rule,
                                            const string& sample,int support) const {
        int additional[2]={0,0},adopted=0;
        array<vector<unsigned char>,2> reachable;
        for(auto& h:reachable)h.resize(menus.m.n);
        for(const auto& row:menus.rows) {
            int i=row.applicant, removable=0,eligible=0;
            for(int e:row.old)removable+=!required[e];
            bool target[2]={false,false};
            for(int e:row.proposed)if(!forbidden[e]) {
                ++eligible;
                for(int t=0;t<2;++t)target[t]|=menus.hit(t,i,e);
            }
            adopted+=min(removable,eligible);
            for(int t=0;t<2;++t) {
                reachable[t][i]=removable>0 && !row.baseline_hit[t] && target[t];
                additional[t]+=reachable[t][i];
            }
        }
        for(int t=0;t<2;++t)
            csv<<year<<','<<rule<<','<<sample<<','<<support<<','<<(t?"april_school":"initial_offer")
               <<','<<menus.shown[t]<<','<<additional[t]<<','<<adopted<<'\n';
        csv.flush();
        return reachable;
    }
};
struct OracleMenu {
    vector<char> shown;
    vector<int> adds,drops;
};
OracleMenu build_oracle(const Certificate& cert,int target) {
    const auto& menus=cert.menus;
    OracleMenu result{menus.a,{},{}};
    for(const auto& row:menus.rows) {
        if(row.baseline_hit[target])continue;
        auto drop=find_if(row.old.rbegin(),row.old.rend(),
                         [&](int e){return !cert.required[e];});
        auto add=find_if(row.proposed.begin(),row.proposed.end(),[&](int e){
            return !cert.forbidden[e] && menus.hit(target,row.applicant,e);
        });
        if(add==row.proposed.end()||drop==row.old.rend())continue;
        result.shown[*add]=1;result.shown[*drop]=0;
        result.adds.push_back(*add);result.drops.push_back(*drop);
    }
    return result;
}

template<class Rule>
void run(Rule& rule,Market& m,const Masks& menus,const vector<unsigned char>& tier,int year,
         bool institutional,ostream& csv,const string& name,const string& bits,int draws,
         const string& family_file,ostream& audit,int holdout_draws,bool exhaustive) {
    auto original_priority=m.priority;
    Certificate before(menus);
    auto actual=rule.clear(menus.a);
    before.observe(rule,actual);
    before.summarize(csv,year,name,"recorded",1);
    Certificate train(menus),prefix(menus);
    train.summarize(csv,year,name,"design",0);
    SimulatedPriority lottery(m,tier,year,513),test(m,tier,year,515);
    Interleave interleave(m,tier,year,513),interleave_test(m,tier,year,515);
#ifndef POOLED_ONLY
    unique_ptr<FamilySchoolLottery> family, family_test;
    if(family_file!="-") {
        family=make_unique<FamilySchoolLottery>(m,family_file,year,513);
        family_test=make_unique<FamilySchoolLottery>(m,family_file,year,515);
    }
#else
    if(family_file!="-")throw runtime_error("family support is absent in pooled-only build");
#endif
    auto generate=[&](bool testing,int draw) {
#ifndef POOLED_ONLY
        if(family)(testing?family_test:family)->draw(m.priority,draw);
        else
#endif
        {
            (testing?test:lottery).draw(m.priority,draw);
            if(institutional)(testing?interleave_test:interleave).apply(m,draw);
        }
#ifndef POOLED_ONLY
        if constexpr(!is_same_v<Rule,PooledRule>)rule.set_simulated();
#endif
    };
    for(int d=1;d<=draws;++d) {
        generate(false,d);
        auto baseline=rule.clear(menus.a);
        train.observe(rule,baseline);
#ifndef POOLED_ONLY
        if constexpr(!is_same_v<Rule,PooledRule>)prefix.observe_prefix(baseline);
#endif
        if(d!=16&&d!=64&&d!=256&&d!=1024&&d!=draws)continue;
        auto reached=train.summarize(csv,year,name,"design",d);
        if(institutional)prefix.summarize(csv,year,name,"proposal_prefix",d);
        if(d==draws && bits!="-") {
            ofstream file(bits,ios::binary);if(!file)throw runtime_error("cannot open private bits");
            for(auto& v:reached)file.write(reinterpret_cast<const char*>(v.data()),v.size());
        }
    }
    // Construct the April-label oracle, then rerun the full grade sequence.
    // Only the upper-bound counts above are also computed for initial offers.
    vector<int> owner(m.ne);
    for(int i=0;i<m.n;++i)fill(owner.begin()+m.at[i],owner.begin()+m.at[i+1],i);
    vector<pair<string,OracleMenu>> oracles;
    oracles.emplace_back("stable",build_oracle(train,1));
    if(institutional)oracles.emplace_back("proposal_prefix",build_oracle(prefix,1));
    for(string split:{"recorded","design_check","holdout"}) {
        int reps=split=="recorded"?1:split=="design_check"?(exhaustive?draws:min(4,draws)):holdout_draws;
        for(int z=1;z<=reps;++z){
            int d=split=="design_check"?1+((z-1)*max(0,draws-1))/max(1,reps-1):z;
            if(split=="recorded") {
                m.priority=original_priority;
#ifndef POOLED_ONLY
                if constexpr(!is_same_v<Rule,PooledRule>)rule.set_recorded();
#endif
            }else generate(split=="holdout",d);
            auto old=rule.clear(menus.a);
            for(const auto& labeled:oracles) {
            if(exhaustive && labeled.first=="proposal_prefix" && split=="design_check" && z>3 && z!=draws)continue;
                const auto& oracle=labeled.second;
                int missing=0,blocking=0;
                for(int e:oracle.drops) missing+=old.partner[owner[e]]==e;
                for(int e:oracle.adds) blocking+=rule.blocks(owner[e],e,old);
                if(split=="design_check"&&(missing||blocking))throw runtime_error("training certificate failed");
                auto newer=rule.clear(oracle.shown);
                int placed=0,loss=0,gain=0,changed=0,changed_detail=0;
                for(int i=0;i<m.n;++i) {
                    int x=old.partner[i],y=newer.partner[i];
                    placed+=y>=0;loss+=x>=0&&y<0;gain+=x<0&&y>=0;changed+=x!=y;
#ifndef POOLED_ONLY
                    if constexpr(!is_same_v<Rule,PooledRule>)changed_detail+=x!=y || old.category[i]!=newer.category[i];
#endif
                }
                if(labeled.first=="proposal_prefix" && split=="design_check" && changed_detail)
                    throw runtime_error("prefix certificate changed detailed assignment");
                audit<<year<<','<<name<<','<<labeled.first<<','<<split<<','<<d<<','<<oracle.adds.size()<<','
                     <<missing<<','<<blocking<<','<<loss<<','<<gain<<','<<changed<<','<<placed<<',';
                if(institutional) audit<<changed_detail;
                else audit<<changed;
                audit<<'\n';
            }
        }
        audit.flush();
    }
}
#ifndef CEILING_LIBRARY
int main(int argc,char**argv) {
    if(argc!=12&&argc!=13){cerr<<"usage: ceiling RULE NUMERIC YEAR TIERS REGISTER INSTITUTION_FILE OUTPUT PRIVATE_BITS DRAWS FAMILY_FILE_OR_DASH HOLDOUT_DRAWS [all-design]\n";return 2;}
    string rule=argv[1];int year=stoi(argv[3]),draws=stoi(argv[9]),holdout=stoi(argv[11]);
    bool exhaustive=argc==13;
    if(exhaustive && string(argv[12])!="all-design")throw runtime_error("unrecognized design-audit flag");
    if(draws<1||holdout<0)throw runtime_error("invalid draw count");
    string prefix=string(argv[2])+"/"+argv[3];Market m(prefix+".bin");
    ifstream tif(argv[4],ios::binary);auto tier=readv<unsigned char>(tif,m.ne);
    if(tif.peek()!=EOF)throw runtime_error("tier length mismatch");
    Masks menus(m,prefix,argv[5]);
    ofstream out(argv[7]),audit(string(argv[7])+".audit.csv");
    if(!out||!audit)throw runtime_error("cannot open output");
    out<<"year,rule,priority,support,target,present_at_baseline,additional_reachable,adoptable_slots\n";
    audit<<"year,rule,certificate,split,draw,oracle_edits,missing,blocking,exits,entries,changed_program,placed,changed_detailed\n";
    if(rule=="pooled") {
        PooledRule pooled{m};run(pooled,m,menus,tier,year,false,out,rule,argv[8],draws,argv[10],audit,holdout,exhaustive);
    } else {
#ifndef POOLED_ONLY
        Institutional inst(m,argv[6]);
        inst.set_recorded();
        run(inst,m,menus,tier,year,true,out,rule,argv[8],draws,argv[10],audit,holdout,exhaustive);
#else
        throw runtime_error("institutional rule is not in this build");
#endif
    }
}
#endif
