// Finite-scenario optimization and independent holdout under the tier lottery.
#define main original_replay_main
#include "replay.cpp"
#undef main
#include "simulated_priority.hpp"

using Mask = uint32_t;
constexpr int training = 16, testing = 64;
constexpr Mask full_support = (Mask(1) << training) - 1;

struct Baseline {
    vector<int> partner, count, worst;
    Baseline(const Market& m, const vector<char>& blind)
        : partner(m.da(blind)), count(m.m), worst(m.m,-1) {
        m.verify(partner,blind);
        for(int e:partner) if(e>=0) {
            int j=m.prog[e];
            ++count[j];
            worst[j]=max(worst[j],m.priority[e]);
        }
    }
    bool blocks(const Market& m,int i,int e) const {
        int j=m.prog[e];
        return (partner[i]<0 || e<partner[i]) && m.cap[j]>0 &&
               (count[j]<m.cap[j] || m.priority[e]<worst[j]);
    }
};

struct Changes { vector<int> old_only, new_only; };
struct Design {
    const Market& m;
    vector<char> a,b;
    vector<Mask> required, forbidden;
    vector<Changes> rows;
    int available=0;
    Design(const Market& market,const string& prefix) : m(market), required(m.ne),forbidden(m.ne) {
        for(string name:{"blind","rank"}) {
            ifstream f(prefix+"-"+name+".mask",ios::binary);
            (name=="blind"?a:b)=readv<char>(f,m.ne);
            if(f.peek()!=EOF) throw runtime_error("unexpected mask length");
        }
        for(int i=0;i<m.n;++i) {
            Changes row;
            for(int e=m.at[i];e<m.at[i+1];++e) {
                if(m.prog[e]==m.ret[i]) assert(a[e]&&b[e]);
                if(a[e]&&!b[e]) row.old_only.push_back(e);
                if(!a[e]&&b[e]) row.new_only.push_back(e);
            }
            assert(row.old_only.size()==row.new_only.size());
            available+=int(row.old_only.size());
            if(!row.old_only.empty()) rows.push_back(row);
        }
    }
    void learn(const Baseline& base,Mask bit) {
        for(int i=0;i<m.n;++i) {
            int p=base.partner[i];
            if(p>=0) required[p]|=bit;
            for(int e=m.at[i];e<(p<0?m.at[i+1]:p);++e)
                if(b[e] && !a[e] && base.blocks(m,i,e)) forbidden[e]|=bit;
        }
    }
    int adopted(Mask good) const {
        int score=0;
        for(const auto& row:rows) {
            int blocked=0, pinned=0;
            for(int e:row.new_only) blocked+=bool(forbidden[e]&good);
            for(int e:row.old_only) pinned+=bool(required[e]&good);
            score+=int(row.new_only.size())-max(blocked,pinned);
        }
        return score;
    }
    vector<char> menu(Mask good) const {
        auto chosen=a;
        for(const auto& row:rows) {
            vector<int> removable;
            for(int e:row.old_only) if(!(required[e]&good)) removable.push_back(e);
            for(int e:row.new_only) if(!(forbidden[e]&good) && !removable.empty()) {
                chosen[e]=1; chosen[removable.back()]=0; removable.pop_back();
            }
        }
        int score=0;
        for(int e=0;e<m.ne;++e) score+=chosen[e]&&!a[e];
        assert(score==adopted(good));
        return chosen;
    }
};
struct Policy { string name; vector<char> menu; Mask excluded; int adopted; };
void evaluate(ostream& out,const Market& m,const Baseline& base,const Policy& p,int year,
              const string& split,int draw,int available) {
    auto after=m.da(p.menu);
    m.verify(after,p.menu);
    int exits=0,enters=0,missing=0,blocking=0,placed=0,worse=0;
    for(int i=0;i<m.n;++i) {
        int old=base.partner[i],now=after[i];
        placed+=now>=0; exits+=(old>=0&&now<0); enters+=(old<0&&now>=0);
        worse+=(old>=0 && (now<0 || now>old));
        missing+=old>=0&&!p.menu[old];
        for(int e=m.at[i];e<(old<0?m.at[i+1]:old);++e) if(p.menu[e]&&base.blocks(m,i,e)) {
            ++blocking;break;
        }
    }
    if(!missing&&!blocking) assert(!worse&&!exits&&!enters);
    if(split=="design" && p.name!="rank" && !(p.excluded&(Mask(1)<<(draw-1))))
        assert(!missing&&!blocking);
    out<<year<<','<<split<<','<<draw<<','<<p.name<<','<<p.excluded<<','<<p.adopted<<','<<available
       <<','<<placed<<','<<exits<<','<<enters<<','<<worse<<','<<missing<<','<<blocking<<'\n';
}
#ifndef LOTTERY_DESIGN_LIBRARY
int main(int argc,char** argv) {
    if(argc!=5) {cerr<<"usage: lottery NUMERIC YEAR TIER_FILE OUTPUT\n";return 2;}
    int year=stoi(argv[2]); string prefix=string(argv[1])+"/"+argv[2];
    Market m(prefix+".bin");
    auto recorded=m.priority;
    ifstream data(argv[3],ios::binary);
    auto tier=readv<unsigned char>(data,m.ne);
    if(data.peek()!=EOF) throw runtime_error("unexpected tier length");
    SimulatedPriority train(m,tier,year,513),test(m,tier,year,514);
    Design design(m,prefix);
    for(int d=1;d<=training;++d) {
        train.draw(m.priority,d);
        design.learn(Baseline(m,design.a),Mask(1)<<(d-1));
    }
    vector<Policy> policies;
    Mask best=0;
    int value=design.adopted(full_support);
    for(int budget=0;budget<=2;++budget) {
        auto visit=[&](Mask exclude) {
            int current=design.adopted(full_support^exclude);
            if(current>value) {best=exclude; value=current;}
        };
        for(int i=0;i<training;++i) {
            if(budget==1) visit(Mask(1)<<i);
            if(budget==2) for(int j=i+1;j<training;++j) visit((Mask(1)<<i)|(Mask(1)<<j));
        }
        policies.push_back({"budget"+to_string(budget),design.menu(full_support^best),best,value});
    }
    policies.push_back({"rank",design.b,full_support,design.available});
    ofstream out(argv[4]);if(!out)throw runtime_error("output failure");
    out<<"year,split,draw,policy,excluded_mask,adopted_slots,available_slots,placed,exits,entrants,worse,missing_old_pairs,rows_with_blockers\n";
    for(string split:{"design","holdout","recorded"}) {
        int draws=split=="design"?training:split=="holdout"?testing:1;
        for(int d=1;d<=draws;++d) {
            if(split=="design") train.draw(m.priority,d);
            else if(split=="holdout") test.draw(m.priority,d);
            else m.priority=recorded;
            Baseline base(m,design.a);
            for(const auto& p:policies) evaluate(out,m,base,p,year,split,d,design.available);
        }
    }
}
#endif
