// An outer bound on historical hits using DA on universally present proposal prefixes.
#define CEILING_LIBRARY
#include "institution_ceiling.cpp"
#include <set>
#include <sstream>

struct RowDesign {
    int pupil, quota;
    vector<int> common, optional, targets;
    vector<vector<int>> optional_masks;
};

struct LowerCutoffs {
    const Market &m;
    const Institutional &rule;
    const Masks &menus;
    vector<char> prefix;
    vector<unsigned char> reorders;
    Institutional::Outcome favorable;
    vector<RowDesign> designs;
    int known_prefix_rows = 0, known_prefix_edges = 0;
    LowerCutoffs(const Market &market,const Institutional &r,const Masks &x)
        : m(market),rule(r),menus(x),prefix(m.ne,0),reorders(m.n) {
        favorable.recent_sibling.assign(m.ne,0);
        for(int i=0;i<m.n;++i) {
            bool variable_priority=false;
            for(auto link:rule.relatives[i]) {
                for(int e=m.at[i];e<m.at[i+1];++e) if(x.a[e] || x.b[e])
                    for(int o=m.at[link.older];o<m.at[link.older+1];++o)
                        if((x.a[o] || x.b[o]) && x.school[m.prog[e]]==x.school[m.prog[o]]) {
                            favorable.recent_sibling[e] |= link.sibling;
                            variable_priority |= link.sibling || link.block;
                            reorders[i] |= link.block;
                        }
            }
            if(variable_priority) continue;
            bool nonempty=false;
            for(int e=m.at[i];e<m.at[i+1];++e) {
                if(x.a[e]!=x.b[e]) break;
                if(x.a[e]) {
                    prefix[e]=1;
                    ++known_prefix_edges;
                    nonempty=true;
                }
            }
            known_prefix_rows += nonempty;
        }
        for(const auto &row:x.rows) {
            if(row.baseline_hit[1])continue;
            RowDesign d;d.pupil=row.applicant;d.quota=row.old.size();
            for(int e=m.at[d.pupil];e<m.at[d.pupil+1];++e) {
                if(x.a[e] && x.b[e])d.common.push_back(e);
                if(x.a[e]!=x.b[e]) {
                    if(x.hit(1,d.pupil,e))d.targets.push_back(e);
                    d.optional.push_back(e);
                }
            }
            if(d.targets.empty())continue;
            // Represent exact displayed course sets; there are at most C(10,5)=252.
            vector<char> choose(d.optional.size());
            fill(choose.begin(),choose.begin()+d.quota,1);
            do {
                vector<int> active;
                bool hit=false;
                for(size_t j=0;j<d.optional.size();++j) if(choose[j]) {
                    int e=d.optional[j];active.push_back(e);hit |= x.hit(1,d.pupil,e);
                }
                if(hit)d.optional_masks.push_back(active);
            } while(prev_permutation(choose.begin(),choose.end()));
            designs.push_back(std::move(d));
        }
    }
    bool rejected(int i,int e,const Institutional::Outcome &ghost) const {
        if(rule.flags[e]&Institutional::Current)return false;
        int j=m.prog[e];
        for(int c=0;c<4;++c) {
            int slot=4*j+c;
            if(!rule.capacity[slot])continue;
            if(ghost.occupancy[slot]<rule.capacity[slot])return false;
            // Strictly lower than the worst holder at this program/category.
            if(!(ghost.worst[slot] < rule.score(i,e,c,favorable)))return false;
        }
        return true;
    }
};

#include "hall_filter.hpp"

int main(int argc,char **argv) {
    if(argc<11||argc>12){cerr<<"ghost PREFIX YEAR TIER REG SIDE FROM TO STEP BATCH OUTPUT [hall]\n";return 1;}
    string prefix=argv[1];int year=stoi(argv[2]),first=stoi(argv[6]),end=stoi(argv[7]),step=stoi(argv[8]),batch=stoi(argv[9]);
    Market m(prefix+".bin"); Institutional rule(m,argv[5]);rule.set_simulated();Masks x(m,prefix,argv[4]);
    ifstream ts(argv[3],ios::binary);auto tier=readv<unsigned char>(ts,m.ne);
    if(ts.peek()!=EOF)throw runtime_error("Tier dimensions");
    if(first==0) {
        ofstream mapping(argv[10]);
        for(size_t k=0;k<x.rows.size();++k) {
            const auto &row=x.rows[k];
            if(row.baseline_hit[1])continue;
            for(int e:row.proposed)if(x.hit(1,row.applicant,e)){mapping<<k<<'\n';break;}
        }
        return !mapping;
    }
    LowerCutoffs oracle(m,rule,x);
    vector<vector<unsigned char>> allowed;
    for(const auto &d:oracle.designs)allowed.emplace_back(d.optional_masks.size(),1);
    SimulatedPriority lottery(m,tier,year,batch);Interleave interleave(m,tier,year,batch);
    int ndraw=0;
    const bool veto=argc==12 && string(argv[11])=="veto";
    const bool hall=veto || (argc==12 && string(argv[11])=="hall");
    for(int s=first;s<=end;s+=step) {
        lottery.draw(m.priority,s);interleave.apply(m,s);
        auto base=rule.clear(x.a); auto ghost=rule.clear(oracle.prefix);
        unique_ptr<HallFilter> supported;
        if(hall)supported=make_unique<HallFilter>(oracle,base,ghost);
        auto possible_edge=[&](int i,int e) {
            return !oracle.rejected(i,e,ghost) && (!supported || supported->permits(oracle,i,e));
        };
        for(size_t n=0;n<oracle.designs.size();++n) {
            const auto &d=oracle.designs[n];int i=d.pupil;
            const bool incumbent=base.partner[i]>=0;
            if(!incumbent && (!veto || oracle.reorders[i]))continue;
            bool common_ok=false;
            for(int e:d.common)if(possible_edge(i,e)){common_ok=true;break;}
            if(common_ok && (!veto || oracle.reorders[i]))continue;
            for(size_t k=0;k<allowed[n].size();++k) {
                if(!allowed[n][k])continue;
                int first=m.ne,first_veto=m.ne;
                for(const auto *list:{&d.common,&d.optional_masks[k]})for(int e:*list) {
                    if(possible_edge(i,e))first=min(first,e);
                    if(veto && supported->after_target[e])first_veto=min(first_veto,e);
                }
                const bool retains=!incumbent || first<m.ne;
                const bool avoids_forced_loss=!veto || oracle.reorders[i] || first_veto==m.ne || first<first_veto;
                allowed[n][k]=retains && avoids_forced_loss;
            }
        }
        // As an extra audit check all actual holders (even pupils omitted in the relaxed
        // market) are above at least one relaxed cutoff under their best possible priority.
        if(ndraw==0)for(int i=0;i<m.n;++i)if(base.partner[i]>=0 && oracle.rejected(i,base.partner[i],ghost))
            throw runtime_error("Ghost cutoff rejected a real baseline holder");
        ++ndraw;
    }
    int possible=0,count=0;
    ofstream out(argv[10], ios::binary);
    for(auto &v:allowed) {
        possible+=find(v.begin(),v.end(),1)!=v.end();count+=v.size();
        out.write(reinterpret_cast<const char*>(v.data()),v.size());
    }
    if(!out)throw runtime_error("Output");
    cout<<year<<','<<batch<<','<<first<<','<<end<<','<<step<<','<<ndraw<<','<<oracle.known_prefix_rows<<','<<oracle.known_prefix_edges<<','<<oracle.designs.size()<<','<<possible<<','<<count<<'\n';
    ofstream sizes(string(argv[10])+".sizes");
    for(const auto &v:allowed)sizes<<v.size()<<'\n';
}
