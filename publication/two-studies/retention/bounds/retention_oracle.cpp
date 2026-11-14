// Compute necessary row constraints and construct hindsight seat-retaining menus.
// Include the versioned matched-course lottery and institutional DA implementation.
#define CEILING_LIBRARY
#include "institution_ceiling.cpp"
#include <set>
#include <sstream>

struct BoundMarket {
    Market& m; Institutional& rule; const Masks& menus;
    vector<int> row_of, candidates;
    vector<vector<int>> guaranteed_proposers, children;
    vector<unsigned char> connection;
    vector<vector<int>> choices, positive_sets;

    explicit BoundMarket(Market& m_, Institutional& r_, const Masks& x)
        :m(m_),rule(r_),menus(x),row_of(m.n,-1),guaranteed_proposers(m.m),children(m.n),
         connection(m.ne),choices(menus.rows.size()),positive_sets(menus.rows.size()) {
        for(int i=0;i<m.n;++i) for(auto link:rule.relatives[i]){
            children[link.older].push_back(i);
            for(int e=m.at[i];e<m.at[i+1];++e) if(x.a[e]||x.b[e])
                for(int h=m.at[link.older];h<m.at[link.older+1];++h)
                    if((x.a[h]||x.b[h]) && rule.school[m.prog[e]]==rule.school[m.prog[h]]) {
                        if(link.sibling) connection[e]|=1;
                        if(link.block) connection[e]|=2;
                    }
        }
        for(int i=0;i<m.n;++i){
            int first=-1;bool reorder=false;
            for(int e=m.at[i];e<m.at[i+1];++e) if(x.a[e]||x.b[e]){
                if(first<0) first=e;
                if(connection[e]&2)reorder=true;
            }
            if(first>=0 && x.a[first] && x.b[first] && !reorder)
                guaranteed_proposers[m.prog[first]].push_back(first);
        }
        for(int idx=0;idx<int(x.rows.size());++idx){
            const auto& row=x.rows[idx];
            row_of[row.applicant]=idx;
            for(int e=m.at[row.applicant];e<m.at[row.applicant+1];++e)
                if(x.a[e]||x.b[e])choices[idx].push_back(e);
            if(row.baseline_hit[1])continue;
            for(int e:row.proposed) if(x.hit(1,row.applicant,e)) positive_sets[idx].push_back(e);
            if(!positive_sets[idx].empty()){candidates.push_back(idx);}
        }
    }
    // More than Q-1 first-choice applicants beating i in every nonempty
    // category suffices to reject i. Exclude continuity (fallback) claimants
    // from the set of competitors to avoid counting people outside fixed slots.
    bool impossible(int e,const Institutional::Outcome& o_min,const Institutional::Outcome& o_max)const{
        if(rule.flags[e]&Institutional::Current)return false;
        int j=m.prog[e],i=rule.owner[e],cap=0;array<int,4>cats{};int ncat=0;
        for(int c=0;c<4;++c)if(rule.capacity[4*j+c]){
            cap+=rule.capacity[4*j+c];cats[ncat++]=c;
        }
        if(!cap)return true;
        if(int(guaranteed_proposers[j].size())<cap)return false;
        array<Institutional::Score,4> threshold;
        for(int t=0;t<ncat;++t)threshold[t]=rule.score(i,e,cats[t],o_max);
        int strong=0;
        for(int h:guaranteed_proposers[j]){
            if(h==e||rule.flags[h]&Institutional::Current)continue;
            bool wins=true;
            for(int t=0;t<ncat;++t)
                if(!(rule.score(rule.owner[h],h,cats[t],o_min)<threshold[t])){wins=false;break;}
            if(wins && ++strong>=cap)return true;
        }
        return false;
    }
    void update_upper(const Institutional::Outcome& base, const Institutional::Outcome& o_min,const Institutional::Outcome& o_max,vector<unsigned char>& upper)const{
      for(int k:candidates){
        if(!upper[k])continue;
        const auto& row=menus.rows[k];int i=row.applicant;
        if(base.partner[i]<0)continue;
        bool can=false;
        // A positive-hit feasible menu has all common edges plus q optional
        // edges, at least one optional edge at the April school.
        const int q=int(row.old.size());
        for(int e:choices[k]) if(!impossible(e,o_min,o_max)){
            if(menus.a[e]&&menus.b[e]){can=true;break;}
            if(q>=2||menus.hit(1,i,e)){can=true;break;}
        }
        if(!can)upper[k]=0;
      }
    }
};
struct ChangedRowCandidates {
    vector<int> rows,adds,drops;
};
ChangedRowCandidates construct(const BoundMarket& s, const Certificate& cert,const vector<unsigned char>& upper) {
    ChangedRowCandidates changes;
    for(int k:s.candidates) {
        if(!upper[k])continue;
        const auto &row=s.menus.rows[k];
        // Keep selection deterministic. Safe certificate edges have priority.
        int add=s.positive_sets[k][0],drop=row.old.back();
        for(int e:s.positive_sets[k])if(!cert.forbidden[e]){add=e;break;}
        for(auto j=row.old.rbegin();j!=row.old.rend();++j)if(!cert.required[*j]){drop=*j;break;}
        changes.rows.push_back(k);changes.adds.push_back(add);changes.drops.push_back(drop);
    }
    return changes;
}
static void bytes_out(const string& file,const vector<unsigned char>& data){
    ofstream out(file,ios::binary);
    out.write(reinterpret_cast<const char*>(data.data()),data.size());
    if(!out)throw runtime_error("write error: "+file);
}
static vector<unsigned char> bytes_in(const string& file,size_t n){
    ifstream in(file,ios::binary);auto out=readv<unsigned char>(in,n);
    if(in.peek()!=EOF)throw runtime_error("bad input size: "+file);
    return out;
}
static void merge(const string& frm,const string& out,int workers, bool intersect){
    auto path=[&](int k){return frm+"."+to_string(k);};
    ifstream first(path(0),ios::binary|ios::ate);size_t n=first.tellg();
    vector<unsigned char> m(n,intersect);
    for(int k=0;k<workers;++k){
        auto b=bytes_in(path(k),n);
        for(size_t j=0;j<n;++j)m[j]=intersect?(m[j]&&b[j]):(m[j]||b[j]);
    }
    bytes_out(out,m);
}
// Reconstruct which edits caused an exit. DA regions are traced by comparing
// the two *realized* match/proposal prefixes. An independent proof of the
// resulting policy is always obtained by clearing every design draw again.
struct UnionFind {
    vector<int> parent;
    explicit UnionFind(int n):parent(n,-1){}
    int root(int x){return parent[x]<0?x:parent[x]=root(parent[x]);}
    void join(int x,int y){x=root(x);y=root(y);if(x==y)return;if(parent[x]>parent[y])swap(x,y);parent[x]+=parent[y];parent[y]=x;}
};
bool reached(const Institutional::Outcome &o,int i,int e) {
    const int held=o.partner[i];
    return held<0 || e==held || (o.promoted[e]!=o.promoted[held]?bool(o.promoted[e]):e<held);
}
int main(int argc,char** argv) {
    if(argc>=2&&string(argv[1])=="merge"){
        if(argc!=6)throw runtime_error("merge FROM TO WORKERS and|or");
        merge(argv[2],argv[3],stoi(argv[4]),string(argv[5])=="and");return 0;
    }
    if(argc!=13){cerr<<"stage prefix year tier register side tmp worker workers start end batch\n";return 1;}
    string phase=argv[1],prefix=argv[2],dir=argv[7];int yr=stoi(argv[3]),worker=stoi(argv[8]),workers=stoi(argv[9]),end=stoi(argv[11]),batch=stoi(argv[12]);
    int start=stoi(argv[10]);Market m(prefix+".bin");Masks x(m,prefix,argv[5]);Institutional r(m,argv[6]);r.set_simulated();
    ifstream tf(argv[4],ios::binary);auto tier=readv<unsigned char>(tf,m.ne);
    BoundMarket s(m,r,x);Certificate cert(x);
    vector<unsigned char> upper(x.rows.size(),1);
    auto named=[&](const string& file){return dir+"/"+file;};
    auto shard=[&](const string& file){return named(file)+"."+to_string(worker);};
    if(phase!="train"){
        upper=bytes_in(named("upper"),x.rows.size());
        cert.required=vector<char>(m.ne);cert.forbidden=vector<char>(m.ne);
        auto required=bytes_in(named("required"),m.ne),forbidden=bytes_in(named("forbidden"),m.ne);
        copy(required.begin(),required.end(),cert.required.begin());
        copy(forbidden.begin(),forbidden.end(),cert.forbidden.begin());
    }
    if(phase=="core"){
        vector<unsigned char> core(x.rows.size(),1);
        for(int k:s.candidates){
            const auto& row=x.rows[k];
            bool feasible_add=false,feasible_drop=false;
            for(int e:s.positive_sets[k])feasible_add|=!cert.forbidden[e];
            for(int h:row.old)feasible_drop|=!cert.required[h];
            core[k]=feasible_add&&feasible_drop;
        }
        bytes_out(named("core"),core);return 0;
    }
    if(phase=="status"){
        int top=0,exact=0;
        for(int k:s.candidates){
            top+=upper[k];const auto& row=x.rows[k];
            bool add=false,drop=false;
            for(int e:s.positive_sets[k])add|=!cert.forbidden[e];
            for(int e:row.old)drop|=!cert.required[e];
            exact+=add&&drop;
        }
        cout<<yr<<','<<s.candidates.size()<<','<<top<<','<<exact<<'\n';return 0;
    }
    ChangedRowCandidates changes=construct(s,cert,upper);
    vector<unsigned char> active(x.rows.size());
    vector<unsigned char> frozen_safe(x.rows.size());
    vector<char> shown=x.a;
    if(phase!="train"){
        active=bytes_in(named("active"),x.rows.size());
        for(size_t z=0;z<changes.rows.size();++z){
            int k=changes.rows[z],e=changes.adds[z],h=changes.drops[z];
            if(!active[k])continue;
            shown[e]=1;shown[h]=0;
            frozen_safe[k]=!cert.forbidden[e]&&!cert.required[h];
        }
    }
    SimulatedPriority generator(m,tier,yr,batch);Interleave interleave(m,tier,yr,batch);
    auto o_min=Institutional::Outcome(),o_max=Institutional::Outcome();o_min.recent_sibling.assign(m.ne,0);o_max.recent_sibling=s.connection;
    // recent_sibling is boolean, independent of hypothetical preference promotion.
    for(auto& v:o_max.recent_sibling)v=(v&1)!=0;
    // Optional addback anchors an already certified (or subsequently re-certified)
    // feasible subset and changes the exchanged old course. Each variant is evaluated
    // from scratch, so pruning can never itself masquerade as a retention guarantee.
    vector<unsigned char> anchor(x.rows.size());
    const bool use_anchor=ifstream(named("anchor")).good();
    if(use_anchor && phase!="train"){
        anchor=bytes_in(named("anchor"),x.rows.size());
        ifstream risk_in(named("risk"),ios::binary);
        vector<double> loss_score=readv<double>(risk_in,m.ne);
        if(risk_in.peek()!=EOF)throw runtime_error("risk alignment");
        shown=x.a;
        for(size_t z=0;z<changes.rows.size();++z){
            int k=changes.rows[z],e=changes.adds[z],h=changes.drops[z];
            if(!active[k])continue;
            if(!anchor[k]){
                const auto& old=x.rows[k].old;
                h=*min_element(old.begin(),old.end(),[&](int lhs,int rhs){
                    return loss_score[lhs]==loss_score[rhs]?lhs>rhs:loss_score[lhs]<loss_score[rhs];});
                const auto& targets=s.positive_sets[k];
                e=*min_element(targets.begin(),targets.end(),[&](int lhs,int rhs){
                    return loss_score[lhs]==loss_score[rhs]?lhs<rhs:loss_score[lhs]<loss_score[rhs];});
            }
            shown[e]=1;shown[h]=0; frozen_safe[k]=anchor[k];
        }
    }
    if(phase=="score"){
        int added_hits=0,lost_hits=0,changed=0,edge_in=0,edge_out=0,total_baseline=0,total_final=0;
        for(int i=0;i<m.n;++i){
            int old_n=0,new_n=0; bool old_hit=false,new_hit=false,edited=false;
            for(int e=m.at[i];e<m.at[i+1];++e){
                if((shown[e]&&!x.a[e]&&!x.b[e]) || (x.a[e]&&x.b[e]&&!shown[e]))
                    throw runtime_error("union/common membership");
                if((r.flags[e]&Institutional::Current) && !shown[e])throw runtime_error("continuity removed");
                old_n+=x.a[e];new_n+=shown[e];
                old_hit|=x.a[e]&&x.hit(1,i,e);new_hit|=shown[e]&&x.hit(1,i,e);
                edge_in+=shown[e]&&!x.a[e];edge_out+=x.a[e]&&!shown[e];edited|=shown[e]!=x.a[e];
            }
            if(old_n!=new_n)throw runtime_error("row quota changed");
            total_baseline+=old_hit;total_final+=new_hit;added_hits+=new_hit&&!old_hit;
            lost_hits+=old_hit&&!new_hit;changed+=edited;
        }
        if(edge_in!=edge_out||added_hits!=changed||lost_hits)throw runtime_error("inconsistent label objective");
        ofstream f(named("final.mask"),ios::binary);
        f.write(shown.data(),shown.size());if(!f)throw runtime_error("score mask");
        cout<<yr<<','<<m.n<<','<<m.ne<<','<<changed<<','<<edge_in<<','<<edge_out<<','<<total_baseline<<','<<total_final<<','<<added_hits<<','<<lost_hits<<'\n';
        return 0;
    }
    vector<unsigned char> bad(x.rows.size());
    ofstream detail(shard("receipt"));
    for(int d=start+worker;d<=end;d+=workers){
        generator.draw(m.priority,d);interleave.apply(m,d);
        auto base=r.clear(x.a);
        if(phase=="train"){
            cert.observe(r,base);s.update_upper(base,o_min,o_max,upper);
            continue;
        }
        auto newout=r.clear(shown);int exits=0,entries=0,switches=0,details=0;
        UnionFind uf(m.n+m.m);
        const bool exact_trace=use_anchor;
        for(int i=0;i<m.n;++i){
            int a=base.partner[i],b=newout.partner[i];
            exits+=a>=0&&b<0;entries+=a<0&&b>=0;switches+=a>=0&&b>=0&&a!=b;
            bool altered=a!=b||base.category[i]!=newout.category[i];details+=altered;
            if(phase!="prune")continue;
            if(altered){
                for(int e=m.at[i];e<m.at[i+1];++e){
                    const bool old_seen=x.a[e]&&reached(base,i,e), new_seen=shown[e]&&reached(newout,i,e);
                    // Rejected proposals common to the two traces cannot explain a change
                    // by themselves. Always include distinct endpoint courses and any
                    // course whose relative family status changed.
                    if(exact_trace?(old_seen!=new_seen||a==e||b==e||base.recent_sibling[e]!=newout.recent_sibling[e]||base.promoted[e]!=newout.promoted[e]):(old_seen||new_seen))
                        uf.join(i,m.n+m.prog[e]);
                }
                if(a!=b)for(int c:s.children[i])uf.join(i,c);
            }else{
                int k=s.row_of[i];
                if(k<0||!active[k])continue;
                for(int e=m.at[i];e<m.at[i+1];++e)
                    if((shown[e]&&!x.a[e]&&reached(newout,i,e))||(!shown[e]&&x.a[e]&&reached(base,i,e)))
                        uf.join(i,m.n+m.prog[e]);
            }
        }
        if(phase=="prune"&&exits){
            vector<unsigned char> harmful(m.n+m.m);
            for(int i=0;i<m.n;++i)if(base.partner[i]>=0&&newout.partner[i]<0)harmful[uf.root(i)]=1;
            int marked=0;
            for(int k:s.candidates)if(active[k]&&!frozen_safe[k]&&harmful[uf.root(x.rows[k].applicant)]){
                bad[k]=1;++marked;
            }
            if(!marked && !use_anchor) for(int k:s.candidates)if(active[k]&&harmful[uf.root(x.rows[k].applicant)])bad[k]=1;
        }
        if(phase=="prune"&&use_anchor&&exits){
            // A rare missed path may pass through a pupil ending at the same course.
            // Broad connectivity fallback for these draws ensures progress.
            UnionFind extra(m.n+m.m);
            for(int i=0;i<m.n;++i){
                if(base.partner[i]!=newout.partner[i] || base.category[i]!=newout.category[i] || (s.row_of[i]>=0&&active[s.row_of[i]])){
                    for(int e=m.at[i];e<m.at[i+1];++e)
                        if(((x.a[e]&&reached(base,i,e))||(shown[e]&&reached(newout,i,e))))extra.join(i,m.n+m.prog[e]);
                    if(base.partner[i]!=newout.partner[i])for(int c:s.children[i])extra.join(i,c);
                }
            }
            vector<unsigned char> has(m.n+m.m),need(m.n+m.m);
            for(int k:s.candidates)if(bad[k])has[extra.root(x.rows[k].applicant)]=1;
            for(int i=0;i<m.n;++i)if(base.partner[i]>=0&&newout.partner[i]<0)
                need[extra.root(i)]=1;
            for(int k:s.candidates)if(active[k] && !anchor[k]){
               int g=extra.root(x.rows[k].applicant);
               if(need[g]&&!has[g])bad[k]=1;
            }
        }
        detail<<yr<<','<<batch<<','<<d<<','<<exits<<','<<entries<<','<<switches<<','<<details<<'\n';
        if(!detail)throw runtime_error("bad receipt");
    }
    if(phase=="train"){
        bytes_out(shard("upper"),upper);
        bytes_out(shard("required"),vector<unsigned char>(cert.required.begin(),cert.required.end()));
        bytes_out(shard("forbidden"),vector<unsigned char>(cert.forbidden.begin(),cert.forbidden.end()));
    }else bytes_out(shard("bad"),bad);
}
