// A certified rule computed without observing lottery priorities or later labels.
#define main legacy_replay
#include "replay.cpp"
#undef main
#include "institution.hpp"
#include <map>
#include <set>

using StaticScore = pair<int,int>;

vector<char> read_mask(const string& path, int length) {
    ifstream in(path, ios::binary);
    auto bits = readv<char>(in, length);
    if (in.peek() != EOF) throw runtime_error("mask length mismatch");
    return bits;
}

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        cerr << "structural PREFIX INSTITUTION MASK_OUT SUMMARY YEAR prospective|oracle [REGISTER]\n";
        return 2;
    }
    const string mode = argv[6];
    const bool oracle = mode == "oracle";
    if ((!oracle && mode != "prospective") || argc != 7 + int(oracle)) return 2;
    const string prefix = argv[1];
    Market m(prefix + ".bin");
    Institutional r(m, argv[2]);
    auto a = read_mask(prefix + "-blind.mask", m.ne);
    auto b = read_mask(prefix + "-rank.mask", m.ne);
    vector<int> april;
    if (oracle) {
        ifstream stream(argv[7], ios::binary);
        if (readv<int>(stream, m.m) != r.school) throw runtime_error("school keys differ");
        april = readv<int>(stream, m.n);
        if (stream.peek() != EOF) throw runtime_error("pupil keys differ");
    }

    vector<int> load(m.m), capacity(m.m), stop(m.n,-1);
    vector<char> promoted(m.ne), sibling(m.ne), active(m.ne), ample(m.m), safe(m.n);
    vector<vector<int>> courses(m.m), parents(m.n);
    for (int i=0; i<m.n; ++i) {
        for (int e=m.at[i]; e<m.at[i+1]; ++e) if (a[e] || b[e]) {
            int j=m.prog[e]; ++load[j]; courses[j].push_back(e);
        }
        for (auto link:r.relatives[i]) {
            set<int> possible;
            for (int e=m.at[link.older]; e<m.at[link.older+1]; ++e)
                if (a[e] || b[e]) possible.insert(r.school[m.prog[e]]);
            bool interacts=false;
            for (int e=m.at[i]; e<m.at[i+1]; ++e) if ((a[e] || b[e]) && possible.count(r.school[m.prog[e]])) {
                promoted[e] |= link.block; sibling[e] |= link.sibling;
                interacts |= link.block || link.sibling;
            }
            if(interacts) parents[i].push_back(link.older);
        }
    }
    for (int j=0; j<m.m; ++j) {
        capacity[j]=accumulate(r.capacity.begin()+4*j,r.capacity.begin()+4*j+4,0);
        ample[j]=load[j]<=capacity[j];
    }
    Institutional::Outcome best,worst;
    best.recent_sibling.assign(sibling.begin(),sibling.end()); worst.recent_sibling.assign(m.ne,0);
    vector<char> guarantee(m.ne);
    for(int j=0;j<m.m;++j) for (int cat=0;cat<4;++cat) {
        const int cap=r.capacity[4*j+cat];
        if(!cap)continue;
        vector<StaticScore> priority;
        for(int e:courses[j]){
            const auto s=r.score(r.owner[e],e,cat,best);
            priority.emplace_back(s.tier,s.special);
        }
        sort(priority.begin(),priority.end());
        for(int e:courses[j]) if(a[e] && b[e]){
            const auto s=r.score(r.owner[e],e,cat,worst);
            const int possible_higher=int(upper_bound(priority.begin(),priority.end(),StaticScore{s.tier,s.special})-priority.begin())-1;
            if(possible_higher<cap)guarantee[e]=1;
        }
    }
    for(int i=0;i<m.n;++i){
        for(int e=m.at[i];e<m.at[i+1];++e) if(a[e]&&b[e]&&
            ((r.flags[e]&Institutional::Current)||ample[m.prog[e]]||guarantee[e])){stop[i]=e;safe[i]=1;break;}
        for(int e=m.at[i];e<m.at[i+1];++e) if(a[e]||b[e]){
            int j=m.prog[e];bool nonempty=capacity[j]>0 || (r.flags[e]&Institutional::Current);
            active[e]=nonempty && (stop[i]<0 || e<=stop[i] || promoted[e]);
        }
    }

    // Reverse reachability. A course node connects all of its active potential proposers.
    vector<char> frozen(m.n), visited(m.m);
    deque<int> pending;
    auto freeze=[&](int i){if(!frozen[i]){frozen[i]=1;pending.push_back(i);}};
    for(int i=0;i<m.n;++i) if(!safe[i]) freeze(i);
    while(!pending.empty()){
        int i=pending.front();pending.pop_front();
        for(int older:parents[i])freeze(older);
        for(int e=m.at[i];e<m.at[i+1];++e) if(active[e]){
            int j=m.prog[e];
            if(ample[j] || visited[j] || !capacity[j])continue;
            visited[j]=1;
            for(int f:courses[j])if(active[f])freeze(r.owner[f]);
        }
    }
    auto shown=a;
    int rows=0,edits=0,gains=0,active_rows=0,free_edits=0;
    for(int i=0;i<m.n;++i){
        vector<int> candidates;
        int keep=0;
        bool hit=false;
        for(int e=m.at[i];e<m.at[i+1];++e){
            if(oracle && a[e] && r.school[m.prog[e]]==april[i])hit=true;
            if(a[e]==b[e] || (frozen[i] && active[e]))continue;
            candidates.push_back(e);keep+=a[e];
        }
        if (oracle) {
            if(hit || !keep)continue;
            auto add=find_if(candidates.begin(),candidates.end(),[&](int e){return !a[e] && r.school[m.prog[e]]==april[i];});
            if(add==candidates.end())continue;
            auto drop=find_if(candidates.rbegin(),candidates.rend(),[&](int e){return a[e];});
            if(drop==candidates.rend())throw runtime_error("invalid quota");
            shown[*add]=1;shown[*drop]=0;++gains;
        }else{
            // Candidates occur in the family's submitted order.
            for(int e:candidates){shown[e]=keep>0;if(keep>0)--keep;}
        }
        int changed=0,operative=0;
        for(int e:candidates){
            if(shown[e]&&!a[e])++changed;
            if(shown[e]!=a[e] && active[e])++operative;
        }
        rows+=changed>0;edits+=changed;active_rows+=operative>0;
        if(!frozen[i])free_edits+=changed;
    }
    ofstream mask(argv[3],ios::binary);mask.write(shown.data(),m.ne);if(!mask)throw runtime_error("mask write");
    ofstream out(argv[4]);
    out<<"year,mode,applicants,safe,free_rows,changed_rows,added_courses,free_added_courses,changed_active_rows,oracle_gains\n";
    out<<argv[5]<<','<<mode<<','<<m.n<<','<<count(safe.begin(),safe.end(),1)<<','<<count(frozen.begin(),frozen.end(),0)<<','<<rows<<','<<edits<<','<<free_edits<<','<<active_rows<<','<<gains<<'\n';
    if(!out)throw runtime_error("summary write");
}
