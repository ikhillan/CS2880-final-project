// Form each menu from full-list match frequencies in independent fresh lotteries.
// Current year D1 assignments and realized within-tier orders enter only evaluation.
#define main original_school_replay
#include "replay.cpp"
#undef main
#include "simulated_priority.hpp"

struct Diagnostics {
    int long_rows=0, cut_ties=0, zero_cut_ties=0, selected_zero=0, nonzero_dropped=0;
};

vector<char> shortlist(const Market& m, const vector<int>& counts, Diagnostics& d) {
    vector<char> shown(m.ne,0);
    for(int i=0;i<m.n;++i) {
        vector<int> eligible;
        for(int e=m.at[i];e<m.at[i+1];++e) {
            if(m.prog[e]==m.ret[i]) shown[e]=1;
            else eligible.push_back(e);
        }
        // Original edge order is the applicant's preference order.
        stable_sort(eligible.begin(),eligible.end(),[&](int a,int b){return counts[a]>counts[b];});
        if(eligible.size()>5) {
            ++d.long_rows;
            if(counts[eligible[4]]==counts[eligible[5]]) {
                ++d.cut_ties;
                if(counts[eligible[4]]==0) ++d.zero_cut_ties;
            }
            for(size_t k=0;k<eligible.size();++k) {
                if(k<5) d.selected_zero+=counts[eligible[k]]==0;
                else d.nonzero_dropped+=counts[eligible[k]]>0;
            }
        }
        for(size_t k=0;k<min<size_t>(5,eligible.size());++k) shown[eligible[k]]=1;
    }
    return shown;
}

void evaluate(ostream& out, const Market& m, int year, const string& name,
              int draws, const Diagnostics& diag, const vector<int>& full, const vector<char>& menu) {
    auto result=m.da(menu);
    m.verify(result,menu);
    int placed=0, historical=0, covered=0, named=0, first=0;
    for(int i=0;i<m.n;++i) {
        placed+=result[i]>=0;
        covered+=full[i]>=0 && menu[full[i]];
        named+=full[i]>=0 && result[i]==full[i];
        first+=result[i]>=0 && result[i]==m.at[i] && m.prog[result[i]]!=m.appended[i];
        int shown=0;
        for(int e=m.at[i];e<m.at[i+1];++e) if(menu[e]) {
            historical+=m.prog[e]==m.official[i];
            shown++;
        }
        assert(shown==min(5,m.degree[i])+(m.ret[i]>=0));
    }
    out<<year<<','<<name<<','<<draws<<','<<m.n<<','<<placed<<','<<historical<<','<<covered<<','
       <<named<<','<<first<<','<<diag.long_rows<<','<<diag.cut_ties<<','<<diag.zero_cut_ties<<','
       <<diag.selected_zero<<','<<diag.nonzero_dropped<<'\n';
}

int main(int argc,char** argv) {
    if(argc!=5) {cerr<<"usage: simulated_menu NUMERIC YEAR OUTPUT.csv BATCH < public-tiers\n";return 2;}
    const int year=stoi(argv[2]);
    const string prefix=string(argv[1])+"/"+argv[2];
    Market m(prefix+".bin");
    vector<unsigned char> tier(m.ne);
    cin.read(reinterpret_cast<char*>(tier.data()),m.ne);
    if(!cin || cin.peek()!=EOF) throw runtime_error("tier stream length mismatch");
    const auto priority=m.priority;
    SimulatedPriority simulation(m,tier,year,stoul(argv[4]));
    vector<int> counts(m.ne);
    vector<char> all(m.ne,1);
    vector<vector<char>> menus;
    vector<Diagnostics> diagnoses;
    const array<int,3> checkpoints={16,32,64};
    for(int d=1;d<=64;++d) {
        simulation.draw(m.priority,d);
        auto outcome=m.da(all);
        m.verify(outcome,all);
        for(int e:outcome) if(e>=0) ++counts[e];
        if(find(checkpoints.begin(),checkpoints.end(),d)!=checkpoints.end()) {
            diagnoses.emplace_back();
            menus.push_back(shortlist(m,counts,diagnoses.back()));
        }
    }
    // Only now restore the realized priority order and evaluate each saved menu.
    m.priority=priority;
    auto full=m.da(all);
    m.verify(full,all);
    ofstream out(argv[3]);
    if(!out) throw runtime_error("cannot open output");
    out<<"year,policy,lottery_draws,applicants,placed,historical_shown,full_shown,full_named,submitted_first,long_rows,cut_ties,zero_cut_ties,selected_zero_long_edges,nonzero_dropped_long_edges\n";
    for(string pol:{"blind","rank"}) {
        ifstream file(prefix+"-"+pol+".mask",ios::binary);
        auto mask=readv<char>(file,m.ne);
        if(file.peek()!=EOF) throw runtime_error("menu length mismatch");
        evaluate(out,m,year,pol,0,Diagnostics(),full,mask);
    }
    for(size_t k=0;k<menus.size();++k)
        evaluate(out,m,year,"simulated",checkpoints[k],diagnoses[k],full,menus[k]);
}
