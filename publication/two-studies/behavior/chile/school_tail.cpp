// Change only the position of the highest-ranked hidden option restored to both menus.
#define ADVICE_RECOVERY_LIBRARY
#include "school_recovery.cpp"

struct Arm {
    vector<int> edge_order; // Flattened applicant rows; -1 marks unused positions.
    vector<char> shown;
    vector<int> z;
    RecoveryCounts counts;
    int gains = 0, losses = 0, improved = 0, worsened = 0, named = 0;
    int new_in_hidden = 0, n_extended = 0, n_entitled_extended = 0, entitled_named = 0;
    int inc_programs = 0, dec_programs = 0, increased_seats = 0, decreased_seats = 0;
};

Arm run_arm(const Experiment& x, const vector<char>& base, const vector<int>& hidden,
            const vector<int>& baseline, const string& mode) {
    const Market& m=x.m;
    Arm a;
    a.shown=base;
    a.edge_order.assign(m.ne,-1);
    // Pass a newly ordered copy of the graph to the serial solver, then map each assigned
    // local edge back to the original submitted row for all outcome measurements.
    Market reordered=m;
    vector<char> compact(m.ne,0);
    for(int i=0;i<m.n;++i) {
        int ptr=m.at[i], entitlement=-1;
        const int extra=hidden[i];
        const bool full_recovery=mode=="all_last" || mode=="all_original";
        if(extra>=0 && mode!="binding") {
            if(full_recovery) fill(a.shown.begin()+m.at[i],a.shown.begin()+m.at[i+1],1);
            else a.shown[extra]=1;
            a.n_extended++;
            a.n_entitled_extended+=m.ret[i]>=0;
        }
        for(int e=m.at[i];e<m.at[i+1];++e) {
            if(!a.shown[e] || (mode=="all_last" && !base[e])) continue;
            if(mode=="last" || mode=="before_right") {
                if(e==extra) continue;
                if(mode=="before_right" && m.prog[e]==m.ret[i]) { entitlement=e;continue; }
            }
            a.edge_order[ptr++]=e;
        }
        if((mode=="last" || mode=="before_right") && extra>=0) a.edge_order[ptr++]=extra;
        if(entitlement>=0) a.edge_order[ptr++]=entitlement;
        if(mode=="all_last") for(int e=m.at[i];e<m.at[i+1];++e) if(!base[e]) a.edge_order[ptr++]=e;
        for(int j=m.at[i];j<ptr;++j) {
            const int e=a.edge_order[j];
            reordered.prog[j]=m.prog[e]; reordered.priority[j]=m.priority[e];compact[j]=1;
        }
        assert(ptr<=m.at[i+1]);
    }
    const auto result=reordered.da(compact);
    reordered.verify(result,compact);
    vector<int> occupancy(m.m),old_occupancy(m.m);
    a.z.resize(m.n);
    for(int i=0;i<m.n;++i) {
        const int old=baseline[i];
        const int e=result[i]<0?-1:a.edge_order[result[i]];
        a.z[i]=e;
        a.gains+=old<0 && e>=0;
        a.losses+=old>=0 && e<0;
        a.improved+=old>=0 && e>=0 && e<old;
        a.worsened+=old>=0 && e>=0 && e>old;
        a.named+=e>=0 && !base[e];
        a.new_in_hidden+=old<0 && e>=0 && !base[e];
        a.entitled_named+=m.ret[i]>=0 && e>=0 && !base[e];
        if(e>=0) occupancy[m.prog[e]]++;
        if(old>=0) old_occupancy[m.prog[old]]++;
        if(mode=="last" || mode=="all_last") {
            if(old<0 && e>=0) assert(!base[e]);
            if(e>=0 && base[e] && old>=0) assert(e>=old);
            if(m.ret[i]>=0 && e>=0) assert(base[e]);
        }
    }
    for(int j=0;j<m.m;++j) {
        const int diff=occupancy[j]-old_occupancy[j];
        a.inc_programs+=diff>0; a.dec_programs+=diff<0;
        a.increased_seats+=max(0,diff);a.decreased_seats+=max(0,-diff);
    }
    if(mode=="last" || mode=="all_last") assert(a.dec_programs==0);
    a.counts=x.measure(a.shown,a.z);
    assert(a.gains-a.losses==a.increased_seats-a.decreased_seats);
    return a;
}

int main(int argc,char** argv) {
    if(argc!=4){ cerr<<"Usage: school_tail NUMERIC YEAR OUT\n";return 2; }
    const string y=argv[2],prefix=string(argv[1])+"/"+y,out=string(argv[3])+"/school-tail-"+y;
    Experiment e(prefix);
    const auto baseline_a=e.m.da(e.a), baseline_b=e.m.da(e.b);
    ofstream arm(out+"-arms.csv"),diff(out+"-comparison.csv");
    if(!arm || !diff) throw runtime_error("cannot write aggregates");
    arm<<"year,response,policy,n,history,full,placed,first,named,edges,extended,entitled_extended,placed_gain,placed_loss,improved_among_both,worsened_among_both,received_recovered,newly_placed_at_recovered,entitled_at_recovered,more_occupied_programs,less_occupied_programs,more_occupied_seats,less_occupied_seats,assignment_hash\n";
    diff<<"year,response,n,same_set,same_order,history,full,placed,first,gains,losses,same_set_gains,same_set_losses,same_order_gains,same_order_losses\n";
    for(const string& mode:{"binding","in_order","before_right","last","all_last","all_original"}) {
        const auto a=run_arm(e,e.a,e.hidden_a,baseline_a,mode);
        const auto b=run_arm(e,e.b,e.hidden_b,baseline_b,mode);
        if(mode=="all_original") assert(a.z==e.full_partner && b.z==e.full_partner);
        int same=0,ordered=0,gain=0,loss=0,sg=0,sl=0,og=0,ol=0;
        for(int i=0;i<e.m.n;++i) {
            const int start=e.m.at[i], end=e.m.at[i+1];
            const bool same_set=equal(a.shown.begin()+start,a.shown.begin()+end,b.shown.begin()+start);
            const bool same_order=equal(a.edge_order.begin()+start,a.edge_order.begin()+end,b.edge_order.begin()+start);
            if(same_order) assert(same_set);
            const bool g=a.z[i]<0 && b.z[i]>=0, l=a.z[i]>=0 && b.z[i]<0;
            same+=same_set;ordered+=same_order;gain+=g;loss+=l;
            sg+=same_set&&g;sl+=same_set&&l;og+=same_order&&g;ol+=same_order&&l;
        }
        for(const string& pol:{"blind","rank"}) {
            const auto& v=pol=="blind"?a:b;
            arm<<y<<','<<mode<<','<<pol<<','<<e.m.n;write_counts(arm,v.counts);
            arm<<','<<v.n_extended<<','<<v.n_entitled_extended<<','<<v.gains<<','<<v.losses<<','<<v.improved<<','<<v.worsened
               <<','<<v.named<<','<<v.new_in_hidden<<','<<v.entitled_named<<','<<v.inc_programs<<','<<v.dec_programs
               <<','<<v.increased_seats<<','<<v.decreased_seats<<','<<e.digest(v.z)<<'\n';
        }
        diff<<y<<','<<mode<<','<<e.m.n<<','<<same<<','<<ordered<<','<<b.counts.shown_official-a.counts.shown_official
            <<','<<b.counts.shown_full-a.counts.shown_full<<','<<b.counts.placed-a.counts.placed<<','<<b.counts.first-a.counts.first
            <<','<<gain<<','<<loss<<','<<sg<<','<<sl<<','<<og<<','<<ol<<'\n';
    }
    return 0;
}
