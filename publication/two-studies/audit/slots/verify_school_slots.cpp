// Independent row-by-row implementation of safe score selection and oracle counts.
// Build with the original publication/research/observed/school as an include directory.
#define main original_school_replay_main
#include "replay.cpp"
#undef main
#include <functional>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    Market m(argv[1]);
    const int year = stoi(argv[2]);
    mt19937_64 rng(28311 + year*1000);
    vector<uint64_t> tie(m.ne);
    generate(tie.begin(), tie.end(), [&]{ return rng(); });
    vector<char> all(m.ne,1);
    auto a = m.menu(m.blind,tie,5,"blind");
    auto b = m.menu(m.rank,tie,5,"rank");
    auto old = m.da(a), full = m.da(all), candidate = m.da(b);
    vector<int> used(m.m), worst(m.m,-1);
    for (int i=0; i<m.n; ++i) if (old[i]>=0) {
        const int e=old[i], j=m.prog[e];
        ++used[j];
        worst[j]=max(worst[j],m.priority[e]);
    }
    cout << "year,pool,full_slot_oracle_gain,history_slot_oracle_gain,full_slot_score_gain,history_slot_score_gain,improved_assignments,full_control_placement_gain_vs_b,history_full_control_gain_vs_b\n";
    for (bool entire : {false,true}) {
        vector<char> repair(m.ne,0), control(m.ne,0);
        long long missing_full_safe=0, missing_history_safe=0;
        long long full_net=0, history_net=0, control_history_net=0;
        for (int i=0; i<m.n; ++i) {
            vector<int> optional, optional_control;
            int q=0,nreq=0,ncontrol=0,historical=-1;
            vector<char> eligible(m.at[i+1]-m.at[i],0);
            for (int e=m.at[i]; e<m.at[i+1]; ++e) {
                int j=m.prog[e];
                if (j==m.official[i]) historical=e;
                q+=a[e];
                if (j==m.ret[i] || e==full[i]) {control[e]=1; ++ncontrol;}
                else optional_control.push_back(e);
                if (!entire && !a[e] && !b[e]) continue;
                if ((old[i]<0 || e<old[i]) && (used[j]<m.cap[j] || m.priority[e]<worst[j])) continue;
                eligible[e-m.at[i]]=1;
                if (j==m.ret[i] || e==old[i]) {repair[e]=1; ++nreq;}
                else optional.push_back(e);
            }
            auto preference=[&](int l,int r) {
                if (m.rank[l]!=m.rank[r]) return m.rank[l]>m.rank[r];
                if (b[l]!=b[r]) return b[l]>b[r];
                if (a[l]!=a[r]) return a[l]>a[r];
                return l<r;
            };
            sort(optional.begin(),optional.end(),preference);
            sort(optional_control.begin(),optional_control.end(),preference);
            assert(q>=nreq && q-nreq<=int(optional.size()));
            assert(q>=ncontrol && q-ncontrol<=int(optional_control.size()));
            for (int k=0; k<q-nreq; ++k) repair[optional[k]]=1;
            for (int k=0; k<q-ncontrol; ++k) control[optional_control[k]]=1;
            auto obtainable=[&](int e){
                return e>=0 && !a[e] && eligible[e-m.at[i]] && q>nreq;
            };
            missing_full_safe+=obtainable(full[i]);
            missing_history_safe+=obtainable(historical);
            if(full[i]>=0) full_net+=repair[full[i]]-a[full[i]];
            if(historical>=0) {
                history_net+=repair[historical]-a[historical];
                control_history_net+=control[historical]-b[historical];
            }
        }
        m.verify(old,repair);
        m.verify(full,control);
        auto after = m.da(repair), control_after=m.da(control);
        m.verify(after,repair);
        m.verify(control_after,control);
        long long better=0,control_net_placed=0;
        for(int i=0;i<m.n;++i) {
            assert((after[i]>=0)==(old[i]>=0));
            assert((control_after[i]>=0)==(full[i]>=0));
            if(old[i]>=0) {
                assert(after[i]<=old[i]);
                better+=after[i]<old[i];
            }
            control_net_placed+=(control_after[i]>=0)-(candidate[i]>=0);
        }
        cout<<year<<","<<(entire?"all":"union")<<","<<missing_full_safe<<","<<missing_history_safe<<","<<full_net<<","<<history_net<<","<<better<<","<<control_net_placed<<","<<control_history_net<<"\n";
    }
}
