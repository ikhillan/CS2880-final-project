// Exhaust all strict complete 3-by-3 priority/rank profiles up to relabeling
// and all ordered display pairs. Directly enumerate stable partial assignments.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

constexpr int n = 3;
constexpr int graphs = 1 << (n*n);
using Row = std::array<int,n>;
using Profile = std::array<Row,n>;
using Count = std::uint64_t;
int count(int bits) { return __builtin_popcount(static_cast<unsigned>(bits)); }
int edge(int i, int j) { return 1 << (n*i+j); }
int rows(int mask) {
    int r = 0;
    for (int i=0; i<n; ++i) if ((mask >> (n*i)) & 7) r |= 1 << i;
    return r;
}
struct Partial {
    Row partner{}, at{-1,-1,-1};
    int edges = 0, matched = 0;
    std::array<int, 1<<n> seats{};
};
std::vector<Partial> enumerate() {
    std::vector<Partial> out;
    for (int x=-1;x<n;x++) for (int y=-1;y<n;y++) for (int z=-1;z<n;z++) {
        Partial m; m.partner = {x,y,z};
        bool ok = true;
        for (int i=0;i<n;i++) {
            int j = m.partner[i];
            if (j<0) continue;
            if (m.at[j]>=0) {ok=false;break;}
            m.at[j]=i;m.edges|=edge(i,j);m.matched|=1<<i;
        }
        if (!ok) continue;
        for (int f=0;f<(1<<n);f++) for (int i=0;i<n;i++)
            if ((f&(1<<i)) && m.partner[i]>=0) m.seats[f] |= 1<<m.partner[i];
        out.push_back(m);
    }
    assert(out.size()==34);
    return out;
}
int blockers(const Partial& m,const Profile& pref,const Profile& pri) {
    int mask=0;
    for (int i=0;i<n;i++) for(int j=0;j<n;j++) {
        int old=m.partner[i], previous=m.at[j];
        if ((old<0 || pref[i][j]<pref[i][old]) &&
            (previous<0 || pri[j][i]<pri[j][previous])) mask|=edge(i,j);
    }
    return mask;
}
int main() {
    const auto matchings=enumerate();
    Row id{0,1,2}, perm=id;
    std::vector<Row> orders;
    do {orders.push_back(perm);} while(std::next_permutation(perm.begin(),perm.end()));
    Count settings=0,pairs=0,safe_tests=0,sharp_nonzero=0,tighter_nonzero=0,baseline_rejection=0;
    for (auto a1:orders) for(auto a2:orders) for(auto p0:orders) for(auto p1:orders) for(auto p2:orders) {
        if (p0[1]>p0[2]) continue;
        ++settings;
        Profile pref{id,a1,a2}, pri{p0,p1,p2};
        std::vector<std::vector<Partial>> stable(graphs);
        std::vector<int> best(graphs,-1);
        for (int g=0;g<graphs;g++) {
            int best_rank=100;
            for (const auto& m:matchings) {
                if ((m.edges & ~g) || (blockers(m,pref,pri)&g)) continue;
                int sum=0;
                for(int i=0;i<n;i++) sum+=m.partner[i]<0 ? n : pref[i][m.partner[i]];
                if(sum<best_rank) {best_rank=sum;best[g]=int(stable[g].size());}
                stable[g].push_back(m);
            }
            assert(best[g]>=0);
        }
        // For fixed baseline matching and B, arithmetic is independent of the
        // particular A supporting that baseline. Count each supporting A.
        for (const auto& old:matchings) {
            const int blocking=blockers(old,pref,pri);
            int sources=0;
            for(int a=0;a<graphs;a++) if(!(old.edges & ~a) && !(a&blocking)) ++sources;
            if(!sources) continue;
            for(int b=0;b<graphs;b++) {
                const int f=rows((old.edges&~b)|(b&blocking));
                const auto& da=stable[b][best[b]];
                if(!f) {
                    for(int i=0;i<n;i++) {
                        if(old.partner[i]>=0) {
                           assert(da.partner[i]>=0 && pref[i][da.partner[i]]<=pref[i][old.partner[i]]);
                        } else assert(da.partner[i]<0);
                    }
                    safe_tests+=sources;
                } else baseline_rejection+=sources;
                for(const auto& out:stable[b]) {
                    const int overlap=count(old.seats[f]&out.seats[f]);
                    const int bound=count(f&(old.matched|out.matched))-overlap;
                    const int loss=count(old.matched&~out.matched);
                    const int gain=count(out.matched&~old.matched);
                    assert(loss<=bound && gain<=bound && bound<=count(f));
                    pairs+=sources;
                    if(bound>0 && std::max(loss,gain)==bound) sharp_nonzero+=sources;
                    if(std::max(loss,gain)>0 && bound<count(f)) tighter_nonzero+=sources;
                }
            }
        }
    }
    std::cout << "{\"symmetry_reduced_rank_profiles\":"<< settings
              <<",\"ordered_A_B_stable_outcome_pairs\":"<<pairs
              <<",\"baseline_display_pairs_with_safe_B_rank_checks\":"<<safe_tests
              <<",\"baseline_display_pairs_that_invalidate_the_old_assignment\":"<<baseline_rejection
              <<",\"positive_bounds_attained\":"<<sharp_nonzero
              <<",\"positive_turnover_cases_with_bound_strictly_below_flag_count\":"<<tighter_nonzero<<"}\n";
}
