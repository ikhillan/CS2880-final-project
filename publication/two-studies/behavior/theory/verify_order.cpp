// Enumerate fixed graphs, orders and all stable assignments directly, without DA.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
using Triple = std::array<int,3>;
using Profile = std::array<Triple,3>;
struct Outcome { Triple partner; int profile; };
int main() {
    std::vector<Triple> perms;
    Triple item{0,1,2};
    do { perms.push_back(item); } while (std::next_permutation(item.begin(),item.end()));
    int64_t comparisons=0, turnover=0, tight=0, both_tight=0;
    for (int graph=0;graph<512;++graph) {
        std::vector<Profile> applicant_orders(1);
        for (int i=0;i<3;++i) {
            std::vector<Profile> expanded;
            std::vector<int> options;
            for (int j=0;j<3;++j) if (graph & (1<<(3*i+j))) options.push_back(j);
            do {
                for (auto p : applicant_orders) {
                    p[i].fill(9);
                    for (size_t r=0;r<options.size();++r) p[i][options[r]]=int(r);
                    expanded.push_back(p);
                }
            } while (std::next_permutation(options.begin(),options.end()));
            applicant_orders.swap(expanded);
        }
        for (const Triple cap : std::vector<Triple>{{1,1,1},{1,2,1},{2,0,1}}) {
            std::vector<Triple> feasible;
            for (int code=0;code<64;++code) {
                Triple match{},counts{}; bool valid=true; int key=code;
                for(int i=0;i<3;++i) {
                    int j=(key%4)-1; key/=4; match[i]=j;
                    if(j>=0 && (!(graph&(1<<(3*i+j))) || ++counts[j]>cap[j])) valid=false;
                }
                if(valid) feasible.push_back(match);
            }
            // Relabeling applicants permits fixing the first program's full priority order.
            for(const auto& p1:perms) for(const auto& p2:perms) {
                const Profile pri{perms[0],p1,p2}; // priority numbers, lower is better
                std::vector<Outcome> stable;
                for(const auto& m:feasible) {
                    Triple occupancy{}, worst{-1,-1,-1};
                    for(int i=0;i<3;++i) if(m[i]>=0) {
                        ++occupancy[m[i]]; worst[m[i]]=std::max(worst[m[i]],pri[m[i]][i]);
                    }
                    for(size_t k=0;k<applicant_orders.size();++k) {
                        bool blocked=false; const auto& rank=applicant_orders[k];
                        for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
                            if(rank[i][j] < (m[i]>=0 ? rank[i][m[i]] : 8) &&
                              (occupancy[j]<cap[j] || pri[j][i]<worst[j])) blocked=true;
                        }
                        if(!blocked) stable.push_back({m,int(k)});
                    }
                }
                for(const auto& a:stable) for(const auto& b:stable) {
                    int loss=0,gain=0,flips=0;
                    for(int i=0;i<3;++i) {
                        const int j=a.partner[i], h=b.partner[i];
                        loss+=j>=0 && h<0; gain+=j<0 && h>=0;
                        if(j>=0 && h>=0) {
                            const auto& ra=applicant_orders[a.profile][i];
                            const auto& rb=applicant_orders[b.profile][i];
                            flips+=ra[j]<ra[h] && rb[h]<rb[j];
                        }
                    }
                    ++comparisons; if(loss||gain) ++turnover;
                    assert(loss<=flips && gain<=flips);
                    tight+=flips>0 && std::max(loss,gain)==flips;
                    both_tight+=flips>0 && loss==flips && gain==flips;
                }
            }
        }
    }
    assert(both_tight>0);
    std::cout<<"{\"applicants\":3,\"programs\":3,\"graphs\":512,\"capacity_vectors\":[[1,1,1],[1,2,1],[2,0,1]],\"program_priority_profiles_per_graph\":36,\"ordered_stable_comparisons\":"<<comparisons<<",\"positive_turnover\":"<<turnover<<",\"positive_tight\":"<<tight<<",\"both_directions_tight_positive\":"<<both_tight<<"}\n";
}
