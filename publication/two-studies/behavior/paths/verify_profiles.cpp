// Check both injective paths on every pair of stable outcomes across all lists in 3-by-2 markets.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using Applicants = std::array<int,3>;
using Schools = std::array<int,2>;
using Priorities = std::array<Applicants,2>;
// Each array is the rank of school 0, school 1, and being outside (=2).
using List = std::array<int,3>;
using Profile = std::array<List,3>;
constexpr int outside = 2;

struct Outcome {
    Profile ranks;
    Applicants assignment;
};

bool stable(const Applicants& assignment, const Profile& ranks,
            const Priorities& priorities, const Schools& capacity) {
    Schools occupied{}, worst{-1,-1};
    for (int i=0; i<3; ++i) {
        const int school = assignment[i];
        if (school == outside) continue;
        if (ranks[i][school] >= ranks[i][outside] || ++occupied[school] > capacity[school]) return false;
        worst[school] = std::max(worst[school], priorities[school][i]);
    }
    for (int i=0; i<3; ++i)
        for (int j=0; j<2; ++j) {
            if (ranks[i][j] >= ranks[i][assignment[i]]) continue;
            if (occupied[j] < capacity[j] || priorities[j][i] < worst[j]) return false;
        }
    return true;
}

int verify_paths(const Outcome& a, const Outcome& b, const Priorities& priorities) {
    Applicants next{-1,-1,-1};
    for (int school=0; school<2; ++school) {
        Applicants before{}, after{};
        int left=0, right=0;
        for (int i=0; i<3; ++i) {
            if (a.assignment[i] == b.assignment[i]) continue;
            if (a.assignment[i] == school) before[left++] = i;
            if (b.assignment[i] == school) after[right++] = i;
        }
        for (int k=0; k<std::min(left,right); ++k) next[before[k]] = after[k];
    }
    int sources=0;
    std::array<bool,3> seen{};
    for(int i=0; i<3; ++i) {
        if (a.assignment[i] == outside || b.assignment[i] != outside) continue;
        ++sources;
        int current=i;
        while (true) {
            assert(!seen[current]);
            seen[current]=true;
            const int old=a.assignment[current], now=b.assignment[current];
            assert(a.ranks[current][old] < a.ranks[current][now]);
            if(b.ranks[current][now] < b.ranks[current][old]) break;
            assert(old != outside && next[current]>=0);
            const int h=next[current];
            assert(b.assignment[h]==old && priorities[old][h]<priorities[old][current]);
            current=h;
        }
    }
    return sources;
}

int main() {
    // Omitted schools are tied below the outside option. Their ordering never enters a test.
    const std::array<List,5> lists = {{{3,3,2},{0,3,2},{3,0,2},{0,1,2},{1,0,2}}};
    std::int64_t pairs=0, changed_graph=0, turnover=0, both_tight=0;
    for(int c0=0; c0<=3; ++c0) for(int c1=0; c1<=3; ++c1) {
        const Schools capacity{c0,c1};
        Applicants p0{0,1,2};
        do {
            Applicants p1{0,1,2};
            do {
                const Priorities priorities{p0,p1};
                std::vector<Outcome> outcomes;
                for(const List& r0:lists) for(const List& r1:lists) for(const List& r2:lists) {
                    const Profile ranks{r0,r1,r2};
                    for(int code=0; code<27; ++code) {
                        const Applicants assignment{code%3, (code/3)%3, code/9};
                        if(stable(assignment,ranks,priorities,capacity)) outcomes.push_back({ranks,assignment});
                    }
                }
                for(const Outcome& a:outcomes) for(const Outcome& b:outcomes) {
                    ++pairs;
                    bool graphs_differ=false;
                    int w=0;
                    for(int i=0;i<3;++i) {
                        const int x=a.assignment[i], y=b.assignment[i];
                        w += a.ranks[i][x] < a.ranks[i][y] && b.ranks[i][y] < b.ranks[i][x];
                        for (int j=0; j<2; ++j)
                            graphs_differ |= (a.ranks[i][j]<2) != (b.ranks[i][j]<2);
                    }
                    const int loss=verify_paths(a,b,priorities), gain=verify_paths(b,a,priorities);
                    assert(std::max(loss,gain)<=w);
                    changed_graph+=graphs_differ;
                    turnover+=loss>0||gain>0;
                    both_tight+=w>0 && loss==w && gain==w;
                }
            } while(std::next_permutation(p1.begin(),p1.end()));
        } while(std::next_permutation(p0.begin(),p0.end()));
    }
    assert(both_tight>0 && changed_graph>0);
    std::cout<<"{\"applicants\":3,\"programs\":2,\"capacity_vectors\":16,\"priority_profiles\":36,"
             <<"\"ordered_stable_comparisons\":"<<pairs<<",\"unequal_graph_pairs\":"<<changed_graph
             <<",\"positive_turnover\":"<<turnover<<",\"positive_both_tight\":"<<both_tight<<"}\n";
}
